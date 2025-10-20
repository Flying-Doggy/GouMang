# -*- coding: utf-8 -*-
"""
==============================================================================
## Gene Segments Unification and Clustering Tool ##
==============================================================================

This script provides a comprehensive pipeline for identifying and reconstructing
conserved gene clusters from syntenic gene segments across multiple species.
It takes raw gene segments and orthology information as input, performs
clustering to group related segments, and reconstructs a consensus gene order
for each resulting cluster.

------------------------------------------------------------------------------
Core Workflow:
------------------------------------------------------------------------------
1.  **Data Parsing**: The script reads two main input files:
    - A CSV file containing gene segments for each species.
    - A TSV file of orthogroups (e.g., from OrthoFinder) which maps genes
      across different species.

2.  **Gene ID Standardization**: All gene identifiers within the segments are
    converted to their corresponding orthogroup IDs (OGs). This normalization
    is crucial for comparing segments from different species. Tandemly
    duplicated genes (consecutive identical OGs) are removed.

3.  **Phased Clustering**: A two-phase hierarchical clustering approach is used
    to group segments into putative conserved clusters:
    - **Phase 1 (Core Clustering)**: A subset of the longest, most informative
      segments is selected. A similarity matrix is built based on their gene
      content, and agglomerative clustering is performed to form stable,
      core clusters.
    - **Phase 2 (Iterative Assignment)**: The remaining shorter segments are
      iteratively assigned to the most similar core cluster, provided their
      similarity score exceeds a defined threshold.

4.  **Gene Order Reconstruction**: For each final cluster, a consensus gene order
    is reconstructed. This is achieved by:
    - First, using the longest segments within the cluster to build a scaffold
      gene order via topological sorting.
    - Then, inserting the remaining genes into the scaffold based on their
      adjacency relationships, derived from a weighted neighbor matrix of all
      segments in the cluster.

5.  **Output Generation**: The final clusters, along with their reconstructed
    gene order, are written to a BED (Browser Extensible Data) format file.
    This format is suitable for visualization in genome browsers or for use in
    downstream synteny analysis.

------------------------------------------------------------------------------
Command-Line Usage:
------------------------------------------------------------------------------
The script is executed from the command line.

Basic Usage:
$ python segments_unifier_fully_commented.py --segments <segments.csv> --orthogroups <orthogroups.tsv>

Example with optional arguments:
$ python segments_unifier_fully_commented.py \
    --segments my_species_segments.csv \
    --orthogroups Orthogroups.tsv \
    --output conserved_clusters.bed \
    --phased-ratio 0.25 \
    --log-level INFO

Argument Details:
  --segments       [Required] Path to the input CSV file containing gene segments.
  --orthogroups    [Required] Path to the input TSV file of orthogroups.
  --output         [Optional] Path for the output BED file (Default: SegClust.bed).
  --phased-ratio   [Optional] The fraction of longest segments to use for the
                   initial clustering phase (Default: 0.2).
  --clade          [Optional] Path to a clade annotation file for species.
  --log-level      [Optional] Set the logging level (DEBUG, INFO, WARNING, ERROR)
                   (Default: INFO).

------------------------------------------------------------------------------
"""
import collections
import logging
import math
import argparse
import sys
import ast  # For safely evaluating string representations of lists

import numpy as np
import pandas as pd

from typing import Dict, List, Set, Literal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def levenshtein_distance(seq1: list, seq2: list,
                         gap: int = 1, mismatch: int = 1) -> int:
    """
    Calculates the Levenshtein distance (edit distance) between two sequences.
    This metric quantifies the minimum number of single-element edits (insertions,
    deletions, or substitutions) required to change one sequence into the other.
    The implementation uses a dynamic programming approach.

    Args:
        seq1 (list): The first sequence.
        seq2 (list): The second sequence.
        gap (int, optional): The penalty for an insertion or deletion. Defaults to 1.
        mismatch (int, optional): The penalty for a substitution. Defaults to 1.

    Returns:
        int: The Levenshtein distance between the two sequences.
    """
    # Optimization: ensure seq2 is the shorter sequence to reduce space complexity.
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1, gap, mismatch)

    # DP initialization: the first row represents the cost of converting an empty
    # string to the prefixes of seq2.
    cur = [gap * i for i in range(len(seq2) + 1)]

    # Iterate through each element of the longer sequence (seq1).
    for i in seq1:
        last = cur
        cur = [last[0] + gap]  # Cost of deleting the first char of seq1
        for tmp_idx, j in enumerate(seq2):
            # Cost is 0 if characters are the same, otherwise it's the mismatch penalty.
            cost = 0 if i == j else mismatch
            # The new cost is the minimum of insertion, deletion, or substitution.
            cur.append(min(cur[tmp_idx] + gap,          # Insertion
                           last[tmp_idx + 1] + gap,      # Deletion
                           last[tmp_idx] + cost))      # Substitution/Match
    return cur[-1]

def seq_order_similarity(seq1: list, seq2: list) -> float:
    """
    Calculates the order-based similarity between two sequences.
    The similarity is derived by normalizing the Levenshtein distance by the
    length of the longer sequence, yielding a score between 0 and 1.

    Args:
        seq1 (list): The first sequence.
        seq2 (list): The second sequence.

    Returns:
        float: A similarity score where 1 means identical order and 0 means
               maximum dissimilarity.
    """
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 1.0  # Two empty sequences are considered perfectly similar.
    lev_dist = levenshtein_distance(seq1, seq2)
    return 1 - (lev_dist / max_len)

def seq_content_similarity(seq1: list, seq2: list) -> float:
    """
    Calculates the content-based similarity using the Overlap Coefficient.
    This is calculated as the size of the intersection of the two sets of elements
    divided by the size of the smaller of the two sets. It measures the degree
    to which one sequence is a subset of the other.

    Args:
        seq1 (list): The first sequence (or any iterable).
        seq2 (list): The second sequence (or any iterable).

    Returns:
        float: A content similarity score between 0 and 1.
    """
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    
    if min_len == 0:
        return 0.0
    return intersection / min_len

def compare_block_obscure_order(query_block: list, ref_block: list, overlap_genes: set) -> str:
    """
    Infers the relative positional order of two partially overlapping gene blocks.
    This heuristic function combines two scores:
    1.  Gap Score: Measures the asymmetry of non-overlapping genes at the ends.
    2.  Trend Score: Measures the average positional shift of the overlapping genes.
    A combined score determines if the query block is likely "ahead of" or
    "behind" the reference block.

    Args:
        query_block (list): The query gene segment.
        ref_block (list): The reference gene segment.
        overlap_genes (set): The set of genes common to both blocks.

    Returns:
        str: A string indicating the relative order: 'query_block is ahead of ref block',
             'query_block is behind of ref block', or 'cannot distinguish the order'.
    """
    if not overlap_genes:
        return 'cannot distinguish the order'
    
    # Calculate the length of the non-overlapping "head" of the query block.
    head_mis = 0
    for gene in query_block:
        if gene in overlap_genes: break
        head_mis += 1
            
    # Calculate the length of the non-overlapping "tail" of the query block.
    tail_mis = 0
    for gene in reversed(query_block):
        if gene in overlap_genes: break
        tail_mis += 1
    
    # Pre-calculate gene positions for efficient lookup.
    query_pos = {gene: idx for idx, gene in enumerate(query_block)}
    ref_pos = {gene: idx for idx, gene in enumerate(ref_block)}
    
    # Calculate the positional differences for all overlapping genes.
    pos_diff = [query_pos[gene] - ref_pos[gene] for gene in overlap_genes]
    
    positive_count = sum(1 for d in pos_diff if d > 0)
    negative_count = sum(1 for d in pos_diff if d < 0)
    
    # Combine metrics into a final weighted score.
    len_query = len(query_block)
    gap_ratio = (head_mis - tail_mis) / len_query if len_query > 0 else 0
    min_block_len = min(len(query_block), len(ref_block))
    overlap_ratio = len(overlap_genes) / min_block_len if min_block_len > 0 else 0
    
    weight = overlap_ratio
    gap_score = gap_ratio * weight
    trend_score = (positive_count - negative_count) / len(pos_diff) * weight if pos_diff else 0
    
    combined_score = (gap_score + trend_score) / 2
    
    # Determine order based on the final score.
    if combined_score > 0.05:
        return 'query_block is ahead of ref block'
    elif combined_score < -0.05:
        return 'query_block is behind of ref block'
    else:
        return 'cannot distinguish the order'

def order_segments(query_segs: List[List[str]]) -> List[int]:
    """
    Determines a linear order for a list of gene segments.
    This function constructs a directed acyclic graph (DAG) where each segment is a node
    and an edge from A to B exists if A is determined to be "ahead of" B.
    It then performs a topological sort (Kahn's Algorithm) to find a valid linear ordering.

    Args:
        query_segs (List[List[str]]): A list of gene segments to be ordered.

    Returns:
        List[int]: A list of indices representing the sorted order of the input segments.
    """
    n = len(query_segs)
    if n == 0:
        return []
    graph: Dict[int, Set[int]] = collections.defaultdict(set)
    indegree: Dict[int, int] = {i: 0 for i in range(n)}

    # Build the directed graph based on pairwise comparisons.
    for i in range(n):
        for j in range(i + 1, n):
            overlap = set(query_segs[i]) & set(query_segs[j])
            if not overlap:
                continue

            order = compare_block_obscure_order(query_segs[i], query_segs[j], overlap)

            if order == 'query_block is ahead of ref block' and j not in graph[i]:
                graph[i].add(j)
                indegree[j] += 1
            elif order == 'query_block is behind of ref block' and i not in graph[j]:
                graph[j].add(i)
                indegree[i] += 1
    
    # Kahn's Algorithm for Topological Sorting.
    # Initialize the queue with all nodes having an in-degree of 0.
    queue = collections.deque([i for i in range(n) if indegree[i] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Sort neighbors to ensure deterministic output for the same graph.
        for neighbor in sorted(list(graph[node])):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # Handle cases where the graph contains cycles (not a perfect DAG).
    if len(topo_order) != n:
        logging.warning("Detected cyclic dependencies among segments. Appending remaining nodes.")
        remaining_nodes = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining_nodes)

    return topo_order

def construct_neighbor_mat(segs: List[List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Constructs a weighted adjacency matrix for genes based on their co-occurrence in segments.
    The matrix (represented as a nested dictionary) stores a score for each pair of
    genes that appear consecutively in any segment. The score is weighted by the
    logarithm of the segment's length.

    Args:
        segs (List[List[str]]): A list of gene segments.

    Returns:
        Dict[str, Dict[str, int]]: The neighbor matrix, e.g., {geneA: {geneB: score}}.
    """
    neighbor_mat = collections.defaultdict(lambda: collections.defaultdict(int))
    for seg in segs:
        seg_len = len(seg)
        if seg_len <= 1:
            continue
        # Weight is based on log(length) to give longer segments more influence.
        weight = int(math.log10(seg_len + 1)) + 1
        for i in range(seg_len - 1):
            neighbor_mat[seg[i]][seg[i+1]] += weight
    return neighbor_mat

def Agglomerative_Cluster_similarity_matrix(similarity_matrix: np.ndarray,
                                            n_clusters: int = None,
                                            linkage: str = 'average') -> tuple:
    """
    Performs agglomerative hierarchical clustering on a symmetric similarity matrix.

    Args:
        similarity_matrix (np.ndarray): An NxN symmetric matrix where higher values
                                        indicate greater similarity.
        n_clusters (int, optional): The target number of clusters. If None, the optimal
                                    number is determined using the silhouette score.
                                    Defaults to None.
        linkage (str, optional): The linkage criterion to use ('ward', 'complete',
                                 'average', 'single'). Defaults to 'average'.

    Returns:
        tuple: A tuple containing:
            - clusters (dict): A dictionary mapping cluster IDs to lists of item indices.
            - labels (np.ndarray): An array where each element is the cluster label
                                   for the corresponding item.
    """
    if not np.allclose(similarity_matrix, similarity_matrix.T):
        raise ValueError("Input matrix must be a symmetric similarity matrix")
    
    n = similarity_matrix.shape[0]
    if n <= 1:
        return {0: list(range(n))}, np.array([0] * n)
    
    # Convert similarity matrix to distance matrix (distance = 1 - similarity).
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Automatically determine the optimal number of clusters if not provided.
    if n_clusters is None:
        best_score = -2  # Initialize score to handle values down to -1.
        best_n = 2
        # Silhouette score is only meaningful for k between 2 and n-1.
        max_candidate = min(10, n - 1)
        
        if max_candidate >= 2:
            for k in range(2, max_candidate + 1):
                clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage=linkage)
                labels = clustering.fit_predict(distance_matrix)
                # Silhouette score requires at least 2 distinct clusters.
                if len(set(labels)) > 1:
                    score = silhouette_score(distance_matrix, labels, metric='precomputed')
                    if score > best_score:
                        best_score = score
                        best_n = k
            n_clusters = best_n
            logging.info(f"Automatically determined optimal number of clusters: {n_clusters} (silhouette score: {best_score:.3f})")
        else:
            n_clusters = 1  # If too few samples, group them into one cluster.
            logging.warning("Not enough samples to determine optimal clusters. Grouping into a single cluster.")

    # Perform the final clustering.
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage)
    labels = clustering.fit_predict(distance_matrix)
    
    # Format the output.
    clusters = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    return clusters, labels

class SegmentsUnifier:
    """
    The main controller class that orchestrates the entire segment unification
    and clustering pipeline.
    """
    def __init__(self, segments_file: str, orthogroups_file: str):
        """
        Initializes the SegmentsUnifier instance by loading and parsing input files.

        Args:
            segments_file (str): Path to the gene segments CSV file.
            orthogroups_file (str): Path to the orthogroups TSV file.
        """
        logging.info(f"Reading segments file from: {segments_file}")
        self.segments_df = self.parse_segments_file(segments_file)
        logging.info(f"Loaded {len(self.segments_df)} segments.")
        
        logging.info(f"Reading orthogroups file from: {orthogroups_file}")
        self.taxon_gene_OG_dic = self.parse_orthogroups_file(orthogroups_file)
        logging.info(f"Loaded orthogroups for {len(self.taxon_gene_OG_dic)} taxa.")
        
        self.clade_dic = None
        self.taxon_clade_dic = None

    def parse_segments_file(self, segments_file: str) -> pd.DataFrame:
        """Safely parses the segments CSV file."""
        tmp_df = pd.read_csv(segments_file)
        # Use ast.literal_eval for safe parsing of stringified lists.
        tmp_df['genes'] = tmp_df['genes'].apply(ast.literal_eval)
        return tmp_df
    
    def parse_orthogroups_file(self, orthogroups_file: str, sp_limit: int = 4) -> Dict[str, Dict[str, str]]:
        """
        Parses the orthogroups file and builds a nested dictionary mapping:
        taxon -> gene_id -> orthogroup_id.
        """
        taxon_gene_OG_dic = collections.defaultdict(dict)
        with open(orthogroups_file, 'r') as f:
            header = f.readline().strip().split('\t')
            for line in f:
                parts = line.strip().split('\t')
                og_id = parts[0]
                # Filter out orthogroups present in too few species.
                if parts.count('') >= (len(header) - sp_limit):
                    continue
                
                for i in range(1, len(parts)):
                    taxon_name = header[i]
                    if not parts[i]: continue
                    genes = parts[i].split(', ')
                    for gene in genes:
                        if gene:
                            taxon_gene_OG_dic[taxon_name][gene] = og_id
        return taxon_gene_OG_dic
    
    def get_clade_dic(self, clade_path: str):
        """Parses an optional clade annotation file."""
        self.clade_dic = collections.defaultdict(list)
        self.taxon_clade_dic = {}
        with open(clade_path) as f:
            for line in f:
                taxon, genus, subfamily = line.strip().split('\t')
                self.clade_dic[genus].append(taxon)
                self.taxon_clade_dic[taxon] = genus
        logging.info(f"Loaded clade information for {len(self.taxon_clade_dic)} taxa.")

    def convert_gene_to_OG(self, taxon_name: str, gene_id: str) -> str:
        """Converts a single gene ID to its corresponding orthogroup ID."""
        gene_id = gene_id.replace(':', '_')
        return self.taxon_gene_OG_dic.get(taxon_name, {}).get(gene_id)
    
    def unify_segement_to_OG(self, segment: List[str], taxon_name: str, rm_tandem: bool = True) -> List[str]:
        """
        Converts a list of gene IDs into a list of orthogroup IDs.
        Optionally removes consecutive identical OGs (tandem duplicates).
        """
        og_seg = []
        for gene in segment:
            og_id = self.convert_gene_to_OG(taxon_name=taxon_name, gene_id=gene)
            if og_id is None:
                continue
            if rm_tandem and og_seg and og_seg[-1] == og_id:
                continue
            og_seg.append(og_id)
        return og_seg
    
    def phased_segments_cluster(self, phased_ratio: float = 0.2):
        """
        Executes the main two-phase clustering workflow.
        """
        logging.info("Converting all gene segments to OG segments...")
        self.OG_segments = [
            self.unify_segement_to_OG(row['genes'], row['species'])
            for _, row in self.segments_df.iterrows()
        ]

        # Sort segment indices by length in descending order.
        seg_idx_by_length = sorted(range(len(self.OG_segments)), key=lambda i: len(self.OG_segments[i]), reverse=True)
        
        n_longest = int(len(seg_idx_by_length) * phased_ratio)
        # Ensure at least a few segments are used for the initial robust clustering.
        n_longest = max(n_longest, min(5, len(seg_idx_by_length)))

        phase1_indices = seg_idx_by_length[:n_longest]
        phase2_indices = seg_idx_by_length[n_longest:]
        
        # --- Phase 1: Initial clustering on longest segments ---
        logging.info(f"Phase 1: Clustering {len(phase1_indices)} longest segments.")
        if not phase1_indices:
            logging.warning("No segments for initial clustering. Aborting.")
            return

        # Build similarity matrix based on content similarity.
        sim_mat = np.zeros((n_longest, n_longest))
        for i in range(n_longest):
            for j in range(i + 1, n_longest):
                idx1 = phase1_indices[i]
                idx2 = phase1_indices[j]
                sim = seq_content_similarity(self.OG_segments[idx1], self.OG_segments[idx2])
                sim_mat[i, j] = sim_mat[j, i] = sim
        
        initial_clusters, _ = Agglomerative_Cluster_similarity_matrix(sim_mat)
        
        # Create SegmentsCluster objects from the initial clustering results.
        self.SegClusters = []
        self.clustered_indices = collections.defaultdict(list)
        for label, local_indices in initial_clusters.items():
            global_indices = [phase1_indices[i] for i in local_indices]
            self.clustered_indices[label].extend(global_indices)
            cluster_segs = [self.OG_segments[i] for i in global_indices]
            self.SegClusters.append(SegmentsCluster(cluster_segs))
        logging.info(f"Phase 1: Formed {len(self.SegClusters)} initial clusters.")

        # --- Phase 2: Assign remaining segments to existing clusters ---
        logging.info(f"Phase 2: Assigning {len(phase2_indices)} remaining segments to clusters.")
        unassigned_indices = set(phase2_indices)
        
        # Loop until no more assignments can be made.
        while unassigned_indices:
            assignments = []
            for seg_idx in unassigned_indices:
                max_sim = -1
                best_cluster_idx = -1
                for cluster_idx, cluster in enumerate(self.SegClusters):
                    sim = cluster.cal_seg_content_similar(self.OG_segments[seg_idx])
                    if sim > max_sim:
                        max_sim = sim
                        best_cluster_idx = cluster_idx
                if best_cluster_idx != -1:
                    assignments.append((max_sim, seg_idx, best_cluster_idx))
            
            if not assignments: break
                
            # Sort potential assignments by similarity score, descending.
            assignments.sort(key=lambda x: x[0], reverse=True)
            
            assigned_this_round = 0
            for max_sim, seg_idx, cluster_idx in assignments:
                # Assign if the segment is still unassigned and similarity is high.
                if seg_idx in unassigned_indices and max_sim > 0.8:
                    self.SegClusters[cluster_idx].add_seg(self.OG_segments[seg_idx])
                    self.clustered_indices[cluster_idx].append(seg_idx)
                    unassigned_indices.remove(seg_idx)
                    assigned_this_round += 1

            logging.info(f"Phase 2: Assigned {assigned_this_round} segments in this round. {len(unassigned_indices)} remaining.")
            # Break the loop if no assignments were made, to prevent infinite loops.
            if assigned_this_round == 0:
                logging.warning(f"Could not assign {len(unassigned_indices)} segments, similarity might be too low.")
                break

        # --- Final step: Reorder genes within each cluster ---
        logging.info("Reordering genes within each cluster...")
        for i, cluster in enumerate(self.SegClusters):
            cluster.pahsed_reorder_genes()
            logging.debug(f"Cluster {i} has {len(cluster.gene_order)} ordered genes.")

        # Update the original dataframe with cluster labels.
        self.segments_df['cluster'] = -1
        for clust_id, seg_indices in self.clustered_indices.items():
            self.segments_df.loc[seg_indices, 'cluster'] = clust_id
    
    def write_Segment_Cluster_bed(self, outfile: str = 'SegClust.bed'):
        """Writes the final clustered and ordered segments to a BED file."""
        logging.info(f"Writing segment clusters to BED file: {outfile}")
        with open(outfile, 'w') as f:
            for clust_id, seg_cluster in enumerate(self.SegClusters):
                if not seg_cluster.gene_order:
                    logging.warning(f"Cluster {clust_id} is empty and will be skipped.")
                    continue
                # BED format: chrom, start, end, name, score, strand
                for i, gene in enumerate(seg_cluster.gene_order):
                    f.write(f"{clust_id}\t{i}\t{i+1}\t{gene}\t0\t+\n")
        logging.info("BED file writing complete.")

class SegmentsCluster:
    """
    Represents a single cluster of gene segments. This class holds all segments
    belonging to the cluster and is responsible for reconstructing the consensus
    gene order.
    """
    def __init__(self, segments: List[List[str]]):
        """Initializes a cluster with a list of member segments."""
        self.segments = segments
        self.gene_content = self.merge_segments_content(segments)
        self.gene_order = []
        self.loc_dic = {}

    def merge_segments_content(self, segs: List[List[str]]) -> Set[str]:
        """Merges gene content from multiple segments into a single set."""
        gene_set = set()
        for seg in segs:
            gene_set.update(seg)
        return gene_set

    def add_seg(self, seg: List[str]):
        """Adds a new segment to the cluster and updates its gene content."""
        self.segments.append(seg)
        self.gene_content.update(seg)

    def cal_seg_content_similar(self, query_seg: List[str]) -> float:
        """Calculates the content similarity between a query segment and this cluster."""
        return seq_content_similarity(list(self.gene_content), query_seg)

    def get_longest_segs(self, limit_coverage: float = 0.9) -> List[int]:
        """
        Selects a subset of the longest segments that collectively cover a
        specified fraction of the cluster's total unique gene content.
        """
        sorted_indices = sorted(range(len(self.segments)), key=lambda i: len(self.segments[i]), reverse=True)
        
        total_genes = len(self.gene_content)
        if total_genes == 0:
            return []
            
        covered_genes = set()
        selected_indices = []
        for idx in sorted_indices:
            selected_indices.append(idx)
            covered_genes.update(self.segments[idx])
            if len(covered_genes) / total_genes >= limit_coverage:
                break
        return selected_indices

    def pahsed_reorder_genes(self):
        """
        Reconstructs the consensus gene order for the cluster in two phases.
        """
        if not self.gene_content:
            logging.warning("Attempted to reorder genes in an empty cluster.")
            return

        # --- Phase 1: Build a gene order scaffold from the longest segments ---
        top_indices = self.get_longest_segs(limit_coverage=0.9)
        ordered_genes = []
        if top_indices:
            top_segs = [self.segments[i] for i in top_indices]
            # Topologically sort these influential segments to get a robust order.
            ordered_seg_indices = order_segments(top_segs)
            
            # Create the scaffold by adding unique genes in the sorted order.
            seen_genes = set()
            for seg_idx in ordered_seg_indices:
                for gene in top_segs[seg_idx]:
                    if gene not in seen_genes:
                        ordered_genes.append(gene)
                        seen_genes.add(gene)
        
        # --- Phase 2: Insert remaining genes into the scaffold ---
        remaining_genes = self.gene_content - set(ordered_genes)
        if remaining_genes:
            neighbor_mat = construct_neighbor_mat(self.segments)
            
            insertions = []
            for gene in remaining_genes:
                best_pos = -1
                max_score = -1
                # Find the best insertion point by checking against all predecessors in the scaffold.
                for i in range(len(ordered_genes)):
                    pred_gene = ordered_genes[i]
                    # The score is the neighbor score between a gene in the scaffold and the gene to be inserted.
                    score = neighbor_mat.get(pred_gene, {}).get(gene, 0)
                    if score > max_score:
                        max_score = score
                        best_pos = i + 1  # Insert after the best predecessor.
                insertions.append((max_score, best_pos, gene))
            
            # Sort insertions by score (desc) and then position (asc) to be deterministic.
            insertions.sort(key=lambda x: (-x[0], x[1]))
            
            # Insert genes into the scaffold from the end to the beginning to avoid
            # invalidating position indices.
            for _, pos, gene in reversed(insertions):
                if pos != -1:
                    ordered_genes.insert(pos, gene)
                else:  # If no predecessor found, append to the end.
                    ordered_genes.append(gene)
                    
        self.gene_order = ordered_genes
        self.get_loc_dic()

    def get_loc_dic(self):
        """Updates the internal dictionary that maps genes to their final positions."""
        self.loc_dic = {gene: i for i, gene in enumerate(self.gene_order)}

def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unify and cluster gene segments based on orthogroups and sequence similarity.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--segments', required=True, help='Path to the segments CSV file.')
    parser.add_argument('--orthogroups', required=True, help='Path to the orthogroups file (TSV format).')
    parser.add_argument('--output', default='SegClust.bed', help='Output BED file path (default: SegClust.bed).')
    parser.add_argument('--phased-ratio', type=float, default=0.2, help='Ratio of longest segments for initial clustering (default: 0.2).')
    parser.add_argument('--clade', help='Optional path to clade annotation file.')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: INFO).')
    return parser.parse_args()

def main():
    """The main function to execute the script's workflow."""
    args = get_args()
    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        unifier = SegmentsUnifier(segments_file=args.segments, orthogroups_file=args.orthogroups)
        
        if args.clade:
            unifier.get_clade_dic(clade_path=args.clade)
        
        unifier.phased_segments_cluster(phased_ratio=args.phased_ratio)
        unifier.write_Segment_Cluster_bed(outfile=args.output)
        
        logging.info("Script completed successfully!")
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}. Please check the input file paths.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
