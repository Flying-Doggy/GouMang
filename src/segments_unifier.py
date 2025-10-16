import collections
import logging
import math
import argparse
import sys
import ast

import numpy as np
import pandas as pd

from typing import Dict, List, Set, Literal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def levenshtein_distance(seq1: list, seq2: list, gap: int = 1, mismatch: int = 1) -> int:
    """
    Compute the Levenshtein edit distance between two sequences.

    Parameters:
        seq1 (list): First sequence (list of comparable items).
        seq2 (list): Second sequence.
        gap (int): Penalty for an insertion or deletion operation.
        mismatch (int): Penalty for a substitution operation.

    Returns:
        int: Minimum edit cost (distance) to transform seq1 into seq2.

    Notes:
        - This implementation uses a rolling array to maintain O(min(len1, len2)) memory.
        - Elements are compared using equality (i != j) to account for mismatch cost.
    """
    cur = [gap * i for i in range(len(seq1) + 1)]

    for i in seq2:
        last = cur
        cur = [last[0] + gap]
        for tmp_idx, j in enumerate(seq1):
            # cost: insertion, deletion, or substitution
            cur.append(
                min(
                    cur[tmp_idx] + gap,
                    last[tmp_idx + 1] + gap,
                    last[tmp_idx] + (i != j) * mismatch,
                )
            )

    return cur[-1]


def seq_order_similarity(seq1: list, seq2: list) -> float:
    """
    Compute an order-based similarity between two sequences.

    This function converts the Levenshtein distance into a similarity score
    in the range (-inf, 1], and for typical inputs returns values in [0, 1].

    Parameters:
        seq1 (list): First sequence.
        seq2 (list): Second sequence.

    Returns:
        float: Similarity score where 1.0 means identical order.
    """
    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 1.0
    lev_dist = levenshtein_distance(seq1, seq2)
    return 1 - (lev_dist / max_len)


def seq_content_similarity(seq1: list, seq2: list) -> float:
    """
    Compute content similarity between two sequences using the Overlap Coefficient.

    Overlap Coefficient = |intersection(set(seq1), set(seq2))| / min(|set(seq1)|, |set(seq2)|)
    The coefficient equals 1 when the smaller set is entirely contained in the larger.

    Parameters:
        seq1 (list): First sequence.
        seq2 (list): Second sequence.

    Returns:
        float: Content similarity in [0.0, 1.0].
    """
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))

    # Avoid division by zero if one of the sequences is empty
    if min_len == 0:
        return 0.0

    return intersection / min_len


def compare_block_obscure_order(query_block: list, ref_block: list, overlap_genes: set):
    """
    Compare two blocks (lists of gene identifiers) where only partial overlap exists
    and try to infer whether query_block is positioned ahead or behind ref_block.

    Parameters:
        query_block (list): Gene list for query block.
        ref_block (list): Gene list for reference block.
        overlap_genes (set): Set of gene IDs present in both blocks.

    Returns:
        str: One of the following labels:
            - 'query_block is ahead of ref block'
            - 'query_block is behind of ref block'
            - 'cannot distinguish the order'

    Rationale:
        - The function estimates how many genes at head/tail are non-overlapping,
          computes positional differences for overlapping genes and combines
          gap/trend signals into a single score to decide relative order.
    """
    if not overlap_genes:
        return 'cannot distinguish the order'

    # Count unmatched genes at the head of query_block
    head_mis = 0
    for i in query_block:
        if i in overlap_genes:
            break
        head_mis += 1

    # Count unmatched genes at the tail of query_block
    tail_mis = 0
    for i in query_block[::-1]:
        if i in overlap_genes:
            break
        tail_mis += 1

    # Build positional maps for overlap genes
    query_pos = {gene: idx for idx, gene in enumerate(query_block)}
    ref_pos = {gene: idx for idx, gene in enumerate(ref_block)}

    pos_diff = []
    for gene in overlap_genes:
        pos_diff.append(query_pos[gene] - ref_pos[gene])

    positive_count = sum(1 for d in pos_diff if d > 0)
    negative_count = sum(1 for d in pos_diff if d < 0)

    len_query = len(query_block)
    gap_ratio = (head_mis - tail_mis) / len_query if len_query > 0 else 0
    overlap_ratio = (
        len(overlap_genes) / min(len(query_block), len(ref_block))
        if min(len(query_block), len(ref_block)) > 0
        else 0
    )

    # Weight signals by overlap ratio: higher overlap should give stronger signal
    weight = overlap_ratio
    gap_score = gap_ratio * weight
    trend_score = (
        (positive_count - negative_count) / len(pos_diff) * weight if pos_diff else 0
    )

    combined_score = (gap_score + trend_score) / 2

    # Thresholds chosen heuristically: small positive/negative combined_score required
    if combined_score > 0.05:
        return 'query_block is ahead of ref block'
    elif combined_score < -0.05:
        return 'query_block is behind of ref block'
    else:
        return 'cannot distinguish the order'


def order_segments(query_segs: List[List[str]]) -> List[int]:
    """
    Produce an ordering of input segments using pairwise "ahead/behind" signals.

    Implementation details:
        - Build a directed graph where an edge i->j means segment i should come before j.
        - Use the compare_block_obscure_order function to infer directions for overlapping segments.
        - Apply Kahn's topological sort to derive a global order.
        - If cycles exist, detect them and append remaining nodes to the resulting order.

    Parameters:
        query_segs (List[List[str]]): List of segments, where each segment is a list of gene/OG ids.

    Returns:
        List[int]: A topologically-consistent ordering of segment indices. If cycles exist, a partial
                   topological order is returned and remaining nodes are appended.
    """
    n = len(query_segs)
    graph: Dict[int, Set[int]] = collections.defaultdict(set)  # adjacency list
    indegree: Dict[int, int] = {i: 0 for i in range(n)}

    # Compare every pair of segments and add directed edges based on inferred relation
    for i in range(n):
        for j in range(i + 1, n):
            overlap = set(query_segs[i]) & set(query_segs[j])
            if not overlap:
                continue

            order = compare_block_obscure_order(query_segs[i], query_segs[j], overlap)

            if order == 'query_block is ahead of ref block':
                if j not in graph[i]:
                    graph[i].add(j)
                    indegree[j] += 1
            elif order == 'query_block is behind of ref block':
                if i not in graph[j]:
                    graph[j].add(i)
                    indegree[i] += 1

    # Kahn's algorithm for topological sorting
    queue = collections.deque([i for i in range(n) if indegree[i] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue.append(nei)

    if len(topo_order) != n:
        # If a cycle is detected, append remaining nodes (simple fallback strategy)
        print("Warning: Detected cyclic dependencies among segments. Partial order returned.")
        remaining_nodes = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining_nodes)

    return topo_order


def construct_neighbor_mat(segs: List[List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Construct a neighbor (adjacency) matrix from a list of segments.

    For every adjacent gene pair (g_i, g_{i+1}) in a segment, the function increments
    a weight for that directed pair. The weight is scaled by log10(length+1) so longer
    segments contribute more strongly.

    Parameters:
        segs (List[List[str]]): List of segments, each a list of gene/OG ids.

    Returns:
        Dict[str, Dict[str, int]]: Nested dictionary mapping predecessor -> successor -> weight.
    """
    neighbor_mat = collections.defaultdict(lambda: collections.defaultdict(int))
    for tmp_seg in segs:
        tmp_len = len(tmp_seg)
        # Assign at least weight 1 for short segments; use log10 scaled weight for longer segments
        weight = int(math.log10(tmp_len + 1)) if tmp_len > 1 else 1
        for i in range(tmp_len - 1):
            neighbor_mat[tmp_seg[i]][tmp_seg[i + 1]] += weight
    return neighbor_mat


def Agglomerative_Cluster_similarity_matrix(similarity_matrix, n_clusters=None, linkage='average'):
    """
    Cluster items given a symmetric similarity matrix using agglomerative clustering.

    Parameters:
        similarity_matrix (np.ndarray): NxN symmetric similarity matrix with higher = more similar.
        n_clusters (int | None): Desired number of clusters. If None, function will try several
                                 candidate cluster counts and choose the best via silhouette score.
        linkage (str): Linkage method for clustering. Passed to sklearn AgglomerativeClustering.

    Returns:
        clusters (dict): Mapping cluster_label -> list of indices in that cluster.
        labels (np.ndarray): Array of cluster labels for each element.
    """
    # Ensure symmetry
    if not np.allclose(similarity_matrix, similarity_matrix.T):
        raise ValueError("Input matrix must be a symmetric similarity matrix")

    n = similarity_matrix.shape[0]
    if n <= 1:
        return {0: [0]}, np.array([0])

    # Convert similarity to distance (assumes values are scaled in [0, 1])
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)

    # If cluster count not provided, find best by silhouette
    if n_clusters is None:
        best_score = -1
        best_n = 2
        max_candidate = min(10, n - 1)
        if max_candidate < 2:
            max_candidate = 2

        for candidate in range(2, max_candidate + 1):
            clustering = AgglomerativeClustering(n_clusters=candidate, metric='precomputed', linkage=linkage)
            labels = clustering.fit_predict(distance_matrix)
            score = silhouette_score(distance_matrix, labels, metric='precomputed')

            if score > best_score:
                best_score = score
                best_n = candidate

        n_clusters = best_n
        print(f"Automatically determined optimal number of clusters: {n_clusters} (silhouette score: {best_score:.3f})")

    # Perform clustering with chosen number of clusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage=linkage)
    labels = clustering.fit_predict(distance_matrix)

    clusters = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters, labels


class SegmentsUnifier:
    """
    Main class to read segments/orthogroups and perform unification + clustering.

    Attributes:
        segments_file (str): Path to CSV file describing segments (expects 'species' and 'genes' columns).
        orthogroups_file (str): Path to orthogroups file (tab-separated mapping of OG->genes by species).
        segments_df (pd.DataFrame): DataFrame loaded from segments_file.
        orthogroups_df / taxon_gene_OG_dic: Parsed orthogroup mapping stored as dict.
    """

    def __init__(self, segments_file: str, orthogroups_file: str):
        self.segments_file = segments_file
        self.orthogroups_file = orthogroups_file
        self.segments_df = self.parse_segments_file(segments_file)
        self.orthogroups_df = self.parse_orthogroups_file(orthogroups_file)

    def parse_segments_file(self, segments_file: str):
        """
        Parse a CSV file that contains gene segments for each record.

        Assumes the CSV has at least columns: ['species', 'genes'] where 'genes' is a
        Python-list-like string (e.g. "['gene1', 'gene2']"). Uses ast.literal_eval to
        safely parse the gene list string into a Python list.

        Returns:
            pd.DataFrame: DataFrame with a parsed 'genes' column (as Python lists).
        """
        tmp_df = pd.read_csv(segments_file)
        # Use safer ast.literal_eval instead of eval
        tmp_df['genes'] = tmp_df['genes'].apply(ast.literal_eval)
        return tmp_df

    def parse_orthogroups_file(self, orthogroups_file: str, sp_limit: int = 4) -> Dict[str, Dict[str, List[str]]]:
        """
        Parse a tab-separated orthogroups file into a mapping from taxon -> gene -> OG id.

        File format assumption:
            - First line is a header with species names starting from column 1.
            - Each subsequent line begins with OG id followed by columns per species listing comma-separated genes.

        Parameters:
            orthogroups_file (str): Path to orthogroups file.
            sp_limit (int): Minimum number of non-empty species columns required for an OG to be considered.
                            If a line contains `>= sp_limit` empty columns, the OG is skipped.

        Returns:
            dict: Nested dict mapping taxon_name -> gene_id -> OG_id
        """
        self.taxon_gene_OG_dic = collections.defaultdict(dict)
        with open(orthogroups_file, 'r') as f:
            head_line = f.readline().strip().split('\t')
            for line in f:
                line = line.strip().split('\t')

                if line.count('') >= sp_limit:  # exclude poorly conserved OGs
                    continue

                for i in range(1, len(line)):
                    # Each cell may contain comma+space separated gene ids
                    for tmp_gene in line[i].split(', '):
                        # Map gene -> OG id for the corresponding species (header)
                        self.taxon_gene_OG_dic[head_line[i]][tmp_gene] = line[0]

        return self.taxon_gene_OG_dic

    def get_clade_dic(self, clade_path: str = None):
        """
        Parse a clade annotation file and store clade mappings.

        Expected file format (tab-separated with three columns per line):
            taxon\tgenus\tsubfamily

        Stores two attributes:
            - self.clade_dic: mapping taxon -> list of subfamilies (or related clade values)
            - self.taxon_clade_dic: mapping subfamily -> genus (reverse mapping)

        Returns:
            tuple: (clade_dic, taxon_clade_dic)
        """
        clade_dic = collections.defaultdict(list)
        taxon_clade_dic = {}
        with open(file=clade_path) as f:
            while tmp_line := f.readline():
                tmp_line = tmp_line.strip().split('\t')
                clade_dic[tmp_line[0]].append(tmp_line[2])
                clade_dic[tmp_line[1]].append(tmp_line[2])
                taxon_clade_dic[tmp_line[2]] = tmp_line[1]

        # store the clade_dic and taxon_clade_dic for later use
        self.clade_dic = clade_dic
        self.taxon_clade_dic = taxon_clade_dic

        return clade_dic, taxon_clade_dic

    def convert_gene_to_OG(self, taxon_name: str, gene_id: str) -> str:
        """
        Map a raw gene identifier to its orthogroup id for a given taxon.

        Parameters:
            taxon_name (str): Species/taxon name as used in the orthogroups header.
            gene_id (str): Raw gene ID (colons are replaced with underscores before lookup).

        Returns:
            str | None: The OG id if mapping exists, otherwise None.
        """
        gene_id = gene_id.replace(':', '_')
        return self.taxon_gene_OG_dic[taxon_name].get(gene_id)

    def unify_segement_to_OG(self, segment: List[str], taxon_name: str, rm_tandem: bool = True) -> List[str]:
        """
        Convert a gene segment (list of gene identifiers) into a list of orthogroup IDs.

        Parameters:
            segment (List[str]): Ordered gene identifiers for a segment.
            taxon_name (str): Taxon/species name used for orthogroup mapping.
            rm_tandem (bool): If True, collapse adjacent identical OG ids (remove tandem duplicates).

        Returns:
            List[str]: Ordered list of orthogroup IDs representing the input segment.
        """
        OG_seg = []
        for tmp_gene in segment:
            tmp_OG_id = self.convert_gene_to_OG(taxon_name=taxon_name, gene_id=tmp_gene)

            if tmp_OG_id is None:
                # gene not found in orthogroups, skip it
                continue
            # remove tandem duplicate OGs if requested
            if rm_tandem:
                if OG_seg and OG_seg[-1] == tmp_OG_id:
                    continue
            OG_seg.append(tmp_OG_id)
        return OG_seg

    def calculate_seq_similarity(
        self,
        seg1: List[str],
        seg2: List[str],
        method: Literal['content', 'order', 'comprehensive'] = 'content',
    ) -> float:
        """
        Compute similarity between two sequences using one of three methods:
            - 'content': content overlap (Overlap Coefficient)
            - 'order': order-based similarity derived from Levenshtein distance
            - 'comprehensive': average of content and order similarity

        Parameters:
            seg1, seg2 (List[str]): Two sequences to compare.
            method (str): Chosen similarity method.

        Returns:
            float: Similarity score in [0, 1].
        """
        if method == 'content':
            return seq_content_similarity(seg1, seg2)
        elif method == 'order':
            return seq_order_similarity(seg1, seg2)
        elif method == 'comprehensive':
            return (seq_content_similarity(seg1, seg2) + seq_order_similarity(seg1, seg2)) / 2
        else:
            raise ValueError("method must be one of: 'content', 'order', 'comprehensive'")

    def phased_segments_cluster(self, phased_ratio: float = 0.2):
        """
        Cluster segments in two phases:
            1. Use the top X% (phased_ratio) longest segments to form initial clusters via
               pairwise similarity and agglomerative clustering.
            2. Assign remaining segments to the best matching initial clusters.

        The clustering results are stored in self.SegClusters (list of SegmentsCluster)
        and the segment-to-cluster assignment is written back to self.segments_df['cluster'].

        Parameters:
            phased_ratio (float): Fraction of segments considered as "longest" for phase 1.
        """
        # Convert raw segments into OG segments for every row
        self.OG_segments = []
        for _, row in self.segments_df.iterrows():
            taxon_name = row['species']
            gene_seg = row['genes']
            OG_seg = self.unify_segement_to_OG(segment=gene_seg, taxon_name=taxon_name, rm_tandem=True)
            self.OG_segments.append(OG_seg)

        # Order segment indices by descending length
        seg_idx_by_length = sorted([i for i in range(len(self.OG_segments))], key=lambda x: len(self.OG_segments[x]), reverse=True)

        n_longest = int(len(seg_idx_by_length) * phased_ratio)
        phase1_segs = seg_idx_by_length[:n_longest]
        phase2_segs = seg_idx_by_length[n_longest:]

        # Phase 1: compute pairwise similarity among the longest segments
        self.SegClusters = []
        inital_mat = np.zeros((n_longest, n_longest))
        for i in range(n_longest):
            for j in range(i + 1, n_longest):
                inital_mat[i][j] = self.calculate_seq_similarity(
                    seg1=self.OG_segments[seg_idx_by_length[i]], seg2=self.OG_segments[seg_idx_by_length[j]]
                )
                inital_mat[j][i] = inital_mat[i][j]
        # Cluster the initial matrix
        clusterd_idx, labels = Agglomerative_Cluster_similarity_matrix(inital_mat)
        # Build SegmentsCluster objects from clustering result
        for clust in clusterd_idx.values():
            new_cluster = SegmentsCluster(segments=[self.OG_segments[phase1_segs[i]] for i in clust])
            self.SegClusters.append(new_cluster)

        # Phase 2: assign remaining segments into existing clusters
        phase2_segs = set(phase2_segs)
        while phase2_segs:
            cur_status = []
            for tmp_idx in phase2_segs:
                max_sim = 0
                best_clust_id = None
                for clust_idx, clust in enumerate(self.SegClusters):
                    tmp_sim = clust.cal_seg_content_similar(self.OG_segments[tmp_idx])
                    if tmp_sim > max_sim:
                        max_sim = tmp_sim
                        best_clust = clust_idx
                cur_status.append((tmp_idx, max_sim, best_clust))

            # Select assignments with highest similarity first
            cur_status.sort(key=lambda x: x[1], reverse=True)
            cnt = 0
            for tmp_idx, max_sim, best_clust in cur_status:
                # assign the segment to the best cluster if similarity is above threshold
                # or if we haven't assigned enough segments yet (heuristic)
                if max_sim >= 0.9 or cnt < n_longest:
                    clusterd_idx[best_clust].append(tmp_idx)
                    self.SegClusters[best_clust].add_seg(self.OG_segments[tmp_idx])
                    phase2_segs.remove(tmp_idx)
                    cnt += 1

        # For each cluster, derive a phased gene order
        for tmp_SegCluster in self.SegClusters:
            tmp_SegCluster.pahsed_reorder_genes()

        # Update internal structures and annotate segments_df with cluster assignments
        self.clustered_idx = clusterd_idx
        self.segments_df['cluster'] = -1
        for clust_id, seg_indices in clusterd_idx.items():
            for seg_idx in seg_indices:
                self.segments_df.at[seg_idx, 'cluster'] = clust_id
        return None

    def write_Segment_Cluster_bed(self, outfile: str = 'SegClust.bed'):
        """
        Export cluster gene orders into a simple BED-like file where each cluster is treated
        as a "chromosome" and genes are assigned sequential integer positions.

        Output columns per line:
            chrom (cluster id), start (0-based), end (1-based), name (cluster-gene), score (.), strand (+)

        Parameters:
            outfile (str): Path to write the result.
        """
        with open(outfile, 'w') as f:
            for clust_id, tmp_SegClust in enumerate(self.SegClusters):
                for tmp_idx, tmp_gene in enumerate(tmp_SegClust.gene_order):
                    bed_line = f"{clust_id}\t{tmp_idx}\t{tmp_idx+1}\t{clust_id}-{tmp_gene}\t.\t+\n"
                    f.write(bed_line)
        return None


class ListNode:
    """
    Doubly-linked list node used to construct and manipulate gene order chains.

    Attributes:
        val (str): Stored gene identifier.
        next (ListNode): Next node reference.
        pre (ListNode): Previous node reference.
    """

    def __init__(self, val: str = '', next_node=None, pre_node=None):
        self.val = val
        self.next = next_node
        self.pre = pre_node

    def __str__(self):
        cur = self
        node_str = ''
        while cur != None:
            node_str += str(cur.val) + '->'
            cur = cur.next
        return node_str[:-2]

    def __repr__(self):
        cur = self
        node_str = ''
        while cur != None:
            node_str += str(cur.val) + '->'
            cur = cur.next
        return node_str[:-2]

    def link_node(self, new_node):
        """
        Insert new_node immediately after this node in the doubly-linked list.

        After insertion, updates next/pre pointers for surrounding nodes.
        """
        new_node.next = self.next
        self.next = new_node
        new_node.pre = self
        new_node.next.pre = new_node


class BlockGraph:
    """
    Directed graph representation for blocks with simple conflict resolution.

    Each block is considered a node. Edges represent an asserted ordering between blocks.
    The class supports adding edges, resolving cycles by removing the least confident edge
    in identified cycles, and producing a topological ordering.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.block_index = {block: idx for idx, block in enumerate(blocks)}
        self.n = len(blocks)
        self.graph = collections.defaultdict(list)
        self.in_degree = [0] * self.n
        self.confidence = collections.defaultdict(float)

    def add_edge(self, u, v, conf=1.0):
        """
        Add a directed edge from node index u to node index v with an associated confidence.
        Duplicate edges are ignored.
        """
        if v not in self.graph[u]:
            self.graph[u].append(v)
            self.in_degree[v] += 1
            self.confidence[(u, v)] = conf

    def resolve_conflicts(self):
        """
        Detect simple cycles using DFS and break cycles by removing the edge with the
        lowest confidence within each cycle.

        This is a heuristic to obtain an acyclic graph prior to topological sorting.
        """
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for i in range(self.n):
            if i not in visited:
                dfs(i, [])

        # Remove the weakest edge in each detected cycle
        for cycle in cycles:
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if v in self.graph[u]:
                    cycle_edges.append((u, v, self.confidence.get((u, v), 1.0)))

            if cycle_edges:
                min_edge = min(cycle_edges, key=lambda x: x[2])
                self.graph[min_edge[0]].remove(min_edge[1])
                self.in_degree[min_edge[1]] -= 1
                del self.confidence[(min_edge[0], min_edge[1])]

    def topological_sort(self):
        """
        Return a topological order of nodes after resolving cycles.

        If some nodes remain (due to unresolved ties or removed edges), they are
        appended in descending order of their block size as a deterministic fallback.
        """
        self.resolve_conflicts()

        queue = collections.deque()
        for i in range(self.n):
            if self.in_degree[i] == 0:
                queue.append(i)

        result = []
        while queue:
            u = queue.popleft()
            result.append(u)

            for v in self.graph[u]:
                self.in_degree[v] -= 1
                if self.in_degree[v] == 0:
                    queue.append(v)

        if len(result) < self.n:
            remaining = [i for i in range(self.n) if i not in result]
            remaining_sorted = sorted(remaining, key=lambda x: len(self.blocks[x].OGs), reverse=True)
            result.extend(remaining_sorted)

        return result


class SegmentsCluster:
    """
    Cluster object representing a collection of segments and utilities to derive
    a combined gene order for the cluster.
    """

    def __init__(self, segments: List[List[str]]) -> None:
        self.segments = segments
        # Precompute union of gene content for fast content-similarity checks
        self.gene_content = self.merge_segments_content(segments)
        return None

    def merge_segments_content(self, segs: List[List[str]]) -> Set[str]:
        """
        Merge gene ids from multiple segments into a single set of unique genes.

        Parameters:
            segs (List[List[str]]): List of segments.

        Returns:
            Set[str]: Union of all genes across input segments.
        """
        gene_set = set()
        for tmp_seg in segs:
            gene_set |= set(tmp_seg)
        return gene_set

    def add_seg(self, seg: List[str]) -> None:
        """
        Add a new segment into the cluster and update the union of genes.
        """
        self.segments.append(seg)
        self.gene_content |= set(seg)
        return None

    def cal_seg_content_similar(self, query_seg: List[str]) -> float:
        """
        Compute content similarity between the cluster's gene union and a query segment.

        Returns:
            float: Overlap coefficient in [0,1].
        """
        return seq_content_similarity(self.gene_content, query_seg)

    def get_longest_segs(self, limit_coverage: float = 0.9):
        """
        Select the smallest set of longest segments whose union covers at least
        `limit_coverage` fraction of the cluster's gene content.

        Returns:
            List[int]: Indices of selected segments (relative to self.segments).
        """
        self.len_sorted_seg_idx = sorted([i for i in range(len(self.segments))], key=lambda x: len(self.segments[x]), reverse=True)
        total_genes = len(self.gene_content)
        covered_genes = set()
        selected_idx = []

        for tmp_idx in self.len_sorted_seg_idx:
            if len(covered_genes) / total_genes >= limit_coverage:
                break
            selected_idx.append(tmp_idx)
            covered_genes |= set(self.segments[tmp_idx])

        return selected_idx

    def pahsed_reorder_genes(self) -> List[str]:
        """
        Derive an ordered gene chain for the cluster using a two-phase heuristic:
            1. Build an initial skeleton from the longest, highest-coverage segments and
               order them using order_segments.
            2. Insert remaining genes based on neighbor adjacency scores constructed
               from all segments (construct_neighbor_mat).

        Returns:
            ListNode: Head node of the resulting linked list representing gene order.

        Notes:
            - The function builds a dummy head and tail and returns the head node for later traversal.
        """
        # Initialize a doubly-linked list with head and tail sentinel nodes
        self.dummy = ListNode('tail')
        self.head = ListNode('head', self.dummy)
        self.dummy.pre = self.head
        self.to_node_dic = {}  # map gene -> node in chain
        self.remain_genes = set(self.gene_content)

        # Phase 1: build skeleton from representative longest segments
        top_idx = self.get_longest_segs(limit_coverage=0.9)
        top_segs = [self.segments[i] for i in top_idx]
        ordered_seg_idx = order_segments(top_segs)

        for tmp_idx in ordered_seg_idx:
            # Insert genes from each top segment into the chain preserving order
            self.sort_order_by_given_segments(self.segments[tmp_idx])

        # Phase 2: insert remaining genes using neighbor matrix scores
        neighbor_mat = construct_neighbor_mat(self.segments)
        self.sort_order_by_score_mat(neighbor_mat)
        self.get_loc_dic()

        return self.head

    def sort_order_by_given_segments(self, seg: List[str]) -> None:
        """
        Insert genes from a given segment into the current chain in their order.
        Genes already present are skipped.
        """
        for cur_gene in seg:
            if self.to_node_dic.get(cur_gene, None) is not None:
                # this gene has been added to gene chain
                continue
            else:
                new_node = ListNode(cur_gene)
                self.to_node_dic[cur_gene] = new_node
                # Insert new node before the tail sentinel (append to current end)
                self.dummy.pre.link_node(new_node)
                self.remain_genes.remove(cur_gene)
        return None

    def sort_order_by_score_mat(self, score_mat: Dict[str, Dict[str, int]]) -> None:
        """
        Iteratively insert remaining genes into the chain based on adjacency scores.

        For each remaining gene, find the existing gene in the chain that has the
        highest directed adjacency score (score_mat[chain_gene][candidate]) and attach
        the candidate after that chain_gene. If no positive score exists for any
        candidate, pick an arbitrary gene to avoid infinite loops.
        """
        while self.remain_genes:
            genes_added_this_round = set()
            # Sort remaining genes to make insertion deterministic
            sorted_remain_genes = sorted(list(self.remain_genes))

            for tmp_gene in sorted_remain_genes:
                best_predecessor = None
                max_score = -1

                # Choose predecessor that yields maximum adjacency score
                for gene_in_chain, node in self.to_node_dic.items():
                    score = score_mat.get(gene_in_chain, {}).get(tmp_gene, 0)
                    if score > max_score:
                        max_score = score
                        best_predecessor = gene_in_chain

                if best_predecessor is not None and max_score > 0:
                    new_node = ListNode(tmp_gene)
                    predecessor_node = self.to_node_dic[best_predecessor]
                    predecessor_node.link_node(new_node)
                    self.to_node_dic[tmp_gene] = new_node
                    genes_added_this_round.add(tmp_gene)

            self.remain_genes -= genes_added_this_round

            # If no gene could be placed this round, force-insert a remaining gene
            if not genes_added_this_round and self.remain_genes:
                gene_to_add = self.remain_genes.pop()
                new_node = ListNode(gene_to_add)
                self.dummy.pre.link_node(new_node)
                self.to_node_dic[gene_to_add] = new_node
        return None

    def get_loc_dic(self) -> None:
        """
        Populate self.loc_dic and self.gene_order by traversing the linked list from head.

        After calling this, self.gene_order contains the ordered list of genes and
        self.loc_dic maps gene -> integer position.
        """
        self.loc_dic = {}
        self.gene_order = []
        loc = 0
        cur = self.head.next
        while cur.val != 'tail':
            self.gene_order.append(cur.val)
            self.loc_dic[cur.val] = loc
            loc += 1
            cur = cur.next
        return None

    def map_gene_loc(self, query_gene: str) -> int:
        """
        Return the integer location of a gene in the cluster order, or -1 if not present.
        """
        if self.loc_dic.get(query_gene) == None:
            return -1
        else:
            return self.loc_dic.get(query_gene)


def get_args():
    """
    Parse command-line arguments used by the script when run as a CLI tool.

    Returns:
        argparse.Namespace: Parsed arguments with attributes matching defined CLI flags.
    """
    parser = argparse.ArgumentParser(description='Unify and cluster gene segments based on orthogroups and sequence similarity.')

    # Required input files
    parser.add_argument('--segments', required=True, help='Path to the segments CSV file containing gene segments for each taxon')
    parser.add_argument('--orthogroups', required=True, help='Path to the orthogroups file (tab-separated) containing orthology information')

    # Optional parameters
    parser.add_argument('--output', default='SegClust.bed', help='Output BED file path for cluster results (default: SegClust.bed)')
    parser.add_argument('--phased-ratio', type=float, default=0.2, help='Ratio of longest segments to use for initial clustering (default: 0.2)')
    parser.add_argument('--clade', help='Optional path to clade annotation file (tab-separated: taxon\tgenus\tsubfamily)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level (default: INFO)')

    return parser.parse_args()


def main():
    """
    Execute the SegmentsUnifier pipeline when invoked as a script.

    Steps:
        1. Parse CLI arguments and configure logging.
        2. Initialize SegmentsUnifier with provided files.
        3. Optionally load clade annotations.
        4. Run phased clustering and write BED output.
    """
    args = get_args()

    logging.basicConfig(level=args.log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        logging.info("Initializing SegmentsUnifier with input files...")
        unifier = SegmentsUnifier(segments_file=args.segments, orthogroups_file=args.orthogroups)

        if args.clade:
            logging.info("Loading clade information from %s", args.clade)
            unifier.get_clade_dic(clade_path=args.clade)

        logging.info("Starting phased segment clustering with ratio: %.2f", args.phased_ratio)
        unifier.phased_segments_cluster(phased_ratio=args.phased_ratio)

        logging.info("Writing cluster results to %s", args.output)
        unifier.write_Segment_Cluster_bed(outfile=args.output)

        logging.info("Segment unification and clustering completed successfully!")

    except Exception as e:
        logging.error("An error occurred during processing: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
