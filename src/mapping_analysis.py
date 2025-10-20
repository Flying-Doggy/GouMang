#!/usr/bin/env python3

"""
mapping_analysis.py

This script provides a comprehensive toolkit for comparative genomics, specifically
focusing on mapping and analyzing syntenic blocks (anchors) between a reference
genome and multiple query taxa.

It includes:
- Classes for representing genomic features (genes, blocks, anchors).
- Functions for reading and parsing common genomics file formats (BED,
  OrthoFinder output, JCVI anchors).
- A central `REF_MAPPING` class that orchestrates the entire analysis pipeline:
  1. Loading reference genome blocks.
  2. Loading gene information for all taxa.
  3. Parsing syntenic anchor files (e.g., from JCVI) to map taxa to the reference.
  4. Counting and aggregating mappings to build a chromosome-level
     composition matrix (taxon_chromosome vs. reference_block).
  5. Calculating similarity matrices (both custom block similarity and
     Jaccard similarity based on orthogroups).
  6. Performing hierarchical clustering on chromosomes to identify syntenic groups.
- Utility functions for data manipulation (e.g., merging regions),
  dimensionality reduction (PCA), and visualization (circos plots, heatmaps,
  network graphs).

The main execution block demonstrates a typical workflow using the REF_MAPPING class.
"""

import os
import collections
import re
import glob
import umap
import argparse
import logging  # 导入日志模块

from math import inf
from itertools import pairwise
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
import scipy.cluster.hierarchy as sch
import networkx as nx


def merge_regions(regions: List[List[int]]) -> List[List[int]]:
    """Merges a list of overlapping intervals.

    Args:
        regions: A list of [start, end] intervals.

    Returns:
        A new list of merged [start, end] intervals, sorted by start position.
    """
    regions.sort()
    stack = []
    for x, y in regions:
        if stack and stack[-1][1] >= x:
            stack[-1][1] = max(y, stack[-1][1])
        else:
            stack.append([x, y])
    return stack


def merge_labeled_regions(regions: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    """Merges overlapping or adjacent regions while combining their labels.

    Args:
        regions: A list of tuples, where each tuple is (label, start, end).

    Returns:
        A new list of merged regions, where the label is a comma-separated
        string of all unique labels from the merged regions.
        Example: [('A', 1, 10), ('B', 5, 15)] -> [('A,B', 1, 15)]
    """
    # Step 1: Sort regions by start position, then end position
    regions.sort(key=lambda x: (x[1], x[2]))

    if not regions:
        return []

    new_regions = []
    current_start, current_end = regions[0][1], regions[0][2]
    current_types = {regions[0][0]}

    # Step 2: Iterate through the regions to merge overlapping ones
    for region in regions[1:]:
        region_type, region_start, region_end = region

        # If the region overlaps with or is adjacent to the current region
        if region_start <= current_end:
            # Merge the regions by extending the end if necessary
            current_end = max(current_end, region_end)
            # Add the new region's type to the set of types
            current_types.add(region_type)
        else:
            # No overlap, so finalize the current region and start a new one
            new_regions.append((','.join(sorted(current_types)), current_start, current_end))
            current_start, current_end = region_start, region_end
            current_types = {region_type}

    # Add the last processed region
    new_regions.append((','.join(sorted(current_types)), current_start, current_end))

    return new_regions


class gene:
    """Stores information for a single gene, including its location.
    
    Attributes:
        id (str): Gene identifier.
        start (int): Start position (1-based or genomic).
        end (int): End position (1-based or genomic).
        chrom (str): Chromosome or scaffold name.
        seq (Optional[str]): Sequence (optional).
        type (str): Gene type (optional).
    """

    def __init__(self, ID: str = '', start: int = 0, end: int = 0, chrom: str = '1', seq: Optional[str] = None):
        """Initializes the gene object."""
        self.id = ID
        self.start = int(start)
        self.chrom = chrom
        self.end = end
        self.type = ''
        self.seq = seq

    def __repr__(self) -> str:
        """Official string representation."""
        return f'{self.id} , located in chr:{self.chrom} site:{self.start} .'

    def __str__(self) -> str:
        """Informal string representation (the gene ID)."""
        return str(self.id)

    def set_newattr(self, attr_key: str = 'OG', attr_val: Optional[str] = None) -> None:
        """Sets a new attribute on the gene object.

        Args:
            attr_key: The name of the attribute to set (e.g., 'OG').
            attr_val: The value for the new attribute.
        """
        setattr(self, attr_key, attr_val)
        return None


class block_region:
    """Represents a fixed-size window (block) on a reference chromosome.
    
    Attributes:
        len (int): The length of the window (number of genes).
        loc (Tuple[str, int, int]): Chromosome, start index, end index (gene indices).
        Gene_List (List[str]): List of gene IDs within this block.
        sp_dic (defaultdict[str, int]): Coverage count of this block in each species.
        gene_dic (defaultdict[str, dict]): Mapping of genes in this block
                                           to corresponding genes in species.
        anchors_list (List[anchors]): List of anchor objects that cover this block.
    """

    def __init__(self, chrom: str, start: int, length: int, gene_list: List[str]):
        """Initializes the block_region object."""
        self.len = length
        self.loc = (chrom, start, start + length)  # Chromosome location (gene index)
        self.Gene_List = gene_list
        self.sp_dic = collections.defaultdict(int)  # Coverage count per species
        self.gene_dic = collections.defaultdict(dict)  # Gene mappings
        self.anchors_list = []

    def __repr__(self) -> str:
        """Official string representation."""
        return f'{self.loc[0]}:{self.loc[1]}-{self.loc[2]}'

    def __str__(self) -> str:
        """Informal string representation."""
        return f'{self.loc[0]}:{self.loc[1]}-{self.loc[2]}'

    def count_clade(self, clade_dic: Dict[str, List[str]]) -> None:
        """Counts block copy number and presence within specified clades.

        This method populates two new attributes on the object:
        - self.clade_total: Total copy number of this block per clade.
        - self.clade_cnt: Number of species within a clade that have this block.

        Args:
            clade_dic: A dictionary mapping clade names to lists of species names.
        """
        self.clade_total = collections.defaultdict(int)
        self.clade_cnt = collections.defaultdict(int)
        for tmp_clade in clade_dic.keys():
            for tmp_sp in clade_dic[tmp_clade]:
                self.clade_total[tmp_clade] += self.sp_dic[tmp_sp]
                if self.sp_dic[tmp_sp] > 0:
                    self.clade_cnt[tmp_clade] += 1
        return None


class anchors:
    """Stores information about a syntenic anchor block between two species.
    
    Attributes:
        ID (str): An identifier for the anchor block.
        sp (List[str]): The names of the two species [sp1, sp2].
        genes (Dict[str, List[str]]): A mapping from species name to its list
                                      of gene IDs in the anchor.
        scores (List[int]): List of scores for each gene pair.
        chrom (List[str]): Chromosome for the block in each species.
        start (List[int]): Start position for the block in each species.
        end (List[int]): End position for the block in each species.
    """

    def __init__(self, genes_tuple_list: List[List[str]], sp1: str, sp2: str):
        """Initializes the anchors object.
        
        Args:
            genes_tuple_list: A list of [gene1, gene2, score] lists.
            sp1: Name of the first species.
            sp2: Name of the second species.
        """
        self.ID = ''
        self.genes = {}
        self.sp = [sp1, sp2]
        self.genes[sp1] = [i[0] for i in genes_tuple_list]
        self.genes[sp2] = [i[1] for i in genes_tuple_list]
        self.scores = [int(i[2]) for i in genes_tuple_list]
        self.chrom = ['', '']
        self.start = [0, 0]
        self.end = [0, 0]
        return None

    def get_location(self, idx: int, gene_dic: Dict[str, gene], query_genes_list: Optional[List[str]] = None) -> None:
        """Calculates and sets the genomic location (chrom, start, end)
        for one species in the anchor.

        Args:
            idx: The index of the species (0 or 1) to get the location for.
            gene_dic: A dictionary mapping gene IDs to 'gene' objects for the
                      target species.
            query_genes_list: If provided, only consider genes in the anchor
                              whose partners (in the *other* species)
                              are in this list.
        """
        if query_genes_list is None:
            candidates = self.genes[self.sp[idx]]
        else:
            # Filter genes based on their partners in the *other* species
            candidates = [
                self.genes[self.sp[idx]][i]
                for i in range(len(self.genes[self.sp[idx]]))
                if self.genes[self.sp[1 - idx]][i] in query_genes_list
            ]

        tmp_chr, tmp_start, tmp_end = '', inf, 0
        for tmp_gene_id in candidates:
            if tmp_gene_id not in gene_dic:
                continue  # Skip if gene ID not found
            tmp_gene_obj = gene_dic[tmp_gene_id]
            tmp_chr = tmp_gene_obj.chrom
            tmp_start = min(tmp_start, tmp_gene_obj.start)
            tmp_end = max(tmp_end, tmp_gene_obj.end)

        self.chrom[idx], self.start[idx], self.end[idx] = tmp_chr, tmp_start, tmp_end
        return None

    def __repr__(self) -> str:
        """Official string representation."""
        return (f" {self.sp[0]} and {self.sp[1]} anchors-{self.ID} has {len(self.scores)} genes ,"
                f" ranged in {self.sp[0]} {self.chrom[0]}:{self.start[0]}-{self.end[0]} ,"
                f" {self.sp[1]} {self.chrom[1]}:{self.start[1]}-{self.end[1]}")


class color_pan:
    """A utility class to assign consistent colors to elements.
    
    It cycles through a list of candidate colors, mapping each new element
    to the next color in the list.
    """

    def __init__(self, candidate_colors: List[str]):
        """Initializes the color pan.
        
        Args:
            candidate_colors: A list of colors (e.g., hex codes, names).
        """
        self.candidates = candidate_colors
        self.idx = 0
        self.pan = {}  # Stores element -> color mapping

    def get_color(self, element: str) -> str:
        """Gets a color for an element. 
        
        If the element is new, it assigns the next color in the list.

        Args:
            element: The item to get a color for.

        Returns:
            The assigned color.
        """
        if self.pan.get(element) is None:
            self.pan[element] = self.candidates[self.idx]
            self.idx += 1
            self.idx %= len(self.candidates)
        return self.pan.get(element)

    def get(self, element: str) -> str:
        """Alias for get_color."""
        return self.get_color(element)

    def __getitem__(self, element: str) -> str:
        """Allows dictionary-style access."""
        return self.get_color(element)

    def __call__(self, element: str) -> str:
        """Allows function-call-style access."""
        return self.get_color(element)


def read_bed_file(file_path: str, gene_dist: bool = True) -> Dict[str, gene]:
    """Reads a BED file and returns a dictionary of gene objects.

    Args:
        file_path: Path to the BED file.
        gene_dist: If True, calculates relative positions (gene order)
                   instead of using genomic coordinates. Start/end
                   will be 1-based indices per chromosome.

    Returns:
        A dictionary mapping gene IDs to 'gene' objects.
    """
    genes_dic = {}
    tmp_cnt = 0
    prev_chrom = prev_gene = None
    with open(file=file_path) as f:
        datas = f.readlines()
        for line in datas:
            if line == '\n':
                continue
            try:
                tmp_chr, tmp_start, tmp_end, tmp_id = line.strip().split('\t')[:4]
            except ValueError:
                logging.warning(f"Skipping malformed BED line: {line.strip()}")
                continue

            if tmp_id == prev_gene:
                continue  # Skip duplicate gene IDs
            prev_gene = tmp_id

            if gene_dist:  # Calculate relative position based on gene order
                if tmp_chr != prev_chrom:
                    tmp_cnt = 1
                tmp_start = tmp_cnt
                tmp_end = tmp_start + 1
                tmp_cnt += 1
                prev_chrom = tmp_chr

            genes_dic[tmp_id] = gene(tmp_id,
                                     start=int(tmp_start),
                                     end=int(tmp_end),
                                     chrom=tmp_chr)
    return genes_dic


def read_block_gene_file(file_path: str) -> Dict[str, List[str]]:
    """Reads a file mapping block IDs to their ordered list of gene IDs.

    Note: This function implementation is specific. The key in the returned
    dictionary is the *line index* (as a string), and the value is the
    *full list* of tab-separated elements on that line (including the
    block ID at index 0).

    Expected line format: block_ID\tgene1\tgene2\t...

    Args:
        file_path: Path to the block-gene file.

    Returns:
        A dictionary mapping line index (str) to a list of elements (List[str]).
    """
    block_dic_original = collections.defaultdict(list)
    with open(file=file_path) as f:
        datas = f.readlines()
        for idx, tmp_line in enumerate(datas):
            tmp_record = tmp_line.strip('\n').split('\t')
            # The key is the line index (as a string).
            # The value is the *entire* split line, including the block ID.
            block_dic_original[str(idx)] = tmp_record
    return block_dic_original


def parse_block_gene_from_bed(file_path: str) -> Dict[str, List[str]]:
    """Parses a BED-like file to map block IDs to gene IDs.

    Assumes a tab-separated file where column 0 is the block ID (e.g.,
    chromosome) and column 3 is the gene ID.

    Args:
        file_path: Path to the BED-like file.

    Returns:
        A dictionary mapping block_ID (str) to a list of gene IDs (List[str]).
    """
    block_dic = collections.defaultdict(list)
    with open(file=file_path) as f:
        for line in f:
            if line.strip():
                tmp_record = line.strip('\n').split('\t')
                if len(tmp_record) >= 4:
                    block_id = tmp_record[0]
                    gene_id = tmp_record[3]
                    block_dic[block_id].append(gene_id)
    return block_dic


def create_chromosome_blocks(chr_ID: str, gene_list: List[str],
                             gene_dic: Dict[str, block_region],
                             window_length: int = 30) -> List[block_region]:
    """Splits a chromosome's gene list into fixed-size block_region objects.

    This function also populates the `gene_dic` mapping as a side effect,
    linking each gene ID to its corresponding block_region object.

    Args:
        chr_ID: The chromosome identifier.
        gene_list: An ordered list of gene IDs on the chromosome.
        gene_dic: A dictionary to be populated with {gene_ID} -> block_region.
        window_length: The number of genes per block.

    Returns:
        A list of 'block_region' objects for the chromosome.
    """
    res_block_list = []
    for idx in range(0, len(gene_list), window_length):
        tmp_block_object = block_region(chrom=chr_ID, start=idx, length=window_length,
                                        gene_list=gene_list[idx: idx + window_length])
        res_block_list.append(tmp_block_object)

        for tmp_gene in tmp_block_object.Gene_List:
            gene_dic[f'{tmp_gene}'] = tmp_block_object
    return res_block_list


def read_anchors_file(file_path: str) -> List[List[List[str]]]:
    """Reads a JCVI-style .anchors file.

    Blocks are separated by '#'. Only blocks with 10 or more
    gene pairs are retained.

    Args:
        file_path: Path to the .anchors file.

    Returns:
        A list of blocks. Each block is a list of [gene1, gene2, score] lists.
    """
    res_list = []
    tmp_list = []
    with open(file=file_path) as f:
        datas = f.readlines()
        for i in datas:
            if i.startswith('#'):  # A new block starts
                if len(tmp_list) >= 10:  # Filter for blocks with >= 10 gene pairs
                    res_list.append(tmp_list)
                tmp_list = []
            else:
                line_content = i.strip('\n').split('\t')
                if len(line_content) >= 3: # Ensure line is valid
                    tmp_list.append(line_content)
                    
    # Add the last block if it meets the criteria
    if len(tmp_list) >= 10:
        res_list.append(tmp_list)
    return res_list


def identify_covered_block(query_gene_list: List[str],
                           gene_region_dic: Dict[str, block_region],
                           covered_limit: int = 6) -> List[str]:
    """Identifies which reference blocks are "covered" by a list of query genes.

    A block is considered "covered" if at least `covered_limit` genes
    from the `query_gene_list` fall into it.

    Args:
        query_gene_list: A list of gene IDs (e.g., from a reference anchor).
        gene_region_dic: A mapping from gene_ID to its block_region object.
        covered_limit: The minimum number of genes from the query list
                       that must fall into a block for it to be "covered".

    Returns:
        A list of block region IDs (as strings) that are covered.
    """
    # Map query gene IDs to their corresponding region IDs (as strings)
    covered_block_ID_list = [
        str(gene_region_dic[i]) for i in query_gene_list if i in gene_region_dic
    ]

    cnt_dic = collections.Counter(covered_block_ID_list)
    return [i for i in cnt_dic.keys() if cnt_dic[i] >= covered_limit]


def int_dict() -> collections.defaultdict:
    """Factory function for defaultdict(int)."""
    return collections.defaultdict(int)


def double_layer_int_dict() -> collections.defaultdict:
    """Factory function for defaultdict(defaultdict(int))."""
    return collections.defaultdict(int_dict)


def split_chromosome_name(chrom_ID: str) -> Tuple[int, str]:
    """Splits a chromosome name into its numeric and string parts.

    Args:
        chrom_ID: Chromosome name, e.g., "chr1", "1", or "chrX".

    Returns:
        A tuple (Numeric part, String part).
        Returns (0, "name") if no number is found.

    Examples:
        >>> split_chromosome_name("chr1")
        (1, 'chr')
        >>> split_chromosome_name("1")
        (1, '')
        >>> split_chromosome_name("chrX")
        (0, 'chrX')
        >>> split_chromosome_name("scaffold123")
        (123, 'scaffold')
    """
    # Handle all-digit chromosome names
    if chrom_ID.isdigit():
        return int(chrom_ID), ''

    # Handle names with numbers (e.g., "chr1", "scaffold123")
    match = re.match(r'^([^\d]*)(\d+)(.*)$', chrom_ID)
    if match:
        prefix, digits, suffix = match.groups()
        # Combine prefix and suffix for the string part
        return int(digits), prefix + suffix

    # Handle names without numbers (e.g., "chrX", "mitochondrion")
    return 0, chrom_ID


def plot_chrom(x: int, composition: List[Tuple[str, int, int]],
               color_pan: color_pan, width: int = 1, legend_plot: bool = True) -> None:
    """Plots a chromosome's composition as a stacked vertical line.

    Args:
        x: The x-coordinate for the vertical line.
        composition: A list of (cluster_id, start, end) tuples.
        color_pan: A color_pan object to assign colors to cluster_ids.
        width: The linewidth for the plot.
        legend_plot: Whether to add a legend for the clusters.
    """
    for clus, start, end in composition:
        plt.vlines(x=x, ymin=start, ymax=end, color=color_pan.get_color(clus), linewidth=width)

    if legend_plot:
        handles = [
            plt.vlines(x=x, ymax=0, ymin=0, label=tmp_clus, color=color_pan.get_color(tmp_clus))
            for tmp_clus in color_pan.pan.keys()
        ]
        plt.legend(handles=handles)
    return None


def get_pca_df(query_df: pd.DataFrame, n_components: int, plot_flag: bool = True) -> pd.DataFrame:
    """Performs PCA on a DataFrame.

    Args:
        query_df: The input DataFrame (features as columns, samples as rows).
        n_components: The number of principal components to return.
        plot_flag: If True, plots the cumulative explained variance.

    Returns:
        A new DataFrame containing the principal components.
    """
    # Fit with a larger number first to show variance plot
    pca_full = PCA(n_components=min(50, query_df.shape[1]))
    pca_full.fit(query_df)

    if plot_flag:
        plt.figure()
        plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                 np.cumsum(pca_full.explained_variance_ratio_))
        plt.vlines(x=n_components, ymin=0, ymax=1, color='red', linestyle='--')
        plt.title('PCA Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance')
        plt.show()

    # Re-fit or select the requested number of components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(query_df)

    pca_df = pd.DataFrame(pca_result,
                          columns=[f'pca_{i}' for i in range(n_components)],
                          index=query_df.index)
    return pca_df


def get_umap_df(query_df: pd.DataFrame, n_components: int, plot_flag: bool = True) -> pd.DataFrame:
    """Performs UMAP dimensionality reduction on a DataFrame.

    Args:
        query_df: The input DataFrame (features as columns, samples as rows).
        n_components: The number of UMAP components to return.
        plot_flag: If True, plots the first two UMAP dimensions.

    Returns:
        A new DataFrame containing the UMAP components.
    """
    umap_model = umap.UMAP(n_components=n_components, random_state=123 , n_jobs=1)
    umap_result = umap_model.fit_transform(query_df)

    if plot_flag and n_components >= 2:
        plt.figure()
        plt.scatter(umap_result[:, 0], umap_result[:, 1], s=5)
        plt.title('UMAP Projection (First 2 Dimensions)')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.show()

    umap_df = pd.DataFrame(umap_result,
                           columns=[f'umap_{i}' for i in range(n_components)],
                           index=query_df.index)
    return umap_df


def get_sim_df(query_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a custom similarity matrix (unoptimized version).
    
    Similarity = 1 - (||v_i - v_j||^2 / ||max(v_i, v_j)||^2)
    where vectors only include columns where at least one row is > 0.

    Note: This is the unoptimized, row-by-row version.
          `get_sim_df_optimized` is preferred for performance.

    Args:
        query_df: Input DataFrame (e.g., chromosomes vs. blocks).

    Returns:
        A square DataFrame of pairwise similarities.
    """
    chr_sim_dic = collections.defaultdict(dict)

    for tmp_chr in query_df.index:
        for q_chr in query_df.index:
            if chr_sim_dic[tmp_chr].get(q_chr) is not None:
                continue

            # Find columns where at least one chromosome has a value
            valid_col = (query_df.loc[tmp_chr, :] > 0) | (query_df.loc[q_chr, :] > 0)
            tmp_val, q_val = query_df.loc[tmp_chr, valid_col], query_df.loc[q_chr, valid_col]

            if tmp_val.empty:
                chr_sim_dic[tmp_chr][q_chr] = chr_sim_dic[q_chr][tmp_chr] = 1.0
                continue

            max_val = [max(tmp_val[i], q_val[i]) for i in range(len(tmp_val))]
            dis_val = [tmp_val[i] - q_val[i] for i in range(len(tmp_val))]

            norm_dis = np.linalg.norm(dis_val)
            norm_max = np.linalg.norm(max_val)

            if norm_max == 0:
                tmp_sim = 1.0  # Both are zero vectors
            else:
                tmp_sim = 1 - (norm_dis / norm_max) ** 2

            chr_sim_dic[tmp_chr][q_chr] = chr_sim_dic[q_chr][tmp_chr] = tmp_sim

    return pd.DataFrame.from_dict(chr_sim_dic)


def get_sim_df_optimized(query_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a custom similarity matrix (NumPy optimized version).
    
    Similarity = 1 - (sum((v_i - v_j)^2) / sum(max(v_i, v_j)^2))
    where vectors only include columns where at least one row is > 0.

    Args:
        query_df: Input DataFrame (e.g., chromosomes vs. blocks).

    Returns:
        A square DataFrame of pairwise similarities.
    """
    index = query_df.index
    n = len(index)
    data = query_df.to_numpy()

    # Initialize similarity matrix
    sim_matrix = np.ones((n, n))  # Default similarity is 1 (for all-zero vectors)

    for i in range(n):
        row_i = data[i]
        for j in range(i, n):
            row_j = data[j]

            # Calculate valid column mask
            valid_col = (row_i > 0) | (row_j > 0)
            tmp_val = row_i[valid_col]
            q_val = row_j[valid_col]

            if tmp_val.size == 0:
                # All-zero case, similarity remains 1
                continue

            max_val = np.maximum(tmp_val, q_val)
            dis_val = tmp_val - q_val

            # Calculate sum of squares
            norm_dis_sq = np.sum(dis_val ** 2)
            norm_max_sq = np.sum(max_val ** 2)

            if norm_max_sq > 0:
                sim = 1 - (norm_dis_sq / norm_max_sq)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    return pd.DataFrame(sim_matrix, index=index, columns=index)


def get_hierarchical_clustering_order(query_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Performs hierarchical clustering and returns the reordered
    index and group/color labels.

    Args:
        query_df: A square similarity/distance matrix (DataFrame).

    Returns:
        A tuple:
        - (List[str]): The reordered index (e.g., species/chromosome names).
        - (List[str]): The color labels for the groups (from dendrogram).
    """
    clustergrid = sns.clustermap(query_df, cmap='Oranges', xticklabels=[], yticklabels=[], method="weighted")

    # Get the order of species/chromosomes from the clustering
    sp_seq = [query_df.index[i] for i in clustergrid.dendrogram_col.reordered_ind]

    # Get the linkage matrix and dendrogram to find cluster groups
    linkage_matrix = clustergrid.dendrogram_row.linkage
    # Use a high distance threshold to get distinct colors for major groups
    dend_plot = sch.dendrogram(linkage_matrix, no_plot=True)

    # Get the colors assigned to the leaves (which represent groups)
    group_seq = dend_plot['leaves_color_list']

    plt.clf()  # Clear the plot generated by clustermap/dendrogram
    return sp_seq, group_seq


def parse_orthogroups_file(file_path: str = '', alter_dict: Optional[Dict[str, Dict[str, str]]] = None,
                           sp_limit: int = 50) -> collections.defaultdict:
    """Parses an OrthoFinder (Orthogroups.tsv) file.

    Args:
        file_path: Path to "Orthogroups.tsv".
        alter_dict: A nested dictionary to translate transcript IDs to
                    gene IDs. Format: {species_name: {transcript_id: gene_id}}.
        sp_limit: Skip orthogroups that are absent from this many species.

    Returns:
        A nested defaultdict: {species_name: {gene_id: OG_id}}.
    """
    gene_OG_dic = collections.defaultdict(dict)
    OG_cnt = 0
    with open(file=file_path) as f:
        datas = f.readlines()
        if not datas:
            return gene_OG_dic
            
        head_line = datas[0].strip('\n').split('\t')

        for tmp_line in datas[1:]:
            tmp_line = tmp_line.strip('\n').split('\t')
            tmp_OG = tmp_line[0]

            # if the number of species, whose genome absents this OG,
            # is more than {sp_limit}, then skip this OG.
            if tmp_line.count('') >= sp_limit:
                continue

            OG_cnt += 1
            for tmp_idx, tmp_genes_str in enumerate(tmp_line):
                if tmp_idx == 0:  # skip the first (0-th) column (OG ID)
                    continue

                species_name = head_line[tmp_idx]
                for tmp_gene in tmp_genes_str.split(', '):
                    if tmp_gene:  # skip empty elements
                        # Fix for potential transcript ID format
                        tmp_gene = tmp_gene.replace('transcript_', 'transcript:')

                        if alter_dict:
                            # convert transcript ID to gene ID (IMPORTANT)
                            tmp_gene = alter_dict[species_name].get(tmp_gene, tmp_gene)

                        gene_OG_dic[species_name][tmp_gene] = tmp_OG
    return gene_OG_dic


def calculate_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculates the Jaccard similarity between two sets.

    Args:
        set_a: The first set.
        set_b: The second set.

    Returns:
        The Jaccard similarity (0.0 to 1.0).
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_count_distance(vec1: List[float], vec2: List[float], method: str = 'manhattan') -> float:
    """Computes the distance between two count vectors.

    Args:
        vec1: The first vector.
        vec2: The second vector.
        method: 'manhattan' or 'cosine'.

    Returns:
        The calculated distance.
        
    Raises:
        ValueError: If an unknown distance method is specified.
    """
    if method == 'manhattan':
        return np.sum(np.abs(np.array(vec1) - np.array(vec2)))
    elif method == 'cosine':
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Cosine distance is 1 if one vector is zero
        return 1 - (dot_product / (norm1 * norm2))
    else:
        raise ValueError(f"Unknown distance method: {method}")


def plot_chromosome_correlation(chromosomes: Dict[str, List[str]],
                                correlations: List[Tuple[str, str]],
                                taxon: str, annot_gene: Optional[List[str]] = None) -> None:
    """Plots a Circos-style visualization of intra-genomic correlations.

    Args:
        chromosomes: A dictionary mapping {chrom_id: [gene_id_list]}.
        correlations: A list of (gene1, gene2) tuples representing links.
        taxon: The name of the taxon for the plot title.
        annot_gene: A list of gene IDs to highlight (optional).
    """
    # Create an empty graph
    G = nx.Graph()

    # Add nodes (genes)
    for chrom, genes in chromosomes.items():
        for gene in genes:
            G.add_node(gene, chromosome=chrom)

    # Add edges (correlations)
    for gene1, gene2 in correlations:
        if G.has_node(gene1) and G.has_node(gene2):
            G.add_edge(gene1, gene2)

    # Calculate total gene count
    total_genes = sum(len(genes) for genes in chromosomes.values())
    if total_genes == 0:
        logging.info(f"No genes to plot for {taxon}.")
        return None

    # Calculate angular range for each chromosome
    num_chromosomes = len(chromosomes)
    interval_angle = 0.1  # Angle spacing between chromosomes
    total_interval = interval_angle * num_chromosomes
    available_angle = 2 * np.pi - total_interval
    chromosome_angles = {}
    start_angle = 0

    # Sort chromosomes for consistent plotting
    sorted_chroms = sorted(chromosomes.keys(), key=lambda x: split_chromosome_name(x))

    for chrom in sorted_chroms:
        genes = chromosomes[chrom]
        gene_count = len(genes)
        # Calculate angle for this chromosome based on gene count
        angle = available_angle * (gene_count / total_genes)
        end_angle = start_angle + angle
        chromosome_angles[chrom] = (start_angle, end_angle)
        start_angle = end_angle + interval_angle

    # Calculate position for each gene
    pos = {}
    for chrom, (start_angle, end_angle) in chromosome_angles.items():
        genes = chromosomes[chrom]
        if not genes:
            continue
        gene_angles = np.linspace(start_angle, end_angle, len(genes))
        for i, gene in enumerate(genes):
            x = np.cos(gene_angles[i]) * 1.05
            y = np.sin(gene_angles[i]) * 1.05
            pos[gene] = (x, y)

    # Plot the circular genome graph
    plt.figure(figsize=(10, 10))

    # Draw chromosome arcs
    ax = plt.gca()
    for chrom, (start_angle, end_angle) in chromosome_angles.items():
        arc = Arc((0, 0), 2.2, 2.2, theta1=start_angle * 180 / np.pi, theta2=end_angle * 180 / np.pi,
                  color='gray', lw=2)
        ax.add_patch(arc)

        # Calculate position for chromosome label
        middle_angle = (start_angle + end_angle) / 2
        label_x = 1.2 * np.cos(middle_angle)
        label_y = 1.2 * np.sin(middle_angle)
        # Add chromosome name
        ax.text(label_x, label_y, f'Chr{chrom}', ha='center', va='center', fontsize=12)

    if annot_gene:
        # Highlight specified genes
        annot_G = nx.Graph()
        annot_pos = {}
        for tmp_gene in annot_gene:
            if tmp_gene in pos:
                annot_G.add_node(tmp_gene)
                annot_pos[tmp_gene] = pos[tmp_gene]
        nx.draw_networkx_nodes(annot_G, annot_pos, node_size=3, node_color='lightblue')

    # Draw correlation lines (edges)
    nx.draw_networkx_edges(G, pos, edge_color='#999999', alpha=0.1)

    # Set graph properties
    plt.axis('equal')
    plt.axis('off')
    plt.title(f'{taxon} Chromosome Correlation Visualization')
    plt.show()
    return None


def get_main_individual_composition(query_compositions: Dict[str, float], judge_limit: float = 0.95) -> List[str]:
    """Finds the "main" components that sum up to a given fraction
    of the total, based on simple counts.

    Args:
        query_compositions: A dictionary {component_id: count}.
        judge_limit: The cumulative fraction to reach (e.g., 0.95 for 95%).

    Returns:
        A list of the main component IDs, sorted by count in descending order.
    """
    total_comp = sum(query_compositions.values())
    if total_comp == 0:
        return []

    cur_count = 0
    main_comp = []
    for tmp_comp, tmp_count in sorted(query_compositions.items(), key=lambda x: -x[1]):
        cur_count += tmp_count
        main_comp.append(tmp_comp)
        if cur_count >= total_comp * judge_limit:
            return main_comp
    return main_comp


def get_main_merged_composition(query_compositions: Dict[str, List[List[int]]],
                                judge_limit: float = 0.95) -> List[str]:
    """Finds the "main" components based on merged genomic region coverage.

    Args:
        query_compositions: A dictionary {component_id: list_of_regions},
                            where each region is [start, end].
        judge_limit: The cumulative fraction of *covered length* to reach.

    Returns:
        A list of the main component IDs, sorted by their total
        individual coverage.
    """
    total_regions = []
    for i in query_compositions.values():
        total_regions += i
    total_regions = merge_regions(total_regions)  # Total covered region
    total_cnt = sum(i[1] - i[0] for i in total_regions)
    if total_cnt == 0:
        return []

    # Sort components by their total covered length (before merging)
    comp_seq = sorted(query_compositions.keys(),
                      key=lambda x: -sum(i[1] - i[0] for i in query_compositions[x]))

    cur_regions = []
    main_comp = []
    for tmp_comp in comp_seq:  # Iterate from largest to smallest
        cur_regions += query_compositions[tmp_comp]
        cur_merge = merge_regions(cur_regions)
        main_comp.append(tmp_comp)

        # Check if cumulative merged length reaches the limit
        if sum(i[1] - i[0] for i in cur_merge) >= total_cnt * judge_limit:
            return main_comp
    return main_comp


class REF_MAPPING:
    """
    A class to manage and execute a reference-based synteny mapping
    and clustering workflow.
    
    This class is stateful. Methods should be called in the following
    general order:
    1. get_clade_dic (Optional)
    2. get_REF_regions
    3. get_TAXON_beds
    4. get_TAXON_anchors
    5. count_TAXON_regions
    6. get_chrom_df
    7. filter_chrom_df
    8. get_chrom_sim_df
    9. plot_cluster_heatmap
    10. (Optional) get_KMeans_label, get_tSNE_dimension
    11. (Optional) get_OG_dic, set_all_genes_OG, get_chrom_Jaccard_sim
    12. (Optional) get_hierarchy_cnt, get_chromosomes_group_similarity
    """

    def __init__(self):
        """Initializes the REF_MAPPING class."""
        self.taxons: Set[str] = set()  # Stores the names of all query taxa
        self.ref_path: Optional[str] = None
        self.window_length: int = 20
        self.ref_genome: Dict[str, List[str]] = {}
        self.ref_regions: Dict[str, block_region] = {}
        self.ref_genes: Dict[str, block_region] = {}
        self.OG_dic: collections.defaultdict = collections.defaultdict(dict)
        self.bed_path: Optional[str] = None
        self.taxon_beds: Dict[str, Dict[str, gene]] = {}
        self.covered_limit: int = 8
        self.anchors_path: Optional[str] = None
        self.taxon_chrom: collections.defaultdict = collections.defaultdict(double_layer_int_dict)
        self.chrom_region: collections.defaultdict = collections.defaultdict(list)
        self.clade_path: Optional[str] = None
        self.clade_dic: collections.defaultdict = collections.defaultdict(list)
        self.taxon_clade_dic: Dict[str, str] = {}
        self.chrom_comp_df: pd.DataFrame = pd.DataFrame()
        self.chrom_map: pd.DataFrame = pd.DataFrame()
        self.KM_labels: pd.DataFrame = pd.DataFrame()
        self.tSNE_D: pd.DataFrame = pd.DataFrame()
        self.chrom_sim_df: pd.DataFrame = pd.DataFrame()
        self.hierachy_group: Dict[str, str] = {}
        self.grouped_chroms: collections.defaultdict = collections.defaultdict(list)
        self.hierachy_seq: List[str] = []
        self.clade_color: Optional[color_pan] = None
        self.hierachy_cnt: collections.defaultdict = collections.defaultdict(dict)
        self.Jacccard_taxons: List[str] = []
        self.chrom_OGs: collections.defaultdict = collections.defaultdict(set)
        self.chrom_Jaccard_df: pd.DataFrame = pd.DataFrame()
        self.group_sim: pd.DataFrame = pd.DataFrame()
        return None

    def get_REF_regions(self, ref_path: str, window_length: int = 20) -> None:
        """Loads the reference genome's block structure from a BED-like file.
        
        Args:
            ref_path: Path to the reference block-gene file (parsed by
                      `parse_block_gene_from_bed`).
            window_length: The number of genes per window/block.
            
        Sets Attributes:
            self.ref_path (str): Path to the reference file.
            self.window_length (int): Genes per block.
            self.ref_genome (Dict): {ref_chrom_id: [gene_list]}.
            self.ref_regions (Dict): {block_region_str: block_region_obj}.
            self.ref_genes (Dict): {gene_id: block_region_obj}.
        """
        self.ref_path = ref_path
        self.window_length = window_length
        self.ref_genome = parse_block_gene_from_bed(ref_path)
        self.ref_regions = {}
        self.ref_genes = {}

        for tmp_chr, tmp_genes in self.ref_genome.items():
            tmp_regions = create_chromosome_blocks(tmp_chr, tmp_genes,
                                                 gene_dic=self.ref_genes,
                                                 window_length=self.window_length)

            for tmp_region in tmp_regions:
                self.ref_regions[str(tmp_region)] = tmp_region
        return None

    def get_OG_dic(self, OG_file: str, alter_dict: Optional[Dict[str, Dict[str, str]]] = None,
                   sp_limit: int = 50) -> None:
        """Loads the OrthoFinder orthogroup mappings.
        
        Args:
            OG_file: Path to "Orthogroups.tsv".
            alter_dict: Optional dictionary for translating gene IDs.
            sp_limit: Orthogroup species sparsity limit.
            
        Sets Attributes:
            self.OG_dic (defaultdict): {species: {gene_id: OG_id}}.
        """
        self.OG_dic = parse_orthogroups_file(file_path=OG_file,
                                             alter_dict=alter_dict,
                                             sp_limit=sp_limit)
        return None

    def get_TAXON_beds(self, bed_path: str) -> None:
        """Loads all taxon BED files from a directory.
        
        Args:
            bed_path: Path to the directory containing .bed files.
            
        Sets Attributes:
            self.bed_path (str): Path to the BED directory.
            self.taxon_beds (Dict): {taxon_name: {gene_id: gene_obj}}.
            self.taxons (set): Updated with taxon names from BED files.
        """
        self.bed_path = bed_path
        self.taxon_beds = {}
        for tmp_bed_file in glob.glob(os.path.join(bed_path, '*.bed')):
            tmp_taxon = os.path.basename(tmp_bed_file).split('.')[0]
            self.taxons.add(tmp_taxon)
            self.taxon_beds[tmp_taxon] = read_bed_file(tmp_bed_file)
        return None

    def parse_TAXON_anchors(self, taxon_anchor_file: str, covered_limit: int = 8) -> None:
        """Parses a single taxon's anchor file and maps it to reference regions.
        
        Args:
            taxon_anchor_file: Path to the .anchors file.
            covered_limit: Minimum number of genes to count as "covered".
        """
        tmp_blocks_raw = read_anchors_file(taxon_anchor_file)
        try:
            cur_taxon, _REF = os.path.basename(taxon_anchor_file).split('.')[:2]
        except ValueError:
            logging.warning(f"Skipping {taxon_anchor_file}: Unexpected file name format.")
            return None

        if cur_taxon not in self.taxon_beds:
            logging.warning(f"Skipping {cur_taxon}: No BED file loaded.")
            return None

        # Annotate each gene in the taxon's BED with its corresponding ref gene
        for tmp_block_raw in tmp_blocks_raw:
            for tmp_gene, ref_gene, _ in tmp_block_raw:
                if tmp_gene in self.taxon_beds[cur_taxon]:
                    # This adds a 'ref_gene' attribute to the gene object
                    self.taxon_beds[cur_taxon][tmp_gene].set_newattr('ref_gene', ref_gene)

        # Convert raw blocks to 'anchors' objects
        tmp_blocks = [anchors(i, cur_taxon, _REF) for i in tmp_blocks_raw]

        for tmp_block in tmp_blocks:
            # Identify which REF regions are included in this block
            ref_genes_in_anchor = tmp_block.genes[_REF]
            tmp_covered_regions = identify_covered_block(ref_genes_in_anchor,
                                                         gene_region_dic=self.ref_genes,
                                                         covered_limit=covered_limit)

            for tmp_region_str in tmp_covered_regions:
                # Get the block_region object
                tmp_region_obj = self.ref_regions[tmp_region_str]

                # Increment the count for this species in this region
                tmp_region_obj.sp_dic[cur_taxon] += 1

                # Store the anchor object that provided this coverage
                tmp_region_obj.anchors_list.append(tmp_block)
        return None

    def get_TAXON_anchors(self, anchors_path: str, ref_id:str = 'AOG', covered_limit: int = 8) -> None:
        """Loads and parses all taxon anchor files from a directory.
        
        Args:
            anchors_path: Path to the directory containing .anchors files.
                          (Expects files like taxon.REF.anchors)
            covered_limit: Minimum genes per block to be considered covered.
            
        Sets Attributes:
            self.covered_limit (int): The coverage limit used.
            self.anchors_path (str): Path to the anchors directory.
        """
        self.covered_limit = covered_limit
        self.anchors_path = anchors_path
        # Example file name: taxon.REF.AOG.anchors
        anchor_files = glob.glob(os.path.join(anchors_path, f'*.{ref_id}.anchors'))
        for tmp_anchor_file in anchor_files:
            self.parse_TAXON_anchors(tmp_anchor_file, covered_limit=covered_limit)
        return None

    def count_TAXON_regions(self) -> None:
        """
        Aggregates anchor mappings to count co-occurrences between
        taxon chromosomes and reference chromosomes.
        
        Also stores the specific genomic coordinates of each mapping.
        
        Sets Attributes:
            self.taxon_chrom (defaultdict):
                {taxon: {taxon_chrom: {ref_chrom: count}}}.
            self.chrom_region (defaultdict):
                {taxon__chrom: [(ref_region_str, start, end)]}.
        """
        self.taxon_chrom = collections.defaultdict(double_layer_int_dict)
        self.chrom_region = collections.defaultdict(list)

        for tmp_region_str, tmp_region_obj in self.ref_regions.items():
            ref_chrom_id = tmp_region_str.split(':')[0]

            for tmp_anchor in tmp_region_obj.anchors_list:
                cur_taxon = tmp_anchor.sp[0]

                if self.taxon_beds.get(cur_taxon) is None:
                    # this taxon is not included in bed files
                    continue

                # Get location of the anchor block in the *query taxon*
                # based *only* on genes that map to the *current ref block*
                tmp_anchor.get_location(idx=0,
                                        gene_dic=self.taxon_beds[cur_taxon],
                                        query_genes_list=tmp_region_obj.Gene_List)

                cur_chrom = tmp_anchor.chrom[0]
                if not cur_chrom:
                    # get_location failed (e.g., no genes found)
                    continue

                # Increment count: this taxon_chrom maps to this ref_chrom
                self.taxon_chrom[cur_taxon][cur_chrom][ref_chrom_id] += 1

                # Store the actual region mapping
                self.chrom_region[f'{cur_taxon}__{cur_chrom}'].append(
                    (tmp_region_str, tmp_anchor.start[0], tmp_anchor.end[0])
                )
        return None

    def get_clade_dic(self, clade_path: str) -> None:
        """Loads a clade/species mapping file.
        
        File format: Clade_Group <tab> Clade_Name <tab> Species_Name
        
        Args:
            clade_path: Path to the clade mapping file.
            
        Sets Attributes:
            self.clade_path (str): Path to the clade file.
            self.clade_dic (defaultdict): {clade_group: [species_list]} and
                                          {clade_name: [species_list]}.
            self.taxon_clade_dic (Dict): {species_name: clade_name}.
        """
        self.clade_path = clade_path
        self.clade_dic = collections.defaultdict(list)
        self.taxon_clade_dic = {}
        with open(file=clade_path) as f:
            for tmp_line in f:
                if not tmp_line.strip():
                    continue
                try:
                    group, clade, species = tmp_line.strip().split('\t')
                    self.clade_dic[group].append(species)
                    self.clade_dic[clade].append(species)
                    self.taxon_clade_dic[species] = clade
                except ValueError:
                    logging.warning(f"Skipping malformed clade line: {tmp_line.strip()}")
        return None

    def get_chrom_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Converts the aggregated counts into a DataFrame.
        
        Sets Attributes:
            self.chrom_comp_df (pd.DataFrame): The main composition matrix.
                Rows: taxon__chrom, Columns: ref_chrom, Values: counts.
            self.chrom_map (pd.DataFrame): Metadata for chromosomes.
                Rows: taxon__chrom, Columns: ['taxon', 'clade'].
                
        Returns:
            A tuple (self.chrom_comp_df, self.chrom_map).
        """
        chr_dic = {}
        chr_meta = collections.defaultdict(dict)

        for tmp_taxon, taxon_chrom_dic in self.taxon_chrom.items():
            for tmp_chrom, chrom_comp in taxon_chrom_dic.items():
                chrom_full_id = f'{tmp_taxon}__{tmp_chrom}'
                chr_dic[chrom_full_id] = chrom_comp
                chr_meta[chrom_full_id]['taxon'] = tmp_taxon
                chr_meta[chrom_full_id]['clade'] = self.taxon_clade_dic.get(tmp_taxon, 'Unknown')

        self.chrom_comp_df = pd.DataFrame.from_dict(chr_dic, orient='index')
        self.chrom_comp_df.fillna(0, inplace=True)
        self.chrom_map = pd.DataFrame.from_dict(chr_meta, orient='index')

        return self.chrom_comp_df, self.chrom_map

    def filter_chrom_df(self, query_df: pd.DataFrame, region_limit: int = 5, replace: bool = True) -> pd.DataFrame:
        """Filters the chromosome composition DataFrame to remove rows
        (chromosomes) with low mapping counts.
        
        Args:
            query_df: The DataFrame to filter (e.g., self.chrom_comp_df).
            region_limit: The minimum row sum (total mappings) to keep.
            replace: If True, updates self.chrom_comp_df with the filtered one.
            
        Returns:
            The filtered DataFrame.
        """
        filtered_df = query_df[query_df.sum(axis=1, numeric_only=True) >= region_limit]
        if replace:
            self.chrom_comp_df = filtered_df
            self.chrom_map = self.chrom_map.loc[filtered_df.index]
        return filtered_df

    def get_KMeans_label(self, query_df: pd.DataFrame, N: int = 10,
                         random_state: int = 123, init: str = 'k-means++') -> pd.DataFrame:
        """Performs K-Means clustering on a DataFrame.
        
        Args:
            query_df: The DataFrame to cluster (e.g., self.chrom_comp_df).
            N: The number of clusters (k).
            random_state: Random state for reproducibility.
            init: K-Means initialization method.
            
        Sets Attributes:
            self.KM_labels (pd.DataFrame): DataFrame of cluster labels.
            
        Returns:
            The labels DataFrame.
        """
        clus_M = KMeans(n_clusters=N, init=init, random_state=random_state, n_init=10)
        clus_M.fit(query_df)
        self.KM_labels = pd.DataFrame(clus_M.predict(query_df), index=query_df.index, columns=['label'])
        return self.KM_labels

    def get_tSNE_dimension(self, query_df: pd.DataFrame, N: int = 3,
                           n_iter: int = 300, random_state: int = 123) -> pd.DataFrame:
        """Performs t-SNE dimensionality reduction.
        
        Args:
            query_df: The DataFrame to reduce.
            N: The number of dimensions to reduce to.
            n_iter: Number of iterations.
            random_state: Random state for reproducibility.
            
        Sets Attributes:
            self.tSNE_D (pd.DataFrame): The t-SNE embedding DataFrame.
            
        Returns:
            The t-SNE embedding DataFrame.
        """
        tSNE_M = TSNE(n_components=N, n_iter=n_iter, random_state=random_state , n_jobs=1)
        tSNE_res = tSNE_M.fit_transform(query_df)
        self.tSNE_D = pd.DataFrame(tSNE_res, index=query_df.index)
        return self.tSNE_D

    def get_chrom_sim_df(self) -> pd.DataFrame:
        """Calculates the similarity matrix from the chromosome composition matrix
        using the optimized similarity function.
        
        Sets Attributes:
            self.chrom_sim_df (pd.DataFrame): The resulting similarity matrix.
            
        Returns:
            The similarity DataFrame.
        """
        self.chrom_sim_df = get_sim_df_optimized(self.chrom_comp_df)
        return self.chrom_sim_df

    def plot_cluster_heatmap(self, query_df: Optional[pd.DataFrame] = None,
                             plot_flag: bool = False, new_chrom: Optional[Set[str]] = None) -> None:
        """
        Performs hierarchical clustering on a similarity matrix and
        optionally plots a clustered heatmap.
        
        Args:
            query_df: The similarity matrix to use (default: self.chrom_sim_df).
            plot_flag: If True, generate and show the clustermap.
            new_chrom: A set of chromosome IDs to label as 'new'
                       in the row colors (optional).
                       
        Sets Attributes:
            self.hierachy_group (Dict): {chrom_id: group_color_label}.
            self.grouped_chroms (defaultdict): {group_color_label: [chrom_list]}.
            self.hierachy_seq (List): Sorted list of chrom_ids from clustering.
            self.clade_color (color_pan): color_pan object used for clades.
        """
        if query_df is None:
            query_df = self.chrom_sim_df

        # Get the clustering order and group colors
        sp_seq, group_seq = get_hierarchical_clustering_order(query_df=query_df)

        # Re-sort chromosomes within each cluster group by their clade
        # This is complex and was commented out in the original.
        # Keeping it faithful to the original active code.
        
        # Re-get the group_seq in the new sorted order
        group_seq_map = dict(zip(sp_seq, group_seq))
        group_seq = [group_seq_map[sp] for sp in sp_seq]

        if plot_flag:
            # Create row colors based on clade (and 'new_chrom' if provided)
            tmp_colors = color_pan(list(sns.color_palette(palette='Set3', n_colors=20)))

            if new_chrom:
                r_c_data = []
                for i in sp_seq:
                    taxon = i.split('__')[0]
                    clade = self.taxon_clade_dic.get(taxon, 'Unknown')
                    is_new_label = 'new' if i in new_chrom else 'old'
                    r_c_data.append((tmp_colors[clade], tmp_colors[is_new_label]))
                    
                r_c = pd.DataFrame(r_c_data,
                                   columns=['clade', 'is_new'],
                                   index=sp_seq)
            else:
                r_c_data = [
                    tmp_colors[self.taxon_clade_dic.get(i.split('__')[0], 'Unknown')]
                    for i in sp_seq
                ]
                r_c = pd.Series(r_c_data, index=sp_seq, name='clade')

            g = sns.clustermap(
                query_df.loc[sp_seq, sp_seq], cmap='Oranges',
                xticklabels=[], yticklabels=[],
                row_cluster=False, col_cluster=False,
                col_colors=group_seq,
                row_colors=r_c
            )

            # Add a custom legend for the row colors (clades)
            ax = g.ax_heatmap
            legend_handles = [
                patches.Patch(color=tmp_colors[tmp_clade], label=tmp_clade)
                for tmp_clade in tmp_colors.pan.keys()
                if tmp_clade not in ['new', 'old']  # Only show clades
            ]
            plt.legend(handles=legend_handles, bbox_to_anchor=(0, 0.2),
                       loc='upper left', title="Clades")
            plt.show()
            self.clade_color = tmp_colors

        self.hierachy_group = dict(zip(sp_seq, group_seq))
        self.grouped_chroms = collections.defaultdict(list)
        for tmp_chrom, tmp_group in self.hierachy_group.items():
            self.grouped_chroms[tmp_group].append(tmp_chrom)
        self.hierachy_seq = sp_seq
        return None

    def get_hierarchy_cnt(self, query_classifcation: Optional[Dict[str, str]] = None) -> None:
        """Counts the number of chromosomes from each clade within
        each synteny group (from clustering).
        
        Args:
            query_classifcation: A {chrom_id: group_id} mapping.
                                 Default: self.hierachy_group.
                                 
        Sets Attributes:
            self.hierachy_cnt (defaultdict): {group_id: {clade: count}}.
        """
        if query_classifcation is None:
            query_classifcation = self.hierachy_group

        self.hierachy_cnt = collections.defaultdict(dict)
        for tmp_chrom, tmp_hier in query_classifcation.items():
            if tmp_chrom in self.chrom_map.index:
                tmp_clade = self.chrom_map.loc[tmp_chrom, 'clade']
                self.hierachy_cnt[tmp_hier][tmp_clade] = self.hierachy_cnt[tmp_hier].get(tmp_clade, 0) + 1
            else:
                logging.warning(f"Warning: Chromosome {tmp_chrom} not found in chrom_map.")
        return None

    def set_all_genes_OG(self, ref_dic: Dict[str, Dict[str, str]]) -> None:
        """Attaches orthogroup (OG) information to each gene object
        in self.taxon_beds.
        
        (Requires self.get_OG_dic() to be run first)
        
        Args:
            ref_dic: A dictionary (like self.OG_dic) to check which
                     taxa are present in the OG file.
                     
        Sets Attributes:
            self.Jacccard_taxons (List): A list of taxon names that
                                        had OGs assigned.
        """
        self.Jacccard_taxons = []
        for tmp_taxon in self.taxon_beds.keys():
            if ref_dic.get(tmp_taxon) is None:
                logging.info(f'{tmp_taxon} is skipped (no OGs found)')
                continue

            self.Jacccard_taxons.append(tmp_taxon)
            for tmp_gene in self.taxon_beds[tmp_taxon].values():
                og_val = self.OG_dic[tmp_taxon].get(tmp_gene.id)
                tmp_gene.set_newattr('OG', og_val)
        return None

    def get_chrom_Jaccard_sim(self) -> None:
        """Calculates Jaccard similarity between all pairs of chromosomes
        based on their shared orthogroups (OGs).
        
        (Requires self.set_all_genes_OG() to be run first)
        
        Sets Attributes:
            self.chrom_OGs (defaultdict): {taxon__chrom: {set_of_OGs}}.
            self.chrom_Jaccard_df (pd.DataFrame): Square similarity matrix
                                                  based on Jaccard.
        """
        self.chrom_OGs = collections.defaultdict(set)
        for tmp_taxon in self.Jacccard_taxons:
            for tmp_gene in self.taxon_beds[tmp_taxon].values():
                if hasattr(tmp_gene, 'OG') and tmp_gene.OG:
                    self.chrom_OGs[f'{tmp_taxon}__{tmp_gene.chrom}'].add(tmp_gene.OG)

        # Get the list of chromosomes from the *block* similarity matrix
        # to ensure compatibility
        chrom_index = self.chrom_sim_df.index
        self.chrom_Jaccard_df = pd.DataFrame(index=chrom_index, columns=chrom_index, dtype=float)

        for i in range(len(chrom_index)):
            tax_i = chrom_index[i]
            og_set_i = self.chrom_OGs[tax_i]
            for j in range(i, len(chrom_index)):
                tax_j = chrom_index[j]
                og_set_j = self.chrom_OGs[tax_j]
                
                sim = calculate_jaccard_similarity(og_set_i, og_set_j)

                self.chrom_Jaccard_df.loc[tax_i, tax_j] = sim
                self.chrom_Jaccard_df.loc[tax_j, tax_i] = sim
        return None

    def get_chromosomes_group_similarity(self,
                                         grouped_chroms: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """Calculates the mean similarity *between* synteny groups.
        
        Args:
            grouped_chroms: A {group_id: [chrom_list]} mapping.
                            Default: self.grouped_chroms.
                            
        Sets Attributes:
            self.group_sim (pd.DataFrame): A square matrix of
                                          inter-group similarities.
                                          
        Returns:
            The inter-group similarity DataFrame.
        """
        grouped_chroms = grouped_chroms if grouped_chroms else self.grouped_chroms
        group_keys = list(grouped_chroms.keys())
        self.group_sim = pd.DataFrame(index=group_keys, columns=group_keys, dtype=float)

        for tmp_group1 in group_keys:
            chroms1 = grouped_chroms[tmp_group1]
            for tmp_group2 in group_keys:
                chroms2 = grouped_chroms[tmp_group2]
                
                # Calculate mean similarity between all chroms in group1
                # and all chroms in group2
                if not chroms1 or not chroms2:
                    sim = np.nan
                else:
                    sim = self.chrom_sim_df.loc[chroms1, chroms2].mean().mean()
                self.group_sim.loc[tmp_group1, tmp_group2] = sim
        return self.group_sim

    def reannotate_ref_regions(self) -> None:
        """
        Re-annotates each reference region with its representative
        chromosome cluster.
        
        The representative cluster is determined by the most frequent
        cluster mapping to this region across all query anchors.
        
        (Requires self.plot_cluster_heatmap() to be run first to generate
        self.hierachy_group)
        """
        for tmp_region in self.ref_regions.values():
            # Find the cluster for each anchor mapping to this region
            cluster_mappings = [
                self.hierachy_group.get(f'{i.sp[0]}__{i.chrom[0]}')
                for i in tmp_region.anchors_list
            ]
            # Filter out None values (e.g., chromosomes that were filtered out)
            valid_clusters = [c for c in cluster_mappings if c is not None]

            if valid_clusters:
                tmp_chrom_cnt = collections.Counter(valid_clusters)
                # Find the cluster with the highest count
                represent_chrom_clust = tmp_chrom_cnt.most_common(1)[0][0]
                tmp_region.chrom_clust = represent_chrom_clust
        return None

    def plot_network_linkage(self, chromosome_df: Optional[pd.DataFrame] = None,
                             chromosome_group_dic: Optional[Dict[str, str]] = None,
                             sim_limit: float = 0.55, ax=None) -> plt.Axes:
        """Plots a network graph of chromosome similarities.

        Nodes are chromosomes, colored by cluster group. Edges are drawn
        if similarity exceeds `sim_limit`.

        Args:
            chromosome_df: The similarity matrix (default: self.chrom_sim_df).
            chromosome_group_dic: Mapping of {chrom: group}
                                  (default: self.hierachy_group).
            sim_limit: Similarity threshold for drawing an edge.
            ax: A matplotlib Axes object to plot on (optional).

        Returns:
            The matplotlib Axes object.
        """
        G = nx.Graph()
        if chromosome_df is None:
            chromosome_df = self.chrom_sim_df
        if chromosome_group_dic is None:
            chromosome_group_dic = self.hierachy_group

        # Add nodes
        for chromosome in chromosome_df.index:
            if chromosome in chromosome_group_dic:
                G.add_node(chromosome, group=chromosome_group_dic[chromosome])

        # Add edges
        for i, row in chromosome_df.iterrows():
            if i not in G.nodes: continue
            for j, similarity in row.items():
                if j not in G.nodes: continue
                if i >= j:  # Only add edges once
                    continue

                if similarity > sim_limit:
                    G.add_edge(i, j, weight=similarity, alpha=similarity ** 2)

        # Node colors and layout
        node_groups = nx.get_node_attributes(G, 'group')
        unique_groups = sorted(set(node_groups.values()))
        color_map = {group: plt.cm.Set3(i % 12) for i, group in enumerate(unique_groups)}
        node_colors = [color_map[node_groups[node]] for node in G.nodes()]

        # Use spring layout, considering edge weights
        pos = nx.spring_layout(G, weight='weight', k=0.1, iterations=50)

        # Create or use specified ax
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 12))

        # Draw edges with alpha based on similarity
        edges = G.edges()
        edge_alphas = [G[u][v].get('alpha', 0) for u, v in edges]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color='gray', alpha=edge_alphas)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=8)

        # Add labels only for specific taxa (e.g., Oryza_sativa)
        labels = {i: i.split('__')[-1] for i in G.nodes if i.startswith('Oryza_sativa')}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10)

        # Legend
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markersize=10, label=group)
            for group, color in color_map.items()
        ]
        ax.legend(handles=legend_handles, title="Groups", loc="best")
        ax.set_title(f'Chromosome Clustering Graph (sim_limit={sim_limit})')
        ax.axis('off')
        return ax


def build_ref_mapping_object(ref_bed: str, bed_dir: str, anchor_dir: str,
                             window_length: int = 20, covered_limit: int = 8) -> REF_MAPPING:
    """A helper function to build and initialize a REF_MAPPING object.
    
    This runs the core data loading and processing pipeline.

    Args:
        ref_bed: Path to the reference BED file.
        bed_dir: Directory containing taxon BED files.
        anchor_dir: Directory containing taxon anchor files.
        window_length: Number of genes per reference block.
        covered_limit: Minimum genes per block to be considered covered.

    Returns:
        A fully initialized REF_MAPPING object.
    """
    am = REF_MAPPING()
    logging.info("Loading reference regions...")
    am.get_REF_regions(ref_path=ref_bed, window_length=window_length)
    ref_id = os.path.basename(ref_bed).split('.')[0]
    logging.info("Loading taxon BED files...")
    am.get_TAXON_beds(bed_path=bed_dir)
    logging.info("Loading and parsing taxon anchors...")
    am.get_TAXON_anchors(anchors_path=anchor_dir, covered_limit=covered_limit , ref_id=ref_id)
    logging.info("Counting taxon region mappings...")
    am.count_TAXON_regions()
    logging.info("Building chromosome composition DataFrame...")
    am.get_chrom_df()
    logging.info("Build complete.")
    return am


def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Reference-based synteny mapping analysis.")
    parser.add_argument('--ref_bed', type=str, required=True,
                        help='Path to the reference BED-like file (col 0: chrom, col 3: gene).')
    parser.add_argument('--bed_dir', type=str, required=True,
                        help='Directory containing taxon BED files (e.g., taxon1.bed, taxon2.bed).')
    parser.add_argument('--anchor_dir', type=str, required=True,
                        help='Directory containing taxon anchor files (e.g., taxon1.ref.AOG.anchors).')
    parser.add_argument('--window_length', type=int, default=20,
                        help='Number of genes per reference block.')
    parser.add_argument('--covered_limit', type=int, default=8,
                        help='Minimum genes per block to be considered covered.')
    parser.add_argument('--region_limit', type=int, default=5,
                        help='Minimum mapping count to keep a chromosome for analysis.')
    parser.add_argument('--output_prefix', type=str, default='chromosome_analysis',
                        help='Prefix for output files.')
    parser.add_argument('--reannote_ref', action='store_true',
                        help='Re-annotate reference regions with cluster info and write extra output files.')
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main execution function for the synteny analysis pipeline.

    Args:
        args: Parsed command-line arguments from argparse.
    """
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Build the main analysis object
    logging.info("Building the main analysis object...")
    tmp_am = build_ref_mapping_object(
        ref_bed=args.ref_bed,
        bed_dir=args.bed_dir,
        anchor_dir=args.anchor_dir,
        window_length=args.window_length,
        covered_limit=args.covered_limit
    )

    # Filter chromosomes with low mapping counts
    logging.info(f"Initial chromosomes: {tmp_am.chrom_comp_df.shape[0]}")
    tmp_am.filter_chrom_df(query_df=tmp_am.chrom_comp_df,
                            region_limit=args.region_limit, replace=True)
    logging.info(f"Filtered chromosomes (>= {args.region_limit} mappings): {tmp_am.chrom_comp_df.shape[0]}")

    if tmp_am.chrom_comp_df.empty:
        logging.warning("No chromosomes passed filtering. Exiting.")
        return

    # Calculate the chromosome-chromosome similarity matrix
    logging.info("Calculating chromosome similarity matrix...")
    tmp_am.get_chrom_sim_df()

    # Perform hierarchical clustering for all chromosomes
    logging.info("Performing hierarchical clustering...")
    tmp_am.plot_cluster_heatmap(query_df=tmp_am.chrom_sim_df, plot_flag=False)

    # Calculate similarity between the clusters
    # tmp_am.get_chromosomes_group_similarity()

    # Save UMAP-reduced coordinates with cluster annotations
    logging.info("Calculating UMAP and saving chromosome annotations...")
    umap_df = get_umap_df(tmp_am.chrom_sim_df, n_components=2, plot_flag=False)
    chrom_annot_df = tmp_am.chrom_comp_df.join(umap_df)
    chrom_annot_df['clust'] = chrom_annot_df.index.map(tmp_am.hierachy_group)
    output_path = f'{args.output_prefix}_chromosomes_annot.tsv'
    chrom_annot_df.to_csv(output_path, sep='\t')
    logging.info(f"Saved chromosome annotations to: {output_path}")

    # Merge labeled regions for each chromosome
    logging.info("Merging labeled regions for query chromosomes...")
    chrom_region_labels = {}
    for tmp_chrom, tmp_comp in tmp_am.chrom_region.items():
        if tmp_chrom not in tmp_am.chrom_comp_df.index: # Skip filtered chromosomes
             continue
        # Labels are the reference chromosome (e.g., 'Os01')
        tmp_labeled_regions = merge_labeled_regions(
            [(a.split(':')[0], b, c) for a, b, c in tmp_comp]
        )
        chrom_region_labels[tmp_chrom] = tmp_labeled_regions

    output_path = f'{args.output_prefix}_chromosome_labeled_regions.tsv'
    with open(output_path, 'w') as f:
        f.write('chromosome\tstart\tend\tref_chrom_labels\n')
        for tmp_chrom in sorted(chrom_region_labels.keys()):
            for label, start, end in chrom_region_labels[tmp_chrom]:
                f.write(f"{tmp_chrom}\t{start}\t{end}\t{label}\n")
    logging.info(f"Saved query chromosome labeled regions to: {output_path}")

    if args.reannote_ref:
        logging.info("Re-annotating reference regions with cluster information...")
        # Re-annotate reference regions with cluster info
        tmp_am.reannotate_ref_regions()

        output_path = f'{args.output_prefix}_ref_regions_reannotated.tsv'
        with open(output_path, 'w') as f:
            f.write('region\tchromosome_cluster\n')
            for tmp_region in tmp_am.ref_regions.values():
                f.write(f"{str(tmp_region)}\t{getattr(tmp_region, 'chrom_clust', 'NA')}\n")
        logging.info(f"Saved re-annotated reference regions to: {output_path}")

        # Merge labeled regions for each chromosome based on *cluster*
        logging.info("Merging re-annotated labeled regions for query chromosomes...")
        reannot_chrom_region_labels = {}
        for tmp_chrom, tmp_comp in tmp_am.chrom_region.items():
            if tmp_chrom not in tmp_am.chrom_comp_df.index: # Skip filtered chromosomes
                 continue
            tmp_labeled_regions = []
            for tmp_region_id, start, end in tmp_comp:
                cluster_label = getattr(tmp_am.ref_regions.get(tmp_region_id), 'chrom_clust', None)
                if cluster_label is None:
                    continue
                tmp_labeled_regions.append((cluster_label, start, end))
            
            if tmp_labeled_regions:
                tmp_labeled_regions = merge_labeled_regions(tmp_labeled_regions)
            reannot_chrom_region_labels[tmp_chrom] = tmp_labeled_regions

        output_path = f'{args.output_prefix}_reannot_chromosome_labeled_regions.tsv'
        with open(output_path, 'w') as f:
            f.write('chromosome\tstart\tend\tcluster_labels\n')
            for tmp_chrom in sorted(reannot_chrom_region_labels.keys()):
                for label, start, end in reannot_chrom_region_labels[tmp_chrom]:
                    f.write(f"{tmp_chrom}\t{start}\t{end}\t{label}\n")
        logging.info(f"Saved query chromosome cluster-labeled regions to: {output_path}")

    logging.info("Analysis finished.")


if __name__ == "__main__":
    args = get_args()
    main(args)