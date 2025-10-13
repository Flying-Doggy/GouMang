#!/usr/bin/env python3
"""
Genome Block Splitting Tool
Automatically identifies chromosome breakpoints and splits genomes based on collinearity analysis
"""

import os
import sys
import argparse
import logging
import bisect
from collections import defaultdict
from sortedcontainers import SortedList
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Gene:
    """Represents a gene with its genomic coordinates and properties"""
    
    def __init__(self, gene_id, chromosome, start, end, strand, species):
        """
        Initialize a Gene object
        
        Args:
            gene_id (str): Unique identifier for the gene
            chromosome (str): Chromosome where the gene is located
            start (int): Start position of the gene
            end (int): End position of the gene
            strand (str): Strand orientation (+ or -)
            species (str): Species to which the gene belongs
        """
        self.id = gene_id
        self.chromosome = chromosome
        self.start = int(start)
        self.end = int(end)
        self.strand = strand
        self.species = species
        self.position = 0  # Sequential position on the chromosome
    
    def __repr__(self):
        return f"Gene({self.id}, {self.chromosome}, pos:{self.position})"


class Chromosome:
    """Represents a chromosome with its genes and genomic features"""
    
    def __init__(self, chrom_id, species):
        """
        Initialize a Chromosome object
        
        Args:
            chrom_id (str): Chromosome identifier
            species (str): Species to which the chromosome belongs
        """
        self.id = chrom_id
        self.species = species
        self.genes = SortedList(key=lambda x: (x.start, x.id))
        self.blocks = []  # Collinear blocks (start_pos, end_pos, block_size)
        self.split_points = []  # Breakpoint positions
        self.interaction_matrix = None
        self.insulation_scores = None
        self.insulation_windows = None  # Store window positions for insulation scores
    
    def add_gene(self, gene):
        """Add a gene to the chromosome"""
        self.genes.add(gene)
    
    def assign_gene_positions(self):
        """Assign sequential positions to genes on the chromosome"""
        for idx, gene in enumerate(self.genes):
            gene.position = idx
    
    def __len__(self):
        return len(self.genes)
    
    def __repr__(self):
        return f"Chromosome({self.species}.{self.id}, genes:{len(self.genes)})"


class GenomeSplitter:
    """Main class for splitting genomes based on collinearity analysis"""
    
    def __init__(self):
        self.species_genes = defaultdict(set)  # Collinear genes for each species
        self.genomes = defaultdict(dict)  # species -> {chrom_id: Chromosome}
        self.all_blocks = []  # All collinear blocks
    
    def load_anchors(self, anchors_dir):
        """
        Load anchors files to identify collinear genes
        
        Args:
            anchors_dir (str): Directory containing anchors files
        """
        logger.info("Loading anchors files from %s", anchors_dir)
        
        if not os.path.exists(anchors_dir):
            raise FileNotFoundError(f"Anchors directory not found: {anchors_dir}")
        
        anchor_files = [f for f in os.listdir(anchors_dir) 
                       if f.endswith('.anchors') and 'lifted' not in f]
        
        if not anchor_files:
            raise ValueError(f"No anchors files found in {anchors_dir}")
        
        for anchor_file in anchor_files:
            file_path = os.path.join(anchors_dir, anchor_file)
            species_a, species_b, _ = os.path.basename(anchor_file).split('.')[:3]
            
            logger.debug("Processing anchors file: %s", anchor_file)
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    gene_a, gene_b, score = line.strip().split('\t')
                    self.species_genes[species_a].add(gene_a)
                    self.species_genes[species_b].add(gene_b)
        
        # Log statistics
        for species, genes in self.species_genes.items():
            logger.info("Species %s: %d collinear genes", species, len(genes))
    
    def load_genomes(self, bed_dir):
        """
        Load BED files to build genome structures
        
        Args:
            bed_dir (str): Directory containing BED files
        """
        logger.info("Loading BED files from %s", bed_dir)
        
        if not os.path.exists(bed_dir):
            raise FileNotFoundError(f"BED directory not found: {bed_dir}")
        
        bed_files = [f for f in os.listdir(bed_dir) if f.endswith('.bed')]
        
        if not bed_files:
            raise ValueError(f"No BED files found in {bed_dir}")
        
        for bed_file in bed_files:
            species = os.path.basename(bed_file).replace('.bed', '')
            file_path = os.path.join(bed_dir, bed_file)
            
            logger.debug("Processing BED file: %s", bed_file)
            
            # Only load genes that appear in collinearity analysis
            collinear_genes = self.species_genes.get(species, set())
            chromosomes = {}
            gene_count = 0
            
            with open(file_path, 'r') as f:
                for line in f:
                    chrom, start, end, gene_id, _, strand = line.strip().split()
                    
                    if gene_id not in collinear_genes:
                        continue
                    
                    if chrom not in chromosomes:
                        chromosomes[chrom] = Chromosome(chrom, species)
                    
                    gene = Gene(gene_id, chrom, start, end, strand, species)
                    chromosomes[chrom].add_gene(gene)
                    gene_count += 1
            
            # Assign positions to genes on each chromosome
            for chrom in chromosomes.values():
                chrom.assign_gene_positions()
            
            self.genomes[species] = chromosomes
            logger.info("Loaded %s: %d chromosomes, %d collinear genes", 
                       species, len(chromosomes), gene_count)
    
    def build_blocks(self, anchors_dir, min_block_size=10):
        """
        Build collinear blocks from anchors files
        
        Args:
            anchors_dir (str): Directory containing anchors files
            min_block_size (int): Minimum number of gene pairs to form a block
        """
        logger.info("Building collinear blocks from anchors")
        
        anchor_files = [f for f in os.listdir(anchors_dir) 
                       if f.endswith('.anchors') and 'lifted' not in f]
        
        processed_pairs = set()
        
        for anchor_file in anchor_files:
            file_path = os.path.join(anchors_dir, anchor_file)
            species_a, species_b, _ = os.path.basename(anchor_file).split('.')[:3]
            
            # Avoid processing the same species pair multiple times
            pair_key = tuple(sorted([species_a, species_b]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            blocks = []
            current_block = None
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        if current_block and len(current_block) >= min_block_size:
                            blocks.append(current_block)
                        current_block = []
                    else:
                        gene_a, gene_b, score = line.strip().split('\t')
                        
                        # Find gene objects
                        gene_obj_a = self._find_gene(species_a, gene_a)
                        gene_obj_b = self._find_gene(species_b, gene_b)
                        
                        if gene_obj_a and gene_obj_b and current_block is not None:
                            current_block.append((gene_obj_a, gene_obj_b, int(score)))
            
            # Add the last block
            if current_block and len(current_block) >= min_block_size:
                blocks.append(current_block)
            
            # Record block information
            for block in blocks:
                self._record_block(block)
            
            logger.debug("Processed %s vs %s: %d blocks", species_a, species_b, len(blocks))
        
        logger.info("Total blocks built: %d", len(self.all_blocks))
    
    def _find_gene(self, species, gene_id):
        """
        Find a gene object by species and gene ID
        
        Args:
            species (str): Species name
            gene_id (str): Gene identifier
            
        Returns:
            Gene: Gene object if found, None otherwise
        """
        if species not in self.genomes:
            return None
        
        for chromosome in self.genomes[species].values():
            for gene in chromosome.genes:
                if gene.id == gene_id:
                    return gene
        return None
    
    def _record_block(self, block):
        """
        Record block information to chromosomes
        
        Args:
            block (list): List of gene pairs in the block
        """
        if not block:
            return
        
        # Get the range of the block in both species
        positions_a = [gene_a.position for gene_a, _, _ in block]
        positions_b = [gene_b.position for _, gene_b, _ in block]
        
        start_a, end_a = min(positions_a), max(positions_a)
        start_b, end_b = min(positions_b), max(positions_b)
        
        # Assume the chromosome of the first gene pair represents the whole block's chromosome
        chrom_a = block[0][0].chromosome
        chrom_b = block[0][1].chromosome
        species_a = block[0][0].species
        species_b = block[0][1].species
        
        # Record to chromosomes
        if species_a in self.genomes and chrom_a in self.genomes[species_a]:
            self.genomes[species_a][chrom_a].blocks.append((start_a, end_a, len(block)))
        
        if species_b in self.genomes and chrom_b in self.genomes[species_b]:
            self.genomes[species_b][chrom_b].blocks.append((start_b, end_b, len(block)))
        
        self.all_blocks.append({
            'species_a': species_a, 'chrom_a': chrom_a, 'start_a': start_a, 'end_a': end_a,
            'species_b': species_b, 'chrom_b': chrom_b, 'start_b': start_b, 'end_b': end_b,
            'size': len(block)
        })
    
    def calculate_interaction_matrix(self, chromosome, window_size=20):
        """
        Calculate interaction matrix for a chromosome
        
        Args:
            chromosome (Chromosome): Chromosome object
            window_size (int): Size of windows for binning genes
            
        Returns:
            numpy.ndarray: Interaction matrix
        """
        if len(chromosome) == 0:
            return np.zeros((1, 1))
        
        matrix_size = len(chromosome) // window_size + 1
        interaction_matrix = np.zeros((matrix_size, matrix_size))
        
        for start, end, block_size in chromosome.blocks:
            # Use log transformation to enhance weight of large blocks
            weight = np.log2(block_size)
            bin_start = start // window_size
            bin_end = end // window_size
            
            if bin_start < matrix_size and bin_end < matrix_size:
                interaction_matrix[bin_start, bin_end] += weight
        
        return interaction_matrix
    
    def calculate_insulation_scores(self, interaction_matrix, window_size=5):
        """
        Calculate insulation scores from interaction matrix
        
        Args:
            interaction_matrix (numpy.ndarray): Interaction matrix
            window_size (int): Window size for insulation score calculation
            
        Returns:
            tuple: (insulation_scores, window_positions)
        """
        n_bins = interaction_matrix.shape[0]
        insulation_scores = np.zeros(n_bins)
        window_positions = list(range(n_bins))
        
        for i in range(n_bins):
            # Upstream window
            upstream_start = max(0, i - window_size)
            upstream_end = i
            upstream_sum = np.sum(interaction_matrix[upstream_start:upstream_end, i])
            
            # Downstream window
            downstream_start = i + 1
            downstream_end = min(n_bins, i + window_size + 1)
            downstream_sum = np.sum(interaction_matrix[i, downstream_start:downstream_end])
            
            # Calculate insulation score
            total = upstream_sum + downstream_sum
            if total > 0:
                insulation_scores[i] = (downstream_sum - upstream_sum) / total
            else:
                insulation_scores[i] = 0
        
        return insulation_scores, window_positions
    
    def find_split_points(self, chromosome, insulation_scores, threshold=-0.65, window_size=20):
        """
        Find split points based on insulation scores
        
        Args:
            chromosome (Chromosome): Chromosome object
            insulation_scores (numpy.ndarray): Array of insulation scores
            threshold (float): Threshold for identifying breakpoints
            window_size (int): Window size used for interaction matrix
            
        Returns:
            list: List of split point positions
        """
        split_points = []
        prev_point = 0
        
        for i, score in enumerate(insulation_scores):
            if score < threshold:
                split_points.append(prev_point * window_size)
                prev_point = i
        
        # Add the last point
        if prev_point < len(insulation_scores):
            split_points.append(prev_point * window_size)
        
        # Add chromosome end
        if len(chromosome.genes) > 0:
            split_points.append(len(chromosome.genes) - 1)
        
        # Remove duplicates and sort
        split_points = sorted(set(split_points))
        return split_points
    
    def split_chromosomes(self, insulation_threshold=-0.65, min_chrom_length=500):
        """
        Split all eligible chromosomes based on insulation threshold
        
        Args:
            insulation_threshold (float): Threshold for insulation scores
            min_chrom_length (int): Minimum chromosome length to process
            
        Returns:
            list: List of split results
        """
        logger.info("Splitting chromosomes with threshold: %f", insulation_threshold)
        
        split_results = []
        
        for species, chromosomes in self.genomes.items():
            for chrom_id, chromosome in chromosomes.items():
                if len(chromosome) < min_chrom_length:
                    continue
                
                logger.debug("Splitting chromosome: %s.%s", species, chrom_id)
                
                # Calculate interaction matrix and insulation scores
                interaction_matrix = self.calculate_interaction_matrix(chromosome)
                insulation_scores, window_positions = self.calculate_insulation_scores(interaction_matrix)
                
                # Store insulation scores and window positions
                chromosome.interaction_matrix = interaction_matrix
                chromosome.insulation_scores = insulation_scores
                chromosome.insulation_windows = window_positions
                
                # Find split points
                split_points = self.find_split_points(chromosome, insulation_scores, insulation_threshold)
                chromosome.split_points = split_points
                
                # Generate split regions
                for i in range(len(split_points) - 1):
                    start = split_points[i]
                    end = split_points[i + 1]
                    
                    split_results.append({
                        'species': species,
                        'chromosome': chrom_id,
                        'block_id': i,
                        'start_position': start,
                        'end_position': end,
                        'num_genes': end - start + 1,
                        'genes': [g.id for g in chromosome.genes[start:end+1]],
                        'insulation_threshold': insulation_threshold
                    })
        
        return split_results
    
    def evaluate_splits(self, insulation_threshold):
        """
        Evaluate the quality of splits for a given insulation threshold
        
        Args:
            insulation_threshold (float): Insulation threshold used for splitting
            
        Returns:
            list: List of evaluation results
        """
        logger.info("Evaluating split quality for threshold: %f", insulation_threshold)
        
        evaluation_results = []
        
        for species, chromosomes in self.genomes.items():
            for chrom_id, chromosome in chromosomes.items():
                if not chromosome.split_points or len(chromosome.blocks) == 0:
                    continue
                
                # Convert split points to set for fast lookup
                split_set = set(chromosome.split_points)
                
                # Count blocks that completely fall within split intervals
                supported_blocks = 0
                for start, end, size in chromosome.blocks:
                    # Find the closest split points to block start and end
                    start_idx = bisect.bisect_left(chromosome.split_points, start)
                    end_idx = bisect.bisect_left(chromosome.split_points, end)
                    
                    # If start and end are in the same interval, it's supported
                    if start_idx == end_idx:
                        supported_blocks += 1
                
                support_ratio = supported_blocks / len(chromosome.blocks)
                
                evaluation_results.append({
                    'species': species,
                    'chromosome': chrom_id,
                    'insulation_threshold': insulation_threshold,
                    'total_blocks': len(chromosome.blocks),
                    'supported_blocks': supported_blocks,
                    'support_ratio': support_ratio,
                    'split_points': len(chromosome.split_points) - 1,  # Subtract ends
                    'chromosome_length': len(chromosome)
                })
        
        return evaluation_results
    
    def calculate_comprehensive_score(self, evaluation_results):
        """
        Calculate a comprehensive score for splitting quality using the formula:
        SegmentScore = log(splitted_regions) * (support_ratio)^2
        
        This formula balances the number of segments and the support ratio,
        giving more weight to solutions with both good segmentation and high support.
        
        Args:
            evaluation_results (list): List of evaluation results
            
        Returns:
            float: Comprehensive score
        """
        if not evaluation_results:
            return 0.0
        
        # Calculate weighted average support ratio
        total_blocks = sum(r['total_blocks'] for r in evaluation_results)
        if total_blocks == 0:
            return 0.0
        
        # Calculate the comprehensive score using the formula:
        # SegmentScore = log(splitted_regions) * (support_ratio)^2
        # We use weighted average based on the number of blocks in each chromosome
        weighted_scores = []
        
        for result in evaluation_results:

            splitted_regions = result['split_points'] + 1  # split_points is segments-1, so add 1 to get segments
            support_ratio = result['support_ratio']
            num_blocks = result['total_blocks']
            
            # Avoid log(0) by adding a small constant
            if splitted_regions <= 0:
                segment_score = 0.0
            else:
                segment_score = np.log(splitted_regions)
            
            # Calculate the chromosome's score using the formula
            chromosome_score = segment_score * (support_ratio ** 2)

            # Weight by the number of blocks in this chromosome
            weighted_scores.append(chromosome_score)
        
        # Calculate weighted average
        comprehensive_score = sum(weighted_scores) / len(weighted_scores)
        
        return comprehensive_score


    def calculate_comprehensive_score_linear(self, evaluation_results):
        """
        Calculate a comprehensive score for splitting quality,
        Assume that the number of chromosome segments is a defined number (default is 5).
        
        Args:
            evaluation_results (list): List of evaluation results
            
        Returns:
            float: Comprehensive score
        """
        if not evaluation_results:
            return 0.0
        
        # Calculate weighted average support ratio
        total_blocks = sum(r['total_blocks'] for r in evaluation_results)
        if total_blocks == 0:
            return 0.0
        
        weighted_support = sum(r['support_ratio'] * r['total_blocks'] for r in evaluation_results) / total_blocks
        
        # Calculate segmentation score (more segments might be better but not too many)
        avg_segments = np.mean([r['split_points'] for r in evaluation_results])
        max_segments = max(r['split_points'] for r in evaluation_results) if evaluation_results else 1
        
        # Normalize segment count (prefer moderate number of segments)
        segment_score = 1.0 - abs(avg_segments - 5) / max(5, max_segments)  # Target around 5 segments
        
        # Combine scores (weighted 70% support, 30% segmentation)
        comprehensive_score = 0.7 * weighted_support + 0.3 * segment_score
        
        return comprehensive_score
    
    def export_insulation_scores(self, output_dir):
        """
        Export insulation scores for all chromosomes
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            pandas.DataFrame: DataFrame with insulation scores
        """
        logger.info("Exporting insulation scores")
        
        insulation_data = []
        
        for species, chromosomes in self.genomes.items():
            for chrom_id, chromosome in chromosomes.items():
                if chromosome.insulation_scores is None or chromosome.insulation_windows is None:
                    continue
                
                for window, score in zip(chromosome.insulation_windows, chromosome.insulation_scores):
                    insulation_data.append({
                        'species': species,
                        'chromosome': chrom_id,
                        'window_position': window,
                        'insulation_score': score
                    })
        
        insulation_df = pd.DataFrame(insulation_data)
        
        if not insulation_df.empty:
            insulation_file = os.path.join(output_dir, 'insulation_scores.csv')
            insulation_df.to_csv(insulation_file, index=False)
            logger.info("Saved insulation scores to %s", insulation_file)
        
        return insulation_df
    
    def export_interaction_matrices(self, output_dir):
        """
        Export interaction matrices for visualization
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            dict: Dictionary with interaction matrices by chromosome
        """
        logger.info("Exporting interaction matrices")
        
        matrix_data = {}
        
        for species, chromosomes in self.genomes.items():
            for chrom_id, chromosome in chromosomes.items():
                if chromosome.interaction_matrix is None:
                    continue
                
                key = f"{species}_{chrom_id}"
                matrix_data[key] = chromosome.interaction_matrix
                
                # Save as CSV
                matrix_file = os.path.join(output_dir, f'interaction_matrix_{key}.csv')
                np.savetxt(matrix_file, chromosome.interaction_matrix, delimiter=',')
        
        logger.info("Saved interaction matrices to %s", output_dir)
        return matrix_data
    
    def evaluate_multiple_thresholds(self, output_dir):
        """
        Evaluate multiple insulation thresholds and find the optimal one
        
        Args:
            output_dir (str): Output directory
            
        Returns:
            tuple: (best_threshold, threshold_results)
        """
        logger.info("Evaluating multiple insulation thresholds")
        
        thresholds = np.arange(-0.9, 1.0 , 0.05)  # From -0.9 to 1 with step 0.05
        threshold_results = []
        
        for threshold in thresholds:
            # Split chromosomes with current threshold
            split_results = self.split_chromosomes(insulation_threshold=threshold)
            
            # Evaluate splits
            evaluation_results = self.evaluate_splits(threshold)
            
            # Calculate comprehensive score
            comp_score = self.calculate_comprehensive_score(evaluation_results)
            
            # Aggregate results
            if evaluation_results:
                avg_support = np.mean([r['support_ratio'] for r in evaluation_results])
                total_splits = len(split_results)
            else:
                avg_support = 0.0
                total_splits = 0
            
            threshold_results.append({
                'insulation_threshold': threshold,
                'comprehensive_score': comp_score,
                'average_support_ratio': avg_support,
                'total_splits': total_splits
            })
            
            logger.debug("Threshold %.2f: Comprehensive score = %.4f, Support = %.4f", 
                        threshold, comp_score, avg_support)
        
        # Find the best threshold
        threshold_df = pd.DataFrame(threshold_results)
        if not threshold_df.empty:
            # find the best threshold
            best_comprehensive_score = 0
            for _,tmp_row in threshold_df.iterrows():
                if tmp_row['comprehensive_score'] >= best_comprehensive_score:
                    best_comprehensive_score = tmp_row['comprehensive_score']
                    best_row = tmp_row
                    
            best_threshold = best_row['insulation_threshold']
            best_score = best_row['comprehensive_score']
            
            logger.info("Best insulation threshold: %.2f with score: %.4f", 
                       best_threshold, best_score)
            
            # Save threshold evaluation results
            threshold_file = os.path.join(output_dir, 'threshold_evaluation.csv')
            threshold_df.to_csv(threshold_file, index=False)
            logger.info("Saved threshold evaluation to %s", threshold_file)
            
            return best_threshold, threshold_df
        else:
            logger.warning("No threshold evaluation results")
            return -0.65, pd.DataFrame()
    
    def run_pipeline(self, anchors_dir, bed_dir, output_dir, 
                    insulation_threshold=None, min_chrom_length=500, 
                    export_interaction_matrices=True):
        """
        Run the complete genome splitting pipeline
        
        Args:
            anchors_dir (str): Directory containing anchors files
            bed_dir (str): Directory containing BED files
            output_dir (str): Output directory for results
            insulation_threshold (float): Specific insulation threshold to use, or None to auto-select
            min_chrom_length (int): Minimum chromosome length (genes) to process
            export_interaction_matrices (bool): Whether to export interaction matrices to CSV files
            
        Returns:
            tuple: (split_results, evaluation_results, best_threshold)
        """
        logger.info("Starting genome splitting pipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load data
        self.load_anchors(anchors_dir)
        self.load_genomes(bed_dir)
        
        # 2. Build collinear blocks
        self.build_blocks(anchors_dir)
        
        # 3. Evaluate multiple thresholds if not specified
        if insulation_threshold is None:
            best_threshold, threshold_results = self.evaluate_multiple_thresholds(output_dir)
            insulation_threshold = best_threshold
        else:
            best_threshold = insulation_threshold
            threshold_results = None
        
        # 4. Split chromosomes with the best threshold
        split_results = self.split_chromosomes(insulation_threshold, min_chrom_length)
        
        # 5. Evaluate split quality
        evaluation_results = self.evaluate_splits(insulation_threshold)
        
        # 6. Export additional data
        self.export_insulation_scores(output_dir)
        
        # 7. Export interaction matrices only if requested
        if export_interaction_matrices:
            self.export_interaction_matrices(output_dir)
        else:
            logger.info("Skipping interaction matrix export as requested")
        
        # 8. Save results
        self.save_results(split_results, evaluation_results, threshold_results, output_dir)
        
        logger.info("Pipeline completed successfully")
        return split_results, evaluation_results, best_threshold
    
    def save_results(self, split_results, evaluation_results, threshold_results, output_dir):
        """
        Save splitting results and evaluation results
        
        Args:
            split_results (list): List of split results
            evaluation_results (list): List of evaluation results
            threshold_results (pandas.DataFrame): DataFrame with threshold evaluation results
            output_dir (str): Output directory
        """
        # Save split results
        if split_results:
            split_df = pd.DataFrame(split_results)
            split_file = os.path.join(output_dir, 'chromosome_splits.csv')
            split_df.to_csv(split_file, index=False)
            logger.info("Saved split results to %s", split_file)
        
        # Save evaluation results
        if evaluation_results:
            eval_df = pd.DataFrame(evaluation_results)
            eval_file = os.path.join(output_dir, 'split_evaluation.csv')
            eval_df.to_csv(eval_file, index=False)
            logger.info("Saved evaluation results to %s", eval_file)
        
        # Save threshold evaluation results
        if threshold_results is not None and not threshold_results.empty:
            threshold_file = os.path.join(output_dir, 'threshold_evaluation.csv')
            threshold_results.to_csv(threshold_file, index=False)
            logger.info("Saved threshold evaluation to %s", threshold_file)
        
        # Save detailed block information
        if self.all_blocks:
            blocks_df = pd.DataFrame(self.all_blocks)
            blocks_file = os.path.join(output_dir, 'collinear_blocks.csv')
            blocks_df.to_csv(blocks_file, index=False)
            logger.info("Saved block information to %s", blocks_file)


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Split genomes into blocks based on collinearity analysis'
    )
    parser.add_argument('--anchors_dir', required=True,
                       help='Directory containing anchors files')
    parser.add_argument('--bed_dir', required=True,
                       help='Directory containing BED files')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--insulation_threshold', type=float, default=None,
                       help='Specific insulation threshold to use (default: auto-select)')
    parser.add_argument('--min_chrom_length', type=int, default=500,
                       help='Minimum chromosome length (genes) to process (default: 500)')
    parser.add_argument('--keep_interaction_matrices', action='store_true',
                       help='Skip exporting interaction matrices to CSV files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main function to run the genome splitting pipeline"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run splitting pipeline
    splitter = GenomeSplitter()
    
    try:
        split_results, eval_results, best_threshold = splitter.run_pipeline(
            anchors_dir=args.anchors_dir,
            bed_dir=args.bed_dir,
            output_dir=args.output_dir,
            insulation_threshold=args.insulation_threshold,
            min_chrom_length=args.min_chrom_length,
            export_interaction_matrices=args.keep_interaction_matrices
        )
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total splits: {len(split_results)}")
        if eval_results:
            avg_support = np.mean([r['support_ratio'] for r in eval_results])
            print(f"Average support ratio: {avg_support:.3f}")
        print(f"Best insulation threshold: {best_threshold:.2f}")
        print(f"Interaction matrices exported: {args.keep_interaction_matrices}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()