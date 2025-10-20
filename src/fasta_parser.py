#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fasta_parser.py

Multifunctional sequence and annotation processing tool.

Main functionalities:
1. Extract gene sequences based on a provided BED file.
2. Generate representative (longest) sequences for each Orthogroup (OG).
3. Filter genes based on OG frequency and annotation to produce limited genome and annotation files.
"""

import collections
import sys
import argparse
import glob
import os
import re
from typing import List, Dict

# ============================================================
# Core class and basic utility functions
# ============================================================

class GENE:
    """
    Represents a gene entry with positional and sequence information.

    Attributes:
        ID (str): Gene identifier.
        sp (str): Species name.
        chrom (str): Chromosome identifier.
        start (int): Start position.
        end (int): End position.
        order (str): Gene strand ('+' or '-').
        pos (int): Positional order on the chromosome.
        seq (str): Nucleotide or peptide sequence.
    """
    def __init__(self, ID:str='', sp:str='', chrom:str='', start:int=0, end:int=0, order:str='+', pos:int=0):
        self.ID, self.sp, self.chrom = ID, sp, chrom
        self.start, self.end, self.order, self.pos = int(start), int(end), order, int(pos)
        self.seq = ''

    def __repr__(self):
        """Return a concise human-readable description of the gene."""
        return f"{self.ID} is {self.pos} gene on chromosome {self.chrom} of {self.sp}, {self.start}-{self.end}"

    def __str__(self):
        """Return the gene ID when printed."""
        return self.ID


def read_bed_file(bed_file:str, query_genes:set=None) -> dict:
    """
    Read a BED file and return a dictionary mapping chromosomes to lists of GENE objects.

    Args:
        bed_file (str): Path to BED file.
        query_genes (set, optional): Set of gene IDs to filter; if None, all genes are included.

    Returns:
        dict: {chromosome: [GENE, GENE, ...]}
    """
    query_bed = collections.defaultdict(list)
    with open(bed_file) as f:
        for line in f:
            if not line.strip():
                continue
            tmp_chrom, _, _, tmp_ID, _, tmp_order = line.strip().split('\t')
            # Skip genes not in query set (if provided)
            if query_genes and tmp_ID not in query_genes:
                continue
            query_bed[tmp_chrom].append(
                GENE(ID=tmp_ID, chrom=tmp_chrom,
                     start=len(query_bed[tmp_chrom]),
                     end=len(query_bed[tmp_chrom])+1,
                     order=tmp_order)
            )
    return query_bed


def read_fasta(file_path:str) -> Dict[str, str]:
    """
    Read a FASTA file and return a dictionary of {sequence_id: sequence}.

    Args:
        file_path (str): Path to FASTA file.

    Returns:
        dict: Mapping from sequence ID to sequence string.
    """
    sequences = {}
    with open(file_path, 'r') as f:
        seq_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
                sequences[seq_id] = ''
            else:
                sequences[seq_id] += line
    return sequences


def read_orthogroups_file(orthogrous_file:str) -> Dict[str, Dict[str, List[str]]]:
    """
    Read an Orthogroups file and return a nested dictionary mapping:
        species → OG_id → list of genes.

    Args:
        orthogrous_file (str): Path to Orthogroups.tsv file.

    Returns:
        dict: {species: {OG_id: [gene1, gene2, ...]}}
    """
    orthogroups = collections.defaultdict(dict)
    with open(orthogrous_file) as f:
        taxon_line = f.readline().strip()
        taxons = taxon_line.split()[1:]
        while line := f.readline():
            parts = line.strip().split('\t')
            OG_id = parts[0]
            for tmp_taxon, gene_list in zip(taxons, parts[1:]):
                gene_list = gene_list.replace('transcript_', 'transcript:')
                orthogroups[tmp_taxon][OG_id] = gene_list.split(', ')
    return orthogroups


def get_longest_seq(sequences: Dict[str, str]) -> str:
    """Return the longest sequence from a dictionary of sequences."""
    return max(sequences.values(), key=len) if sequences else ''


def write_fasta(sequences: Dict[str, str], output_path: str, str_n:int = 60) -> None:
    """
    Write sequences to a FASTA file.

    Args:
        sequences (dict): {sequence_id: sequence}
        output_path (str): Output FASTA file path.
        str_n (int): Number of characters per line (default=60).
    """
    with open(output_path, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f'>{seq_id}\n')
            for i in range(0, len(seq), str_n):
                f.write(seq[i:i+str_n] + '\n')


def write_bed(bed_dict: Dict[str, List[GENE]], output_path: str) -> None:
    """
    Write a BED-like file from a dictionary of GENE objects.

    Args:
        bed_dict (dict): {chromosome: [GENE, GENE, ...]}
        output_path (str): Output BED file path.
    """
    with open(output_path, 'w') as f:
        for chrom, genes in bed_dict.items():
            for gene in genes:
                f.write(f"{gene.chrom}\t{gene.start}\t{gene.end}\t{gene.ID}\t0\t{gene.order}\n")


# ============================================================
# Functional modules
# ============================================================

def generate_OG_represent_seq(orthgroups_seq_dir:str) -> Dict[str, str]:
    """
    Select the longest sequence as a representative for each Orthogroup (OG).

    Args:
        orthgroups_seq_dir (str): Directory containing multiple OG FASTA files.

    Returns:
        dict: {OG_id: longest_sequence}
    """
    OG_files = glob.glob(os.path.join(orthgroups_seq_dir, '*.fasta')) + \
               glob.glob(os.path.join(orthgroups_seq_dir, '*.fa'))
    OG_represent_seqs = {}
    for OG_file in OG_files:
        OG_id = os.path.basename(OG_file).split('.')[0]
        sequences = read_fasta(OG_file)
        longest_seq = get_longest_seq(sequences)
        if longest_seq:
            OG_represent_seqs[OG_id] = longest_seq
    return OG_represent_seqs


def generate_queried_seq(bed_file:str, genome_fasta:str, prefix_pattern:str = '') -> Dict[str, str]:
    """
    Generate gene sequences based on BED and genome FASTA, supporting regex-based ID matching.

    Args:
        bed_file (str|dict): Path to BED file or a preloaded BED dict.
        genome_fasta (str|dict): Path to FASTA file or a preloaded FASTA dict.
        prefix_pattern (str): Regular expression pattern for gene ID prefix.

    Returns:
        dict: {gene_ID: sequence}
    """
    if isinstance(bed_file, str):
        query_bed = read_bed_file(bed_file)
    else:
        query_bed = bed_file

    if isinstance(genome_fasta, str):
        gene_seqs = read_fasta(genome_fasta)
    else:
        gene_seqs = genome_fasta

    # Validate regex pattern
    try:
        re.compile(prefix_pattern)
        gene_id_pattern = f'{prefix_pattern}(.*)' if prefix_pattern else '(.*)'
    except re.error:
        gene_id_pattern = re.escape(prefix_pattern) + '(.*)'

    queried_seqs = {}
    for chrom, genes in query_bed.items():
        for gene in genes:
            match = re.match(gene_id_pattern, gene.ID)
            gene_id = match.group(1) if match else gene.ID
            if gene_id in gene_seqs:
                queried_seqs[gene.ID] = gene_seqs[gene_id]
    return queried_seqs


def count_gene_frequency(bed_file:str, prefix_pattern:str='(.*)') -> Dict[str, int]:
    """
    Count the frequency of each gene ID in a BED file.

    Args:
        bed_file (str): Path to BED file.
        prefix_pattern (str): Regex pattern to strip prefixes from gene IDs.

    Returns:
        dict: {gene_id: frequency}
    """
    try:
        re.compile(prefix_pattern)
        gene_id_pattern = f'{prefix_pattern}(.*)' if prefix_pattern else '(.*)'
    except re.error:
        gene_id_pattern = re.escape(prefix_pattern) + '(.*)'

    gene_frequency = collections.Counter()
    with open(bed_file) as f:
        for line in f:
            if not line.strip():
                continue
            tmp_ID = line.strip().split('\t')[3]
            match = re.match(gene_id_pattern, tmp_ID)
            gene_id = match.group(1) if match else tmp_ID
            gene_frequency[gene_id] += 1
    return gene_frequency


def get_frequency_limited_genes(gene_frequency:Dict[str,int], min_freq:int = 1, max_freq:int = 2) -> set:
    """
    Retrieve genes whose frequencies fall within a specified range.

    Args:
        gene_frequency (dict): {gene_id: frequency}
        min_freq (int): Minimum frequency threshold.
        max_freq (int): Maximum frequency threshold.

    Returns:
        set: Set of gene IDs meeting the frequency criteria.
    """
    return {gene_id for gene_id, freq in gene_frequency.items() if min_freq <= freq <= max_freq}


def extract_bed(bed_file:str, query_genes:set, prefix:str = '') -> Dict[str, List[GENE]]:
    """
    Extract gene entries from a BED file based on a set of target gene IDs.

    Args:
        bed_file (str): Path to BED file.
        query_genes (set): Target gene IDs.
        prefix (str): Regex pattern for prefix matching.

    Returns:
        dict: {chromosome: [GENE, GENE, ...]}
    """
    try:
        re.compile(prefix)
        gene_id_pattern = f'{prefix}(.*)' if prefix else '(.*)'
    except re.error:
        gene_id_pattern = re.escape(prefix) + '(.*)'

    extracted_bed = collections.defaultdict(list)
    with open(bed_file) as f:
        while tmp_line := f.readline():
            tmp_chrom, _, _, tmp_ID, _, tmp_order = tmp_line.strip().split('\t')
            rm_prefix_match = re.match(gene_id_pattern, tmp_ID)
            rm_prefix_gene_id = rm_prefix_match.group(1) if rm_prefix_match else tmp_ID
            if rm_prefix_gene_id in query_genes:
                extracted_bed[tmp_chrom].append(
                    GENE(ID=tmp_ID,
                         chrom=tmp_chrom,
                         start=len(extracted_bed[tmp_chrom]),
                         end=len(extracted_bed[tmp_chrom]) + 1,
                         order=tmp_order)
                )
    return extracted_bed


def extract_taxon_genes(taxon_OG_dic:Dict, target_taxon:str, query_OGs:set) -> set:
    """
    Extract genes of a specific taxon based on a set of Orthogroups.

    Args:
        taxon_OG_dic (dict): {species: {OG_id: [gene_list]}}
        target_taxon (str): Target species name.
        query_OGs (set): Set of OG IDs to extract.

    Returns:
        set: Set of gene IDs for the specified taxon.
    """
    taxon_genes = set()
    if target_taxon in taxon_OG_dic:
        for OG_id in query_OGs:
            if OG_id in taxon_OG_dic[target_taxon]:
                taxon_genes.update(set(taxon_OG_dic[target_taxon][OG_id]))
    return taxon_genes


def check_genome_files_exist(genome_dir:str, taxa:List[str] , suffix:str = 'pep') -> List[str]:
    """
    Verify the presence of genome (.pep and .bed) files for each taxon.

    Args:
        genome_dir (str): Directory containing genome files.
        taxa (List[str]): List of taxon names.

    Returns:
        List[str]: Taxa with both .pep and .bed files available.
    """
    available_taxa = []
    for taxon in taxa:
        pep_path = os.path.join(genome_dir, f"{taxon}.{suffix}")
        bed_path = os.path.join(genome_dir, f"{taxon}.{suffix}")
        if os.path.isfile(pep_path) and os.path.isfile(bed_path):
            available_taxa.append(taxon)
    return available_taxa

def get_seq_suffix(genome_dir:str, taxa:List[str]) -> str:
    """
    Determine the suffix of sequence files in the genome directory.

    Args:
        genome_dir (str): Directory containing genome files.
        taxa (List[str]): List of taxon names.

    Returns:
        str: Suffix of the sequence files (e.g., 'pep', 'faa').
    """
    for taxon in taxa:
        for suffix in ['pep', 'faa', 'fasta', 'fa', 'cds']:
            if os.path.isfile(os.path.join(genome_dir, f"{taxon}.{suffix}")):
                return suffix
    return 'pep'  # Default suffix

# ============================================================
# extract_limited module
# ============================================================

def extract_limited(orthogroups_file:str, ancestor_bed:str, ancestor_genome:str,
                    genome_dir_A:str, output_dir_B:str,
                    min_freq:int=1, max_freq:int=2, prefix_pattern:str='(.*)') -> None:
    """
    Extract restricted genome and annotation files based on OG frequency limits.

    Args:
        orthogroups_file (str): Path to Orthogroups.tsv file.
        ancestor_bed (str): Path to ancestor BED file.
        ancestor_genome (str): Path to ancestor genome FASTA.
        genome_dir_A (str): Directory containing genome .pep and .bed files for all taxa.
        output_dir_B (str): Output directory for limited files.
        min_freq (int): Minimum frequency threshold.
        max_freq (int): Maximum frequency threshold.
        prefix_pattern (str): Regex pattern for prefix stripping.
    """
    os.makedirs(output_dir_B, exist_ok=True)

    # 1. Count OG frequencies in ancestor BED
    freq = count_gene_frequency(ancestor_bed, prefix_pattern)
    limited_genes = get_frequency_limited_genes(freq, min_freq, max_freq)
    print(f"[INFO] Number of frequency-limited genes: {len(limited_genes)}")

    # 2. Parse Orthogroups.tsv
    taxon_OG_dic = read_orthogroups_file(orthogroups_file)
    all_taxa = list(taxon_OG_dic.keys())

    # 3. Check genome availability
    suffix = get_seq_suffix( genome_dir_A , all_taxa)
    valid_taxas = check_genome_files_exist(genome_dir_A, all_taxa , suffix= suffix)
    if len(valid_taxas) == len(all_taxa):
        print(f"[INFO] All species genome files are available.")
    else:
        print(f"[WARNING] Missing genome files for: {set(all_taxa) - set(valid_taxas)}")

    # 4. Generate ancestor-limited files
    ancestor_seq = read_fasta(ancestor_genome)
    ancestor_limited_bed = extract_bed(ancestor_bed, limited_genes, prefix=prefix_pattern)
    ancestor_limited_pep = generate_queried_seq(ancestor_limited_bed, ancestor_seq, prefix_pattern='')

    write_fasta(ancestor_limited_pep, os.path.join(output_dir_B, f'ancestor_limited.{suffix}'))
    write_bed(ancestor_limited_bed, os.path.join(output_dir_B, 'ancestor_limited.bed'))

    # 5. Generate limited files for each valid taxon
    for taxon in valid_taxas:
        pep_path = os.path.join(genome_dir_A, f"{taxon}.{suffix}")
        bed_path = os.path.join(genome_dir_A, f"{taxon}.bed")

        taxon_limited_genes = extract_taxon_genes(taxon_OG_dic, taxon, limited_genes)
        taxon_limited_bed = extract_bed(bed_file=bed_path, query_genes=taxon_limited_genes)

        seqs = read_fasta(pep_path)
        taxon_limited_seq = {gid: seq for gid, seq in seqs.items() if gid in taxon_limited_genes}

        write_fasta(taxon_limited_seq, os.path.join(output_dir_B, f"{taxon}_limited.{suffix}"))
        write_bed(taxon_limited_bed, os.path.join(output_dir_B, f"{taxon}_limited.bed"))

    print(f"[INFO] All frequency-limited files written to {output_dir_B}")


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multifunctional FASTA and BED processing toolkit.")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand 1: extract_seqs
    p1 = subparsers.add_parser("extract_seqs", help="Extract gene sequences based on BED.")
    p1.add_argument("--bed", required=True)
    p1.add_argument("--genome", required=True)
    p1.add_argument("--prefix", required=True, help="Supports regex patterns, e.g., '[0-9]+-' or '(.*?)_'")
    p1.add_argument("--output", required=True)

    # Subcommand 2: generate_OG_representatives
    p2 = subparsers.add_parser("generate_OG_representatives", help="Generate representative (longest) sequence per OG.")
    p2.add_argument("--dir", required=True, help="Directory containing multiple OG FASTA files.")
    p2.add_argument("--output", required=True)

    # Subcommand 3: extract_limited
    p3 = subparsers.add_parser("extract_limited", help="Extract genomes and annotations under frequency constraints.")
    p3.add_argument("--orthogroups", required=True)
    p3.add_argument("--ancestor_bed", required=True)
    p3.add_argument("--ancestor_genome", required=True)
    p3.add_argument("--genome_dir", required=True)
    p3.add_argument("--output_dir", required=True)
    p3.add_argument("--min_freq", type=int, default=1)
    p3.add_argument("--max_freq", type=int, default=2)
    p3.add_argument("--prefix", default="(.*)", help="Regex pattern for matching gene ID prefixes.")

    args = parser.parse_args()

    # Command dispatch
    if args.command == "extract_seqs":
        seqs = generate_queried_seq(args.bed, args.genome, args.prefix)
        write_fasta(seqs, args.output)
    elif args.command == "generate_OG_representatives":
        rep = generate_OG_represent_seq(args.dir)
        write_fasta(rep, args.output)
    elif args.command == "extract_limited":
        extract_limited(args.orthogroups, args.ancestor_bed, args.ancestor_genome,
                        args.genome_dir, args.output_dir,
                        args.min_freq, args.max_freq, args.prefix)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
