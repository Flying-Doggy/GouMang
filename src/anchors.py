#!/usr/bin/env python3
"""
JCVI All-vs-All Pairwise Runner (Class-based version)

Author: Flying-Doggy
Date: 2025-10-12

Description:
    This script performs all-vs-all pairwise synteny (ortholog) comparison 
    using the JCVI toolkit. It scans an input directory for matching .pep/.bed 
    or .cds/.bed files (based on filename prefixes), validates their correspondence, 
    and executes JCVI commands for each species pair.

Features:
    - Validates one-to-one matching between .pep/.bed (or .cds/.bed) files
    - Supports user-defined JCVI arguments
    - Groups comparisons by target species to avoid DB overwrite issues
    - Parallel execution across groups (ThreadPoolExecutor)
    - Collects all .anchors results into one folder
    - Cleans intermediate JCVI database files
    - Logs progress and detailed error messages per failed comparison

Example Usage:
    python jcvi_pairwise_runner.py /path/to/input_dir \
        --anchors-dir anchors_out \
        --jcvi-args "--no_dotplot --no_strip_names --cscore 0.9" \
        --cpus 20 --parallel 3
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import shlex
from itertools import combinations
from pathlib import Path
from collections import defaultdict
import concurrent.futures


class JCVIPairwiseRunner:
    """Class to perform JCVI all-vs-all pairwise synteny comparisons."""
    def __init__(self, input_dir: Path, anchors_dir: Path,  jcvi_args: str,
                 cpus: int = 8, dbtype: str = "prot", parallel: int = 1,
                 keep_intermediates: bool = False, dry_run: bool = False,
                 log_file: Path | None = None , reference_id:str = None):
        self.input_dir = input_dir.resolve()
        self.anchors_dir = anchors_dir
        self.jcvi_args = shlex.split(jcvi_args)
        self.cpus = cpus
        self.dbtype = dbtype
        self.parallel = parallel
        self.keep_intermediates = keep_intermediates
        self.dry_run = dry_run
        self.log_file = log_file or (self.input_dir / "jcvi_pairwise_runner.log")
        self.reference_id = reference_id

        self.LOGGER = logging.getLogger("jcvi_pairwise_runner")
        self._setup_logging()

    def _setup_logging(self):
        """Configure both console and file loggers."""
        fmt = "%(asctime)s [%(levelname)s] %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(self.log_file))
            ]
        )

    def find_files(self, ext: str):
        """Find all files with the specified extension in the input directory."""
        return sorted(self.input_dir.glob(f"*{ext}"))

    def prefixes_from_files(self, file_list, ext):
        """Extract file prefixes (without extension) as keys."""
        prefixes = {}
        for p in file_list:
            prefix = p.name[:-len(ext)] if p.name.endswith(ext) else p.stem
            prefixes[prefix] = p
        return prefixes

    def validate_pairs(self, seq_prefixes: dict, bed_prefixes: dict):
        """Ensure .pep/.bed (or .cds/.bed) files are one-to-one matched."""
        seq_set = set(seq_prefixes.keys())
        bed_set = set(bed_prefixes.keys())

        missing_in_bed = seq_set - bed_set
        missing_in_seq = bed_set - seq_set

        if missing_in_bed or missing_in_seq:
            self.LOGGER.error("Mismatch detected between sequence and bed prefixes.")
            if missing_in_bed:
                self.LOGGER.error("Missing bed for: %s", ", ".join(sorted(missing_in_bed)))
            if missing_in_seq:
                self.LOGGER.error("Missing seq for: %s", ", ".join(sorted(missing_in_seq)))
            raise SystemExit(1)

        prefixes = sorted(list(seq_set))
        self.LOGGER.info("Validated %d taxa with matching seq/bed pairs.", len(prefixes))
        return prefixes

    def build_jcvi_command(self, taxa_a: str, taxa_b: str):
        """Construct the JCVI command for a given taxon pair."""
        base = [sys.executable, "-m", "jcvi.compara.catalog", "ortholog", taxa_a, taxa_b]
        args = list(self.jcvi_args)
        if not any(a.startswith("--cpus") for a in args):
            args += ["--cpus", str(self.cpus)]
        if not any(a.startswith("--dbtype") for a in args):
            args += ["--dbtype", self.dbtype]
        return base + args

    def run_pair(self, taxa_a: str, taxa_b: str):
        """Execute JCVI command for a single taxon pair."""
        cmd = self.build_jcvi_command(taxa_a, taxa_b)
        self.LOGGER.info("Running JCVI for: %s vs %s", taxa_a, taxa_b)
        self.LOGGER.debug("Command: %s", " ".join(shlex.quote(x) for x in cmd))

        if self.dry_run:
            self.LOGGER.info("Dry run enabled; skipping execution.")
            return 0

        proc = subprocess.run(cmd, cwd=str(self.input_dir), check=False,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            self.LOGGER.warning("JCVI returned non-zero code (%d) for %s vs %s",
                                proc.returncode, taxa_a, taxa_b)
            self.LOGGER.warning("Command: %s", " ".join(shlex.quote(x) for x in cmd))

            # Print first 10 lines of stderr
            if proc.stderr.strip():
                stderr_lines = proc.stderr.strip().splitlines()
                short_err = "\n".join(stderr_lines[:10])
                self.LOGGER.warning("Error message (first 10 lines):\n%s", short_err)
            else:
                self.LOGGER.warning("No stderr output captured from JCVI.")

            # Save detailed error log
            errors_dir = self.input_dir / "logs"
            errors_dir.mkdir(exist_ok=True)
            err_file = errors_dir / f"error_{taxa_a}_vs_{taxa_b}.log"
            with err_file.open("w", encoding="utf-8") as ef:
                ef.write("=== JCVI command ===\n")
                ef.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
                ef.write("=== STDOUT ===\n")
                ef.write(proc.stdout or "(empty)\n")
                ef.write("\n=== STDERR ===\n")
                ef.write(proc.stderr or "(empty)\n")
            self.LOGGER.warning("Saved detailed error log to: %s", err_file)
        return proc.returncode

    def collect_anchors(self):
        """Move all .anchors files to the designated anchors directory."""
        # create anchors directory if it doesn't exist
        self.anchors_dir.mkdir(parents=True, exist_ok=True)
        
        moved = []
        for anchors in self.input_dir.glob("*.anchors"):
            dest = self.anchors_dir / anchors.name
            shutil.move(str(anchors), str(dest))
            moved.append(dest)
        if moved:
            self.LOGGER.info("Moved %d anchors to %s", len(moved), self.anchors_dir)
        return moved

    def group_pairs_by_target(self, pairs):
        """Group pair comparisons by the second taxon (target species)."""
        grouped = defaultdict(list)
        for a, b in pairs:
            grouped[b].append((a, b))
        return grouped

    def run_group(self, group_pairs):
        """Run all pairs within a single target-species group."""
        for a, b in group_pairs:
            self.run_pair(a, b)
        return len(group_pairs)

    def clean_intermediates(self):
        """Remove intermediate JCVI database files after processing."""
        suffixes = ['.bck', '.ssp', '.suf', '.tis', '.des', '.prj', '.sds', '.last', '.filtered']
        removed = []
        for suffix in suffixes:
            for f in self.input_dir.glob(f"*{suffix}"):
                try:
                    f.unlink()
                    removed.append(f)
                except Exception as e:
                    self.LOGGER.warning("Failed to remove %s: %s", f, e)
        self.LOGGER.info("Cleaned %d intermediate JCVI files.", len(removed))
        return removed

    def run(self):
        """Main workflow controller for the pairwise JCVI runner."""
        seq_ext = '.cds' if self.dbtype == 'cds' else '.pep'
        seq_files = self.find_files(seq_ext)
        bed_files = self.find_files('.bed')

        if not seq_files or not bed_files:
            self.LOGGER.error("No sequence or bed files found in input directory: %s", self.input_dir)
            sys.exit(1)

        seq_prefixes = self.prefixes_from_files(seq_files, seq_ext)
        bed_prefixes = self.prefixes_from_files(bed_files, '.bed')
        prefixes = self.validate_pairs(seq_prefixes, bed_prefixes)

        if self.reference_id is None:
            pairs = list(combinations(prefixes, 2))
        else:
            pairs = [(b, self.reference_id) for b in prefixes if b != self.reference_id]

        self.LOGGER.info("Total pairwise comparisons: %d", len(pairs))

        grouped = self.group_pairs_by_target(pairs)
        self.LOGGER.info("Grouped by target taxon: %d groups", len(grouped))

        if self.parallel <= 1:
            for b, group in grouped.items():
                self.LOGGER.info("Processing group (target=%s, %d pairs)", b, len(group))
                self.run_group(group)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as exe:
                futures = {
                    exe.submit(self.run_group, group): b
                    for b, group in grouped.items()
                }
                for fut in concurrent.futures.as_completed(futures):
                    b = futures[fut]
                    try:
                        completed = fut.result()
                        self.LOGGER.info("Group %s completed (%d pairs)", b, completed)
                    except Exception as e:
                        self.LOGGER.exception("Group %s failed: %s", b, e)

        self.collect_anchors()
        if not self.keep_intermediates and not self.dry_run:
            self.clean_intermediates()

        self.LOGGER.info("All JCVI comparisons finished successfully.")
        self.LOGGER.info("Anchors collected in: %s", os.path.abspath(self.anchors_dir))
        self.LOGGER.info("Log file: %s", self.log_file)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run JCVI all-vs-all ortholog comparisons across multiple taxa.")
    parser.add_argument("input_dir", help="Directory containing .pep/.bed or .cds/.bed files")
    parser.add_argument("--anchors-dir", default="anchors", help="Output directory for collected .anchors files")
    parser.add_argument("--reference", type=str, default=None , help="Reference species prefix (not used in pairwise mode)")
    parser.add_argument("--jcvi-args", default="--no_dotplot --no_strip_names", help="Additional JCVI parameters")
    parser.add_argument("--cpus", type=int, default=8, help="Number of CPUs for JCVI execution")
    parser.add_argument("--dbtype", default="prot", choices=["prot", "cds"], help="Database type (prot/cds)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of target species groups to process in parallel")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep intermediate JCVI DB files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running JCVI")
    parser.add_argument("--log-file", default=None, help="Path to output log file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    runner = JCVIPairwiseRunner(
        input_dir=Path(args.input_dir),
        anchors_dir=Path(args.anchors_dir),
        jcvi_args=args.jcvi_args,
        cpus=args.cpus,
        dbtype=args.dbtype,
        parallel=args.parallel,
        keep_intermediates=args.keep_intermediates,
        dry_run=args.dry_run,
        log_file=Path(args.log_file) if args.log_file else None,
        reference_id=args.reference
    )
    runner.run()
