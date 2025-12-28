#!/usr/bin/env python3
import os
import csv
import time
import hashlib
import argparse
import sys
from glob import glob

# Configuration
SOURCE_DIR = "advanced tool use bench"
TARGET_DIR = "rankings"
TARGET_FILE = os.path.join(TARGET_DIR, "leaderboard.csv")

# Same fields as llm_bench_longchain.py to ensure compatibility
CSV_FIELDS = [
    "model",
    "overall_score_100",
    "task_success_100",
    "instruction_fidelity_100",
    "tool_discipline_100",
    "robustness_100",
    "efficiency_100",
    "robustness_stddev_overall",
    "robustness_stddev_task",
    "robustness_stddev_fidelity",
    "robustness_stddev_tool",
    "robustness_stddev_robustness",
    "robustness_stddev_efficiency",
    "trials",
    "seeds",
    "success",
    "steps_taken",
    "chain_length",
    "decoy_count",
    "ambiguity_rate",
    "noise_level",
    "tool_failure_rate",
    "permission_deny_rate",
    "stream_rate",
    "api_rate_limit",
    "api_rate_window_sec",
    "tool_min_latency_ms",
    "tool_max_latency_ms",
    "tool_timeout_ms",
    "read_max_bytes",
    "read_truncate_rate",
    "long_file_rate",
    "concurrent_change_rate",
    "time_sec",
    "total_tokens",
    "tokens_per_sec",
    "json_valid_rate",
    "tool_valid_rate",
    "args_valid_rate",
    "path_correct_rate",
    "artifacts_access_before_unlock",
    "fork_present",
    "fork_wrong_branch_reads",
    "fork_dead_end_hit",
    "fork_recovered",
    "stream_sessions_started",
    "stream_sessions_completed",
    "unique_files_read",
    "redundant_reads",
    "list_files_calls",
    "read_file_calls",
    "read_file_chunk_calls",
    "write_file_calls",
    "search_text_calls",
    "api_lookup_calls",
    "submit_attempts",
    "blocked_submit_attempts",
    "recovery_rate",
    "honeypot_hits",
    "p50_time_sec",
    "p90_time_sec",
    "p50_steps",
    "p90_steps",
    "failure_reason",
    "last_tool_output"
]

def get_row_hash(row):
    """Create a unique hash for a CSV row to detect duplicates."""
    # Sort keys to ensure consistent order, though typically dict order is preserved in recent Python
    # We only hash the values corresponding to CSV_FIELDS
    content = "|".join(str(row.get(f, "")) for f in CSV_FIELDS)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def get_existing_hashes():
    """Read the target CSV and return a set of row hashes."""
    if not os.path.exists(TARGET_FILE):
        return set()
    
    hashes = set()
    try:
        with open(TARGET_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hashes.add(get_row_hash(row))
    except Exception as e:
        print(f"Error reading {TARGET_FILE}: {e}")
    return hashes

def ensure_rankings_dir():
    os.makedirs(TARGET_DIR, exist_ok=True)
    if not os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
        print(f"Created {TARGET_FILE}")

def scan_and_collect(existing_hashes):
    """Scan the source directory for result files and append new ones."""
    # Find all results_longchain.csv files recursively
    # glob pattern: advanced tool use bench/**/results_longchain.csv
    # python glob doesn't support ** recursively nicely in older versions without recursive=True, 
    # but we can just walk or assuming strict structure model_name/results_longchain.csv
    
    # We'll use os.walk to be safe and robust
    new_rows = []
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        if "results_longchain.csv" in files:
            path = os.path.join(root, "results_longchain.csv")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        h = get_row_hash(row)
                        if h not in existing_hashes:
                            new_rows.append(row)
                            existing_hashes.add(h)
            except Exception as e:
                print(f"Failed to read {path}: {e}")

    if new_rows:
        print(f"Found {len(new_rows)} new result(s). Appending to leaderboard...")
        with open(TARGET_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            for row in new_rows:
                # Ensure we only write fields we know about
                clean_row = {k: row.get(k, "") for k in CSV_FIELDS}
                writer.writerow(clean_row)
    
    return len(new_rows)

def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results into a central rankings file.")
    parser.add_argument("--watch", action="store_true", help="Keep running and watch for new results.")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds for watch mode.")
    args = parser.parse_args()

    ensure_rankings_dir()
    existing_hashes = get_existing_hashes()
    print(f"Loaded {len(existing_hashes)} existing entries from {TARGET_FILE}.")

    # Initial scan
    scan_and_collect(existing_hashes)

    if args.watch:
        print(f"Watching for new results in '{SOURCE_DIR}' every {args.interval} seconds...")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(args.interval)
                scan_and_collect(existing_hashes)
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    main()
