#!/usr/bin/env python3
import argparse, json, re, time, csv, sys, os, random, statistics, hashlib
from copy import deepcopy
from fnmatch import fnmatch
from typing import Any, Dict, List, Optional, Tuple
import requests

# --------------------------------------------------------------------------------
# CONFIGURATION

class AgentTurnResult:
    def __init__(self, status="N/A", drift=0, finished=False, tool_name=None, tool_args=None, tool_result=None, detail=None, expected_next=None, full_result=None):
        self.status = status
        self.drift = drift
        self.finished = finished
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result
        self.detail = detail
        self.expected_next = expected_next
        self.full_result = full_result
# --------------------------------------------------------------------------------

PROMPT_SYSTEM = """You are an autonomous agent capable of using tools to navigate a virtual file system.
Your goal is to find the "SECRET_CODE" and follow the requirements in task.md.

You have access to the following tools:
1. read_file(path): Returns JSON {"ok": true, "path": "...", "content": "...", "truncated": bool, "next_offset": int or null, "done": bool, "total_size": int}. If truncated, returns {"chunk": "..."} instead of full content.
2. read_file_chunk(path, offset, length): Returns JSON {"ok": true, "path": "...", "chunk": "...", "truncated": bool, "next_offset": int or null, "done": bool, "total_size": int}.
3. list_files(directory, cursor=None, limit=50): Returns JSON {"ok": true, "directory": "...", "entries": ["a", "b"], "next_cursor": int or null}.
4. stat(path): Returns JSON {"ok": true, "path": "...", "exists": true, "type": "file|dir", "size": 123, "locked": false}.
5. glob(directory, pattern): Returns JSON {"ok": true, "directory": "...", "pattern": "...", "matches": ["..."]}.
6. tree(directory, depth): Returns JSON {"ok": true, "directory": "...", "depth": N, "entries": ["dir/", "file.txt"]}.
7. read_json(path): Returns JSON {"ok": true, "path": "...", "data": {...}}.
8. write_json(path, obj): Returns JSON {"ok": true, "path": "...", "message": "..."}.
9. write_file(path, content): Returns JSON {"ok": true, "path": "...", "message": "...", "bytes": int (optional)}.
10. api_lookup(query): Returns JSON {"ok": true, "query": "...", "data": "..."} with rate limits.
11. search_text(query, directory, max_results): Returns JSON {"ok": true, "query": "...", "directory": "...", "max_results": N, "matches": [{"path": "...", "snippet": "..."}]}.
12. capabilities(): Returns JSON {"ok": true, "tools": {...}}.
13. remember(key, value): Returns JSON {"ok": true, "key": "..."}.
14. recall(key): Returns JSON {"ok": true, "key": "...", "value": "...", "found": true|false}.
15. checkpoint(name): Returns JSON {"ok": true, "name": "..."}.
16. rollback(name): Returns JSON {"ok": true, "name": "...", "rolled_back": true|false}.
17. submit_answer(answer): Returns JSON {"ok": true, "message": "..."} when correct.

FORMAT INSTRUCTIONS:
To use a tool, you MUST output a valid JSON object in the following format:
```json
{
    "tool": "tool_name",
    "args": { "arg_name": "value" }
}
```
You can add reasoning before the JSON block.
Do NOT output multiple tool calls in one turn.
Stop after outputting the JSON tool call. I will give you the tool output in the next turn.
Tool errors are returned as JSON objects with {"ok": false, "error": {...}}.
If read_file output is truncated, it returns {"chunk": "...", "next_offset": int} so you can continue with read_file_chunk.
"""

PROMPT_USER_START = "Begin the search. Start by reading start_instructions.txt, then follow the trace chain."

MAX_STEPS = 120  # Safety limit
MAX_CHUNK_READS_PER_FILE = 6

CSV_FIELDS = [
    "model",
    "benchmark_track",
    "time_sec",
    "overall_score_100",
    "task_success_100",
    "compliance_success_100",
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
    "total_tokens",
    "tokens_per_sec",
    "json_valid_rate",
    "tool_valid_rate",
    "args_valid_rate",
    "schema_valid_rate",
    "output_schema_valid_rate",
    "path_correct_rate",
    "artifacts_access_before_unlock",
    "wrong_trace_reads",
    "explore_over_budget_events",
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
    "tool_failed_events",
    "tool_error_recovery_success",
    "honeypot_hits",
    "p50_time_sec",
    "p90_time_sec",
    "p50_steps",
    "p90_steps",
    "failure_reason",
    "last_tool_output"
]

# --------------------------------------------------------------------------------
# VIRTUAL FILE SYSTEM
# --------------------------------------------------------------------------------

class VirtualFileSystem:
    def __init__(
        self,
        chain_length=10,
        decoy_count=0,
        ambiguity_rate=0.0,
        permission_deny_rate=0.0,
        long_file_rate=0.2,
        mode: str = "compliance",
        seed: Optional[int] = None,
    ):
        self.files = {}
        self.chain_length = chain_length
        self.decoy_count = decoy_count
        self.ambiguity_rate = ambiguity_rate
        self.permission_deny_rate = permission_deny_rate
        self.long_file_rate = long_file_rate
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.secret_code = ""
        self.secret_checksum = ""
        self.solution_path = "solution.txt"
        self.solution_written = False
        self.solution_content = ""
        self.artifacts_unlocked = False
        self.fork_present = False
        self.fork_index = None
        self.fork_dir = None
        self.fork_branch_path = None
        self.fork_dead_end_path = None
        self.fork_wrong_branch_reads = 0
        self.fork_dead_end_hit = False
        self.fork_recovered = False
        self.read_counts = {}
        self.read_chunk_counts = {}
        self.changed_files = set()
        self.honeypot_secrets = set()
        self.last_world_change = False
        self.generate_scenario()
        self.api_calls = []

    def _is_artifacts_path(self, path: str) -> bool:
        p = path.strip("./")
        return p == "artifacts" or p.startswith("artifacts/")

    def generate_scenario(self):
        # Create a path of traces with discoverable clues
        dims = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]

        def _random_dir():
            base = self.rng.choice(dims)
            if self.rng.random() < 0.4:
                year = self.rng.choice(["2024", "2025"])
                month = f"{self.rng.randint(1,12):02d}"
                return f"{base}/logs/{year}/{month}"
            return base

        def _trace_id_from_path(p: str) -> str:
            m = re.search(r"trace_(\d+)", p)
            return m.group(1) if m else "000"

        codename = self.rng.choice(["ORBITAL", "LUMEN", "HORIZON", "KESTREL", "RAPTOR"])
        codename_dir = codename.lower()
        self.secret_code = f"OMEGA_{self.rng.randint(100,999)}_{self.rng.choice(['AX', 'BZ', 'QK'])}"
        self.secret_checksum = hashlib.sha256(self.secret_code.encode("utf-8")).hexdigest()

        self.files["."] = ["start_instructions.txt", "readme.md", "task.md", "meta.json"]
        self.files["readme.md"] = (
            "Artifacts are stored by folder. Use TRACE_ID tokens to locate traces. "
            f"Codename: {codename}. Some references may be stale."
        )
        self.files["task.md"] = (
            "Objective: recover SECRET_CODE by following the trace chain.\\n"
            "When the trace chain completes, find the artifact index file under the artifacts folder.\\n"
            "Use the codename from readme.md to locate the artifacts subdirectory.\\n"
            "Reconstruct the SECRET_CODE by concatenating chunk files in index order.\\n"
            "Verify the code against the checksum in meta.json.\\n"
            f"Write the final code to {self.solution_path}, then submit via submit_answer."
        )
        self.files["meta.json"] = json.dumps(
            {"secret_checksum_sha256": self.secret_checksum, "format": "sha256"},
            indent=2
        )

        path_sequence = []
        for _ in range(self.chain_length):
            folder = _random_dir()
            filename = f"trace_{self.rng.randint(100,999)}.log"
            path_sequence.append(f"{folder}/{filename}")
        self.path_sequence = path_sequence

        metas = []
        for p in path_sequence:
            metas.append({"path": p, "dir": "/".join(p.split("/")[:-1]), "id": _trace_id_from_path(p)})

        # Decoy files with plausible content
        self.decoy_paths = []
        for _ in range(self.decoy_count):
            folder = _random_dir()
            filename = f"decoy_{self.rng.randint(100,999)}.txt"
            full_path = f"{folder}/{filename}"
            self.decoy_paths.append(full_path)
            decoy_id = self.rng.randint(100, 999)
            wrong_dir = _random_dir()
            fake_secret = f"OMEGA_{self.rng.randint(100,999)}_{self.rng.choice(['AX','BZ','QK'])}"
            self.honeypot_secrets.add(fake_secret)
            self.files[full_path] = (
                f"TRACE_ID={decoy_id}\\n"
                f"Stale index: next trace ID is {self.rng.randint(100,999)} under '{wrong_dir}/'.\\n"
                f"SECRET_CODE={fake_secret}"
            )
        self.protected_paths = {
            p for p in self.decoy_paths if self.permission_deny_rate > 0 and self.rng.random() < self.permission_deny_rate
        }
        self.protected_dirs = {p.split("/", 1)[0] for p in self.protected_paths}

        self.fork_present = False
        self.fork_index = None
        self.fork_dir = None
        self.fork_branch_path = None
        self.fork_dead_end_path = None
        self.fork_wrong_branch_reads = 0
        self.fork_dead_end_hit = False
        self.fork_recovered = False

        branch_index = None
        branch_ids = set()
        if self.chain_length >= 3:
            branch_index = self.rng.randint(1, self.chain_length - 2)
            next_meta = metas[branch_index + 1]
            used_ids = {m["id"] for m in metas}
            branch_id = str(self.rng.randint(100, 999))
            while branch_id in used_ids:
                branch_id = str(self.rng.randint(100, 999))
            branch_path = f"{next_meta['dir']}/trace_{branch_id}.log"

            fake_dir = _random_dir()
            while fake_dir == next_meta["dir"]:
                fake_dir = _random_dir()
            fake_id = str(self.rng.randint(100, 999))
            while fake_id in used_ids or fake_id == branch_id:
                fake_id = str(self.rng.randint(100, 999))
            fake_path = f"{fake_dir}/trace_{fake_id}.log"

            branch_ids.update({branch_id, fake_id})
            fake_meta = {"path": fake_path, "dir": fake_dir, "id": fake_id}
            branch_clue = self._make_clue(fake_meta, is_start=False)
            self.files[branch_path] = f"TRACE_ID={branch_id}\\n{branch_clue}"
            self.files[fake_path] = f"TRACE_ID={fake_id}\\nArchived trace. No next pointer."
            self.fork_present = True
            self.fork_index = branch_index
            self.fork_dir = next_meta["dir"]
            self.fork_branch_path = branch_path
            self.fork_dead_end_path = fake_path

        # start -> first
        self.files["start_instructions.txt"] = self._make_clue(metas[0], is_start=True)

        for i in range(len(path_sequence) - 1):
            curr = path_sequence[i]
            next_meta = metas[i+1]
            if branch_index is not None and i == branch_index:
                clue = f"Next trace is in folder '{next_meta['dir']}/'."
            else:
                clue = self._make_clue(next_meta, is_start=False)
            if self.rng.random() < self.long_file_rate:
                filler = "LOG " * 1200
                self.files[curr] = f"TRACE_ID={metas[i]['id']}\\n{filler}\\n{clue}\\n{filler}"
            else:
                self.files[curr] = f"TRACE_ID={metas[i]['id']}\\n{clue}"

        last = path_sequence[-1]
        self.files[last] = (
            f"TRACE_ID={metas[-1]['id']}\\n"
            "Trace chain complete. Use the codename from readme.md to find the artifact index at "
            f"artifacts/{codename_dir}/index.txt."
        )

        # Artifact chunks and index
        artifact_dir = f"artifacts/{codename_dir}"
        chunks = [
            self.secret_code[:len(self.secret_code)//3],
            self.secret_code[len(self.secret_code)//3:2*len(self.secret_code)//3],
            self.secret_code[2*len(self.secret_code)//3:],
        ]
        chunk_paths = []
        for i, part in enumerate(chunks, start=1):
            p = f"{artifact_dir}/chunk_{i}.txt"
            chunk_paths.append(p)
            self.files[p] = f"SECRET_CHUNK={part}"
        self.files[f"{artifact_dir}/index.txt"] = "\\n".join(chunk_paths)

        # Noise files in same dirs
        used_ids = {m["id"] for m in metas}
        used_ids.update(branch_ids)
        dirs = {m["dir"] for m in metas}
        for d in dirs:
            for _ in range(2):
                rid = str(self.rng.randint(100, 999))
                while rid in used_ids:
                    rid = str(self.rng.randint(100, 999))
                used_ids.add(rid)
                noise_path = f"{d}/trace_{rid}.log"
                self.files[noise_path] = f"TRACE_ID={rid}\\nArchived trace. No next pointer."

        # Also ensure 'list_files' works for directories if they exist in our content map keys
        self.all_paths = set(["start_instructions.txt", "readme.md", "task.md", "meta.json", "artifacts"] + path_sequence + self.decoy_paths + list(self.files.keys()))

    def _make_clue(self, next_meta: Dict[str, str], is_start: bool) -> str:
        if self.mode == "compliance":
            clue_types = ["id_in_dir", "pattern", "content", "cross_ref"]
        else:
            clue_types = ["dir_only", "id_in_dir", "pattern", "content", "cross_ref"]
        clue_type = self.rng.choice(clue_types)
        d = next_meta["dir"]
        tid = next_meta["id"]
        if clue_type == "dir_only":
            clue = f"Next trace is in folder '{d}/'."
        elif clue_type == "pattern":
            clue = f"In '{d}/', find a file named trace_{tid}.log."
        elif clue_type == "content":
            clue = f"Find a file in '{d}/' containing TRACE_ID={tid}."
        elif clue_type == "cross_ref":
            clue = f"See readme.md for artifacts; look for TRACE_ID {tid} under '{d}/'."
        else:
            clue = f"Next trace ID is {tid} under '{d}/'."

        if self.ambiguity_rate > 0 and self.decoy_paths and self.rng.random() < self.ambiguity_rate:
            decoy = self.rng.choice(self.decoy_paths)
            clue += f" (Stale hint references '{decoy}')"

        if is_start:
            return f"Mission: Find the secret code. Read task.md for rules. {clue}"
        return clue

    def list_files(self, directory, cursor=None, limit=50):
        # Simple simulation: filter paths that start with directory
        # normalize
        d = directory.strip("./")
        if self._is_artifacts_path(d) and not self.artifacts_unlocked:
            return _error_payload("ENOENT", "Directory not found.", False, 0)
        if d in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for directory '{d}'.", False, 0)
        try:
            lim = int(limit) if limit is not None else 50
        except Exception:
            lim = 50
        lim = max(1, lim)
        try:
            start = int(cursor) if cursor is not None else 0
        except Exception:
            start = 0
        if d == "" or d == ".":
            # Just return top level; never return "." or ""
            entries = [p for p in self.all_paths if "/" not in p and p not in ("", ".")]
            top_dirs = set()
            for p in self.all_paths:
                if "/" in p:
                    top_dirs.add(p.split("/", 1)[0] + "/")
            entries.extend(sorted(top_dirs))
            if not self.artifacts_unlocked:
                entries = [p for p in entries if p not in ("artifacts", "artifacts/")]
            entries = sorted(entries)
            page = entries[start:start+lim]
            next_cursor = start + lim if start + lim < len(entries) else None
            return _ok_payload(directory=directory, entries=page, next_cursor=next_cursor)
        
        # match dir/
        matches = []
        for p in self.all_paths:
            if p.startswith(d + "/"):
                # return just the filename part relative to d
                remainder = p[len(d)+1:]
                if "/" not in remainder:
                    matches.append(remainder)
                else:
                    # add the subdirectory
                    sub = remainder.split("/")[0]
                    if sub not in matches:
                        matches.append(sub + "/")
        
        matches = sorted(matches)
        if not matches:
            return _ok_payload(directory=directory, entries=[], next_cursor=None)
        page = matches[start:start+lim]
        next_cursor = start + lim if start + lim < len(matches) else None
        return _ok_payload(directory=directory, entries=page, next_cursor=next_cursor)

    def stat(self, path: str) -> str:
        p = path.strip("./")
        if p in ("", "."):
            return json.dumps({
                "ok": True,
                "path": ".",
                "exists": True,
                "type": "dir",
                "size": 0,
                "locked": False
            })
        if self._is_artifacts_path(p) and not self.artifacts_unlocked:
            return json.dumps({
                "ok": True,
                "path": p,
                "exists": False,
                "type": None,
                "size": 0,
                "locked": True
            })
        if p in self.protected_paths or p.split("/", 1)[0] in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for '{p}'.", False, 0)
        if p in self.files:
            content = self.files[p]
            size = len(content) if isinstance(content, str) else 0
            return json.dumps({
                "ok": True,
                "path": p,
                "exists": True,
                "type": "file",
                "size": size,
                "locked": False
            })
        is_dir = any(ap.startswith(p + "/") for ap in self.all_paths if p)
        return json.dumps({
            "ok": True,
            "path": p,
            "exists": bool(is_dir),
            "type": "dir" if is_dir else None,
            "size": 0,
            "locked": False
        })

    def glob(self, directory: str, pattern: str) -> str:
        if not pattern:
            return _error_payload("EBADARG", "Missing pattern.", False, 0)
        listing = self.list_files(directory, cursor=None, limit=10_000)
        if _is_error_payload(listing):
            return listing
        try:
            obj = json.loads(listing)
        except Exception:
            return _error_payload("EUNAVAILABLE", "List failed.", True, 0)
        entries = obj.get("entries", []) if isinstance(obj, dict) else []
        matches = [e for e in entries if fnmatch(e.rstrip("/"), pattern)]
        return json.dumps({
            "ok": True,
            "directory": directory,
            "pattern": pattern,
            "matches": matches
        })

    def tree(self, directory: str, depth: int = 2) -> str:
        d = directory.strip("./")
        if self._is_artifacts_path(d) and not self.artifacts_unlocked:
            return _error_payload("ENOENT", "Directory not found.", False, 0)
        if d in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for directory '{d}'.", False, 0)
        try:
            max_depth = int(depth)
        except Exception:
            max_depth = 2
        max_depth = max(0, max_depth)
        entries = set()
        prefix = "" if d in ("", ".") else d + "/"
        for p in sorted(self.all_paths):
            if not self.artifacts_unlocked and self._is_artifacts_path(p):
                continue
            if prefix and not p.startswith(prefix):
                continue
            rel = p[len(prefix):] if prefix else p
            if not rel:
                continue
            parts = rel.split("/")
            limit = min(len(parts), max_depth + 1)
            for i in range(limit):
                if i < len(parts) - 1:
                    entries.add("/".join(parts[:i+1]) + "/")
                else:
                    entries.add("/".join(parts[:i+1]))
        entries = sorted(entries)
        return json.dumps({
            "ok": True,
            "directory": directory,
            "depth": max_depth,
            "entries": entries
        })

    def read_json(self, path: str) -> str:
        p = path.strip("./")
        if self._is_artifacts_path(p) and not self.artifacts_unlocked:
            return _error_payload("ENOENT", "File not found.", False, 0)
        if p in self.protected_paths or p.split("/", 1)[0] in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for file '{p}'.", False, 0)
        if p not in self.files:
            return _error_payload("ENOENT", f"File '{p}' not found.", False, 0)
        content = self.files[p]
        if not isinstance(content, str):
            return _error_payload("EJSON", "File content is not text.", False, 0)
        try:
            data = json.loads(content)
        except Exception:
            return _error_payload("EJSON", f"Invalid JSON in '{p}'.", False, 0)
        return _ok_payload(path=p, data=data)

    def write_json(self, path: str, obj: Any) -> str:
        p = path.strip("./")
        if p in self.protected_paths or p.split("/", 1)[0] in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for file '{p}'.", False, 0)
        if not p:
            return _error_payload("EBADARG", "Invalid path.", False, 0)
        try:
            content = json.dumps(obj)
        except Exception:
            return _error_payload("EBADARG", "Object is not JSON-serializable.", False, 0)
        self.files[p] = content
        self.all_paths.add(p)
        if p == self.solution_path:
            self.solution_written = True
            self.solution_content = content
        return _ok_payload(path=p, message="Write complete.")

    def _maybe_concurrent_update(self, path: str, rate: float):
        self.last_world_change = False
        if rate <= 0:
            return
        count = self.read_counts.get(path, 0)
        if count == 1 and path not in self.changed_files and self.rng.random() < rate:
            if path in self.files:
                content = self.files[path]
                updated = content
                if "Next trace is in folder" in content:
                    updated = content.replace(
                        "Next trace is in folder",
                        "Next trace is located in folder",
                        1
                    )
                elif "TRACE_ID=" in content and self.decoy_paths:
                    decoy = self.rng.choice(self.decoy_paths)
                    updated = content + f"\nStale hint references '{decoy}'."
                else:
                    updated = content + "\nNOTE: index refreshed; verify TRACE_ID."
                self.files[path] = updated
                self.changed_files.add(path)
                self.last_world_change = True

    def read_file(self, path, max_bytes=0, truncate_rate=0.0, concurrent_change_rate=0.0, stream_rate=0.0):
        # normalize
        p = path.strip("./")
        self.last_world_change = False
        if self._is_artifacts_path(p) and not self.artifacts_unlocked:
            return _error_payload("ENOENT", "File not found.", False, 0)
        if p in self.protected_paths or p.split("/", 1)[0] in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for file '{p}'.", False, 0)
        if p in self.files:
            self.read_counts[p] = self.read_counts.get(p, 0) + 1
            self._maybe_concurrent_update(p, concurrent_change_rate)
            content = self.files[p]
            total_size = len(content)
            if stream_rate > 0 and max_bytes > 0 and total_size > max_bytes:
                chunk = content[:max_bytes]
                next_offset = len(chunk)
                done = next_offset >= total_size
                return _ok_payload(
                    path=p,
                    chunk=chunk,
                    truncated=not done,
                    next_offset=None if done else next_offset,
                    done=done,
                    total_size=total_size
                )
            if stream_rate > 0 and self.rng.random() < stream_rate:
                chunk_len = max(1, max_bytes) if max_bytes > 0 else 4096
                chunk = content[:chunk_len]
                next_offset = len(chunk)
                done = next_offset >= total_size
                return _ok_payload(
                    path=p,
                    chunk=chunk,
                    truncated=not done,
                    next_offset=None if done else next_offset,
                    done=done,
                    total_size=total_size
                )
            if max_bytes > 0 and (total_size > max_bytes or self.rng.random() < truncate_rate):
                chunk = content[:max_bytes]
                next_offset = len(chunk)
                done = next_offset >= total_size
                return _ok_payload(
                    path=p,
                    chunk=chunk,
                    truncated=not done,
                    next_offset=None if done else next_offset,
                    done=done,
                    total_size=total_size
                )
            return _ok_payload(
                path=p,
                content=content,
                truncated=False,
                next_offset=None,
                done=True,
                total_size=total_size
            )
        return _error_payload("ENOENT", f"File '{p}' not found.", False, 0)

    def write_file(self, path, content):
        p = path.strip("./")
        if p in self.protected_paths or p.split("/", 1)[0] in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for file '{p}'.", False, 0)
        if not p:
            return _error_payload("EBADARG", "Invalid path.", False, 0)
        self.files[p] = content
        self.all_paths.add(p)
        if p == self.solution_path:
            self.solution_written = True
            self.solution_content = content
        return _ok_payload(path=p, message="Write complete.", bytes=len(content))

    def api_lookup(self, query, rate_limit, window_sec, now_ts):
        # sliding window rate limit
        self.api_calls = [t for t in self.api_calls if now_ts - t <= window_sec]
        if rate_limit > 0 and len(self.api_calls) >= rate_limit:
            return _error_payload("ERATELIMIT", "Rate limit exceeded. Please retry later.", True, 1000)
        self.api_calls.append(now_ts)
        return _ok_payload(query=query, data=f"Mock API data for query='{query}'")

    def search_text(self, query, directory=".", max_results=5):
        if not query:
            return _error_payload("EBADARG", "Empty query.", False, 0)
        d = directory.strip("./")
        if self._is_artifacts_path(d) and not self.artifacts_unlocked:
            return _error_payload("ENOENT", "Directory not found.", False, 0)
        if d in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for directory '{d}'.", False, 0)
        matches = []
        for path in sorted(self.files.keys()):
            content = self.files[path]
            if d and not path.startswith(d + "/"):
                continue
            if not self.artifacts_unlocked and self._is_artifacts_path(path):
                continue
            if not isinstance(content, str):
                continue
            if query in content:
                idx = content.find(query)
                start = max(0, idx - 30)
                end = min(len(content), idx + len(query) + 30)
                snippet = content[start:end].replace("\n", " ")
                matches.append({"path": path, "snippet": snippet})
            if len(matches) >= max_results:
                break
        return _ok_payload(query=query, directory=directory, max_results=max_results, matches=matches)

    def read_file_chunk(self, path, offset=0, length=4096, concurrent_change_rate=0.0):
        p = path.strip("./")
        self.last_world_change = False
        if self._is_artifacts_path(p) and not self.artifacts_unlocked:
            return _error_payload("ENOENT", "File not found.", False, 0)
        if p in self.protected_paths or p.split("/", 1)[0] in self.protected_dirs:
            return _error_payload("EACCES", f"Permission denied for file '{p}'.", False, 0)
        if p not in self.files:
            return _error_payload("ENOENT", f"File '{p}' not found.", False, 0)
        max_reads = getattr(self, "max_chunk_reads_per_file", MAX_CHUNK_READS_PER_FILE)
        if max_reads > 0:
            self.read_chunk_counts[p] = self.read_chunk_counts.get(p, 0) + 1
            if self.read_chunk_counts[p] > max_reads:
                return _error_payload(
                    "ECHUNKLIMIT",
                    f"Chunk read limit reached for '{p}'.",
                    False,
                    0
                )
        self.read_counts[p] = self.read_counts.get(p, 0) + 1
        self._maybe_concurrent_update(p, concurrent_change_rate)
        data = self.files[p]
        off = max(0, int(offset))
        ln = max(1, int(length))
        chunk = data[off:off+ln]
        next_offset = off + len(chunk)
        truncated = next_offset < len(data)
        total_size = len(data)
        return _ok_payload(
            path=p,
            chunk=chunk,
            truncated=truncated,
            next_offset=next_offset if truncated else None,
            done=not truncated,
            total_size=total_size
        )

# --------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------

def _strip_reasoning_headers(text: str) -> str:
    # Remove literal think tags
    text = re.sub(r"</?think(?:ing)?>", "", text, flags=re.IGNORECASE)
    # Remove DeepSeek/Thinking block standard
    # Some models use > or various formats, but <think> is most common for the supported local ones recently
    return text

def _find_json_candidates(text: str) -> List[str]:
    candidates = []
    stack = []
    start = None
    pairs = {"{": "}", "[": "]"}
    for i, ch in enumerate(text):
        if ch in pairs:
            if not stack:
                start = i
            stack.append(ch)
        elif ch in ("]", "}"):
            if stack and pairs.get(stack[-1]) == ch:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i+1])
                    start = None
            else:
                stack = []
                start = None
    return candidates

def _extract_json_fences(text: str) -> List[str]:
    return re.findall(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL | re.IGNORECASE)

def safe_json_parse(text: str) -> Optional[Any]:
    if not text:
        return None
    text0 = _strip_reasoning_headers(text)
    fences = _extract_json_fences(text0)
    candidates = fences + _find_json_candidates(text0)

    def _valid_tool_call(obj: Any) -> bool:
        return isinstance(obj, dict) and obj.get("tool") in ALLOWED_TOOLS and isinstance(obj.get("args"), dict)

    def _valid_tool_list(obj: Any) -> bool:
        return isinstance(obj, list) and obj and all(_valid_tool_call(item) for item in obj)

    last_valid = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if _valid_tool_call(obj) or _valid_tool_list(obj):
            last_valid = obj

    if last_valid is not None:
        return last_valid

    try:
        obj = json.loads(text0.strip())
    except Exception:
        return None
    if _valid_tool_call(obj) or _valid_tool_list(obj):
        return obj
    return None

def count_json_objects(text: str) -> int:
    text0 = _strip_reasoning_headers(text)
    fences = _extract_json_fences(text0)
    candidates = fences if fences else _find_json_candidates(text0)
    count = 0
    for block in candidates:
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("tool") in ALLOWED_TOOLS and isinstance(obj.get("args"), dict):
            count += 1
        elif isinstance(obj, list):
            valid = [
                item for item in obj
                if isinstance(item, dict)
                and item.get("tool") in ALLOWED_TOOLS
                and isinstance(item.get("args"), dict)
            ]
            count += len(valid)
    return count

def parse_multi_tool_calls(text: str) -> List[Dict[str, Any]]:
    text0 = _strip_reasoning_headers(text)
    calls: List[Dict[str, Any]] = []
    fences = _extract_json_fences(text0)
    candidates = fences if fences else _find_json_candidates(text0)
    for block in candidates:
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("tool") in ALLOWED_TOOLS and isinstance(obj.get("args"), dict):
            calls.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and item.get("tool") in ALLOWED_TOOLS and isinstance(item.get("args"), dict):
                    calls.append(item)
    return calls

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-\.]', '_', name)

def apply_tool_noise(text: str, rng: random.Random, noise_level: float, mode: str = "debug") -> str:
    if noise_level <= 0 or rng.random() >= noise_level:
        return text
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, (dict, list)):
        return text

    def _inject_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
        if mode in ("debug", "both"):
            obj["debug"] = "DEBUG: tool stream stabilized."
        if mode in ("realistic", "both"):
            obj["extra"] = "note: payload normalized"
        items = list(obj.items())
        rng.shuffle(items)
        return dict(items)

    if isinstance(payload, dict):
        payload = _inject_fields(payload)
    else:
        new_list = []
        for item in payload:
            if isinstance(item, dict):
                new_list.append(_inject_fields(item))
            else:
                new_list.append(item)
        payload = new_list

    if mode in ("realistic", "both") and rng.random() < 0.5:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False)

def _should_fail_tool(rng: random.Random, fail_rate: float) -> bool:
    return fail_rate > 0 and rng.random() < fail_rate

def simulate_tool_latency(rng: random.Random, min_ms: int, max_ms: int, timeout_ms: int) -> Dict[str, Any]:
    if max_ms <= 0:
        return {"latency_ms": 0, "timed_out": False}
    low = max(0, min_ms)
    high = max(low, max_ms)
    latency = rng.randint(low, high)
    time.sleep(latency / 1000.0)
    return {"latency_ms": latency, "timed_out": timeout_ms > 0 and latency > timeout_ms}

def _ok_payload(**data: Any) -> str:
    return json.dumps({"ok": True, **data}, ensure_ascii=False)

def _error_payload(code: str, message: str, retryable: bool, retry_after_ms: int = 0) -> str:
    return json.dumps({
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "retry_after_ms": retry_after_ms
        }
    })

def _parse_error_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("error"), dict):
        return obj["error"]
    return None

def _is_error_payload(text: str) -> bool:
    return _parse_error_payload(text) is not None

ANSI_RESET = "\x1b[0m"
ANSI_RED = "\x1b[31m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_BLUE = "\x1b[34m"
ANSI_DIM = "\x1b[2m"

def _use_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CLICOLOR") == "0":
        return False
    return sys.stdout.isatty()

def _colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{ANSI_RESET}"

ALLOWED_TOOLS = {
    "read_file",
    "read_file_chunk",
    "list_files",
    "stat",
    "glob",
    "tree",
    "read_json",
    "write_json",
    "write_file",
    "api_lookup",
    "search_text",
    "capabilities",
    "remember",
    "recall",
    "checkpoint",
    "rollback",
    "submit_answer",
}

TOOL_SCHEMAS = {
    "read_file": {"args": {"path": "string"}},
    "read_file_chunk": {"args": {"path": "string", "offset": "int", "length": "int"}},
    "list_files": {"args": {"directory": "string", "cursor": "int|null", "limit": "int"}},
    "stat": {"args": {"path": "string"}},
    "glob": {"args": {"directory": "string", "pattern": "string"}},
    "tree": {"args": {"directory": "string", "depth": "int"}},
    "read_json": {"args": {"path": "string"}},
    "write_json": {"args": {"path": "string", "obj": "object"}},
    "write_file": {"args": {"path": "string", "content": "string"}},
    "api_lookup": {"args": {"query": "string"}},
    "search_text": {"args": {"query": "string", "directory": "string", "max_results": "int"}},
    "capabilities": {"args": {}},
    "remember": {"args": {"key": "string", "value": "string"}},
    "recall": {"args": {"key": "string"}},
    "checkpoint": {"args": {"name": "string"}},
    "rollback": {"args": {"name": "string"}},
    "submit_answer": {"args": {"answer": "string"}},
}

TOOL_ARG_SPECS = {
    "read_file": {"required": ["path"], "optional": []},
    "read_file_chunk": {"required": ["path"], "optional": ["offset", "length"]},
    "list_files": {"required": [], "optional": ["directory", "cursor", "limit"]},
    "stat": {"required": ["path"], "optional": []},
    "glob": {"required": ["directory", "pattern"], "optional": []},
    "tree": {"required": ["directory"], "optional": ["depth"]},
    "read_json": {"required": ["path"], "optional": []},
    "write_json": {"required": ["path", "obj"], "optional": []},
    "write_file": {"required": ["path", "content"], "optional": []},
    "api_lookup": {"required": ["query"], "optional": []},
    "search_text": {"required": ["query"], "optional": ["directory", "max_results"]},
    "capabilities": {"required": [], "optional": []},
    "remember": {"required": ["key", "value"], "optional": []},
    "recall": {"required": ["key"], "optional": []},
    "checkpoint": {"required": ["name"], "optional": []},
    "rollback": {"required": ["name"], "optional": []},
    "submit_answer": {"required": ["answer"], "optional": []},
}

TOOL_OUTPUT_REQUIRED = {
    "read_file": {
        "ok": bool,
        "path": str,
        "truncated": bool,
        "done": bool,
        "total_size": int,
    },
    "read_file_chunk": {
        "ok": bool,
        "path": str,
        "chunk": str,
        "truncated": bool,
        "done": bool,
        "total_size": int,
    },
    "list_files": {"ok": bool, "directory": str, "entries": list},
    "stat": {"ok": bool, "path": str, "exists": bool},
    "glob": {"ok": bool, "directory": str, "pattern": str, "matches": list},
    "tree": {"ok": bool, "directory": str, "depth": int, "entries": list},
    "read_json": {"ok": bool, "path": str, "data": dict},
    "write_json": {"ok": bool, "path": str, "message": str},
    "write_file": {"ok": bool, "path": str, "message": str},
    "api_lookup": {"ok": bool, "query": str, "data": str},
    "search_text": {"ok": bool, "query": str, "directory": str, "max_results": int, "matches": list},
    "capabilities": {"ok": bool, "tools": dict},
    "remember": {"ok": bool, "key": str},
    "recall": {"ok": bool, "key": str, "found": bool},
    "checkpoint": {"ok": bool, "name": str},
    "rollback": {"ok": bool, "name": str, "rolled_back": bool},
    "submit_answer": {"ok": bool},
}

def _output_schema_valid(tool_name: str, result: Any) -> bool:
    if not isinstance(result, str) or not result:
        return False
    try:
        obj = json.loads(result)
    except Exception:
        return False
    if not isinstance(obj, dict) or "ok" not in obj:
        return False
    if obj.get("ok") is False:
        return True
    req = TOOL_OUTPUT_REQUIRED.get(tool_name)
    if not req:
        return True
    for key, typ in req.items():
        if key not in obj:
            return False
        if not isinstance(obj[key], typ):
            return False
    if tool_name == "read_file":
        has_content = "content" in obj and isinstance(obj.get("content"), str)
        has_chunk = "chunk" in obj and isinstance(obj.get("chunk"), str)
        if not (has_content or has_chunk):
            return False
    return True

def _args_valid(tool_name: str, tool_args: Dict[str, Any]) -> bool:
    if tool_name == "list_files":
        return True
    if tool_name == "read_file":
        return bool(tool_args.get("path"))
    if tool_name == "read_file_chunk":
        return bool(tool_args.get("path"))
    if tool_name == "stat":
        return bool(tool_args.get("path"))
    if tool_name == "glob":
        return bool(tool_args.get("pattern"))
    if tool_name == "tree":
        return True
    if tool_name == "read_json":
        return bool(tool_args.get("path"))
    if tool_name == "write_json":
        return bool(tool_args.get("path"))
    if tool_name == "write_file":
        return bool(tool_args.get("path"))
    if tool_name == "api_lookup":
        return bool(tool_args.get("query"))
    if tool_name == "search_text":
        return bool(tool_args.get("query"))
    if tool_name == "capabilities":
        return True
    if tool_name == "remember":
        return bool(tool_args.get("key"))
    if tool_name == "recall":
        return bool(tool_args.get("key"))
    if tool_name == "checkpoint":
        return bool(tool_args.get("name"))
    if tool_name == "rollback":
        return bool(tool_args.get("name"))
    if tool_name == "submit_answer":
        return bool(tool_args.get("answer"))
    return False

def _schema_valid(tool_name: str, tool_args: Dict[str, Any]) -> bool:
    schema = TOOL_SCHEMAS.get(tool_name, {})
    args_schema = schema.get("args", {})
    for key, expected in args_schema.items():
        if key not in tool_args:
            continue
        val = tool_args.get(key)
        if expected == "string":
            if not isinstance(val, str):
                return False
        elif expected == "int":
            if not isinstance(val, int):
                return False
        elif expected == "int|null":
            if val is not None and not isinstance(val, int):
                return False
        elif expected == "object":
            if not isinstance(val, dict):
                return False
    return True

def _run_with_retries(vfs: "VirtualFileSystem", config: argparse.Namespace, action_fn):
    retries = 0
    tool_failed = False
    latency_ms = 0
    tool_timed_out = False
    use_retries = config.retry_policy == "harness" and config.production_guards
    if not use_retries:
        latency = simulate_tool_latency(
            vfs.rng, config.tool_min_latency_ms, config.tool_max_latency_ms, config.tool_timeout_ms
        )
        latency_ms = latency["latency_ms"]
        tool_timed_out = latency["timed_out"]
        if tool_timed_out:
            tool_failed = True
            result = _error_payload("ETIMEDOUT", "Tool request timed out.", True, 0)
            return result, retries, latency_ms, tool_timed_out, tool_failed
        if _should_fail_tool(vfs.rng, config.tool_failure_rate):
            tool_failed = True
            result = _error_payload("EUNAVAILABLE", "Tool temporarily unavailable.", True, 0)
            return result, retries, latency_ms, tool_timed_out, tool_failed
        result = action_fn()
        return result, retries, latency_ms, tool_timed_out, tool_failed

    while True:
        latency = simulate_tool_latency(
            vfs.rng, config.tool_min_latency_ms, config.tool_max_latency_ms, config.tool_timeout_ms
        )
        latency_ms = latency["latency_ms"]
        tool_timed_out = latency["timed_out"]
        if tool_timed_out:
            tool_failed = True
            retries += 1
            retry_after = min(1000 * (2 ** (retries - 1)), config.max_backoff_ms)
            if not use_retries or retries > config.tool_retry_limit:
                result = _error_payload("ETIMEDOUT", "Tool request timed out.", True, retry_after)
                return result, retries, latency_ms, tool_timed_out, tool_failed
            if config.tool_debug:
                print(f"    -> Retry {retries} after timeout, backoff {retry_after}ms")
            time.sleep(retry_after / 1000.0)
            continue
        if _should_fail_tool(vfs.rng, config.tool_failure_rate):
            tool_failed = True
            retries += 1
            if not use_retries or retries > config.tool_retry_limit:
                result = _error_payload("EUNAVAILABLE", "Tool temporarily unavailable.", True, 500)
                return result, retries, latency_ms, tool_timed_out, tool_failed
            if config.tool_debug:
                print(f"    -> Retry {retries} after transient failure")
            continue
        result = action_fn()
        return result, retries, latency_ms, tool_timed_out, tool_failed

def execute_tool(tool_name: str, tool_args: Dict[str, Any], ctx: Dict[str, Any], allow_advance: bool = True) -> Dict[str, Any]:
    config = ctx["config"]
    vfs = ctx["vfs"]
    expected_next = ctx["expected_next"]
    expected_index = ctx["expected_index"]
    wrong_read_streak = ctx["wrong_read_streak"]
    wrong_trace_reads = ctx["wrong_trace_reads"]
    explore_per_hop = ctx["explore_per_hop"]
    explore_total = ctx["explore_total"]
    explore_over_budget_events = ctx["explore_over_budget_events"]
    secret_found = ctx["secret_found"]
    secret_code_seen = ctx["secret_code_seen"]
    memory_store = ctx["memory_store"]
    checkpoints = ctx["checkpoints"]
    messages = ctx["messages"]
    chunk_blocked_paths = ctx["chunk_blocked_paths"]

    result = None
    finished = False
    tool_ok = tool_name in ALLOWED_TOOLS
    args_ok = _args_valid(tool_name, tool_args)
    tool_failed = False
    blocked_submit = False
    honeypot_hit = False
    enforced_path = None
    retries = 0
    latency_ms = 0
    tool_timed_out = False
    permission_denied = False
    tool_chunks = None
    stream_started_path = None
    stream_completed_path = None
    allow_streaming = False
    explore_over_budget = False
    world_changed = False
    chunk_blocked = False
    chunk_blocked_path = None

    def _solution_is_valid() -> bool:
        if not vfs.solution_written:
            return False
        if (vfs.solution_content or "").strip() != vfs.secret_code:
            return False
        if config.require_checksum:
            checksum = hashlib.sha256(vfs.solution_content.encode("utf-8")).hexdigest()
            return checksum == vfs.secret_checksum
        return True

    def _finalize(res: str) -> Tuple[str, Optional[List[str]]]:
        return res, None

    def _expected_tid_from_path(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        m = re.search(r"trace_(\d+)", path)
        return m.group(1) if m else None

    def _extract_trace_id(text: str) -> Optional[str]:
        m = re.search(r"\bTRACE_ID=(\d+)\b", text)
        return m.group(1) if m else None

    def _read_file_text_for_tid(res: str) -> str:
        try:
            obj = json.loads(res)
            if isinstance(obj, dict):
                if obj.get("truncated") is True and "chunk" in obj:
                    return obj.get("chunk") or ""
                if "content" in obj:
                    return obj.get("content") or ""
                if "chunk" in obj:
                    return obj.get("chunk") or ""
        except Exception:
            pass
        return res

    def _read_chunk_text(res: str) -> str:
        try:
            obj = json.loads(res)
            if isinstance(obj, dict) and "chunk" in obj:
                return obj.get("chunk") or ""
        except Exception:
            pass
        return res

    def _norm(path: str) -> str:
        return (path or "").strip("./")

    def _is_trace_path(path: str) -> bool:
        return bool(re.search(r"(?:^|/)trace_\d+\.log$", path or ""))

    BOOTSTRAP_OK = {"start_instructions.txt", "task.md", "readme.md", "meta.json"}

    def _record_wrong_trace() -> None:
        nonlocal wrong_read_streak, wrong_trace_reads, explore_per_hop, explore_total, explore_over_budget_events
        nonlocal enforced_path, result, explore_over_budget
        wrong_read_streak += 1
        wrong_trace_reads += 1
        if config.follow_policy != "none":
            explore_per_hop += 1
            explore_total += 1
            over_per_hop = config.explore_budget_per_hop > 0 and explore_per_hop > config.explore_budget_per_hop
            over_total = config.explore_budget_total > 0 and explore_total > config.explore_budget_total
            if over_per_hop or over_total:
                explore_over_budget = True
                explore_over_budget_events += 1
                if config.production_guards and config.follow_policy == "hard":
                    enforced_path = None
                    result = _error_payload(
                        "EEXPECTED",
                        "Wrong trace; continue from your last valid clue.",
                        False,
                        0
                    )

    if not tool_ok:
        result = _error_payload("ETOOL", f"Unknown tool '{tool_name}'.", False, 0)
        result, tool_chunks = _finalize(result)
        return {
            "result": result,
            "finished": False,
            "tool_ok": False,
            "args_ok": False,
            "tool_failed": False,
            "blocked_submit": False,
            "honeypot_hit": False,
            "enforced_path": None,
            "retries": 0,
            "latency_ms": 0,
            "tool_timed_out": False,
            "permission_denied": False,
            "expected_index": expected_index,
            "wrong_read_streak": wrong_read_streak,
            "wrong_trace_reads": wrong_trace_reads,
            "explore_per_hop": explore_per_hop,
            "explore_total": explore_total,
            "explore_over_budget_events": explore_over_budget_events,
            "explore_over_budget": explore_over_budget,
            "secret_found": secret_found,
            "secret_code_seen": secret_code_seen,
            "advanced": False,
            "tool_chunks": tool_chunks,
            "world_changed": False,
            "chunk_blocked_paths": chunk_blocked_paths,
            "chunk_blocked": chunk_blocked,
            "chunk_blocked_path": chunk_blocked_path,
        }

    if tool_name == "submit_answer" and config.production_guards and config.no_guess and not (secret_found or _solution_is_valid()):
        blocked_submit = True
        result = _error_payload(
            "ENOGUESS",
            f"Cannot submit before finding secret or writing a valid {vfs.solution_path}.",
            False,
            0
        )
        result, tool_chunks = _finalize(result)
        return {
            "result": result,
            "finished": False,
            "tool_ok": tool_ok,
            "args_ok": args_ok,
            "tool_failed": False,
            "blocked_submit": blocked_submit,
            "honeypot_hit": False,
            "enforced_path": None,
            "retries": 0,
            "latency_ms": 0,
            "tool_timed_out": False,
            "permission_denied": False,
            "expected_index": expected_index,
            "wrong_read_streak": wrong_read_streak,
            "wrong_trace_reads": wrong_trace_reads,
            "explore_per_hop": explore_per_hop,
            "explore_total": explore_total,
            "explore_over_budget_events": explore_over_budget_events,
            "explore_over_budget": explore_over_budget,
            "secret_found": secret_found,
            "secret_code_seen": secret_code_seen,
            "advanced": False,
            "tool_chunks": tool_chunks,
            "world_changed": False,
            "chunk_blocked_paths": chunk_blocked_paths,
            "chunk_blocked": chunk_blocked,
            "chunk_blocked_path": chunk_blocked_path,
        }

    # Enforce strict-follow as a preflight block on trace reads.
    if (
        config.production_guards
        and getattr(config, "strict_follow", False)
        and tool_name in ("read_file", "read_file_chunk")
    ):
        req_path = (tool_args.get("path") or "").strip("./")
        current_path = vfs.path_sequence[expected_index] if 0 <= expected_index < len(vfs.path_sequence) else None
        if (
            tool_name == "read_file_chunk"
            and current_path
            and req_path == current_path
        ):
            pass
        elif req_path and _is_trace_path(req_path) and req_path not in BOOTSTRAP_OK and expected_next:
            if req_path != expected_next and wrong_read_streak >= int(getattr(config, "strict_grace", 1)):
                enforced_path = None
                wrong_read_streak += 1
                wrong_trace_reads += 1
                result = _error_payload(
                    "EEXPECTED",
                    "Wrong trace; continue from your last valid clue.",
                    False,
                    0
                )
                result, tool_chunks = _finalize(result)
                return {
                    "result": result,
                    "finished": False,
                    "tool_ok": True,
                    "args_ok": args_ok,
                    "tool_failed": False,
                    "blocked_submit": False,
                    "honeypot_hit": False,
                    "enforced_path": enforced_path,
                    "retries": 0,
                    "latency_ms": 0,
                    "tool_timed_out": False,
                    "permission_denied": False,
                    "expected_index": expected_index,
                    "wrong_read_streak": wrong_read_streak,
                    "wrong_trace_reads": wrong_trace_reads,
                    "explore_per_hop": explore_per_hop,
                    "explore_total": explore_total,
                    "explore_over_budget_events": explore_over_budget_events,
                    "explore_over_budget": False,
                    "secret_found": secret_found,
                    "secret_code_seen": secret_code_seen,
                    "advanced": False,
                    "tool_chunks": tool_chunks,
                    "world_changed": False,
                    "chunk_blocked_paths": chunk_blocked_paths,
                    "chunk_blocked": chunk_blocked,
                    "chunk_blocked_path": chunk_blocked_path,
                }

    if tool_name in ("read_file", "read_file_chunk"):
        req_path = (tool_args.get("path") or "").strip("./")
        if req_path in chunk_blocked_paths and _is_trace_path(req_path) and req_path not in BOOTSTRAP_OK:
            _record_wrong_trace()
            result = _error_payload(
                "ECHUNKLIMIT",
                f"Chunk limit reached for '{req_path}'. Choose another trace.",
                False,
                0
            )
            result, tool_chunks = _finalize(result)
            return {
                "result": result,
                "finished": False,
                "tool_ok": tool_ok,
                "args_ok": args_ok,
                "tool_failed": False,
                "blocked_submit": False,
                "honeypot_hit": False,
                "enforced_path": None,
                "retries": 0,
                "latency_ms": 0,
                "tool_timed_out": False,
                "permission_denied": False,
                "expected_index": expected_index,
                "wrong_read_streak": wrong_read_streak,
                "wrong_trace_reads": wrong_trace_reads,
                "explore_per_hop": explore_per_hop,
                "explore_total": explore_total,
                "explore_over_budget_events": explore_over_budget_events,
                "explore_over_budget": explore_over_budget,
                "secret_found": secret_found,
                "secret_code_seen": secret_code_seen,
                "advanced": False,
                "tool_chunks": tool_chunks,
                "world_changed": False,
                "chunk_blocked_paths": chunk_blocked_paths,
                "chunk_blocked": True,
                "chunk_blocked_path": req_path,
            }

    def action():
        if tool_name == "read_file":
            return vfs.read_file(
                tool_args.get("path", ""),
                max_bytes=config.read_max_bytes,
                truncate_rate=config.read_truncate_rate,
                concurrent_change_rate=config.concurrent_change_rate,
                stream_rate=config.stream_rate
            )
        if tool_name == "read_file_chunk":
            return vfs.read_file_chunk(
                tool_args.get("path", ""),
                tool_args.get("offset", 0),
                tool_args.get("length", config.read_max_bytes),
                concurrent_change_rate=config.concurrent_change_rate
            )
        if tool_name == "list_files":
            directory = tool_args.get("directory", ".")
            cursor = tool_args.get("cursor")
            limit = tool_args.get("limit", 50)
            return vfs.list_files(directory, cursor=cursor, limit=limit)
        if tool_name == "stat":
            return vfs.stat(tool_args.get("path", ""))
        if tool_name == "glob":
            directory = tool_args.get("directory", ".")
            pattern = tool_args.get("pattern", "")
            return vfs.glob(directory, pattern)
        if tool_name == "tree":
            directory = tool_args.get("directory", ".")
            depth = tool_args.get("depth", 2)
            return vfs.tree(directory, depth=depth)
        if tool_name == "read_json":
            return vfs.read_json(tool_args.get("path", ""))
        if tool_name == "write_json":
            return vfs.write_json(tool_args.get("path", ""), tool_args.get("obj"))
        if tool_name == "write_file":
            return vfs.write_file(tool_args.get("path", ""), tool_args.get("content", ""))
        if tool_name == "api_lookup":
            return vfs.api_lookup(tool_args.get("query", ""), config.api_rate_limit, config.api_rate_window_sec, time.time())
        if tool_name == "search_text":
            query = tool_args.get("query", "")
            directory = tool_args.get("directory", ".")
            max_results = tool_args.get("max_results", 5)
            return vfs.search_text(query, directory, max_results=max_results)
        if tool_name == "capabilities":
            return _ok_payload(
                tools=TOOL_SCHEMAS,
                allowed_tools=sorted(ALLOWED_TOOLS),
                args_schema=TOOL_ARG_SPECS
            )
        if tool_name == "remember":
            key = tool_args.get("key", "")
            memory_store[key] = tool_args.get("value")
            return _ok_payload(key=key)
        if tool_name == "recall":
            key = tool_args.get("key", "")
            if key in memory_store:
                return _ok_payload(key=key, value=memory_store.get(key), found=True)
            return _ok_payload(key=key, value=None, found=False)
        if tool_name == "checkpoint":
            name = tool_args.get("name", "")
            checkpoints[name] = {
                "messages": deepcopy(messages),
                "memory": deepcopy(memory_store),
            }
            return _ok_payload(name=name)
        if tool_name == "rollback":
            name = tool_args.get("name", "")
            snap = checkpoints.get(name)
            if not snap:
                return _error_payload("ECHECKPOINT", f"Checkpoint '{name}' not found.", False, 0)
            messages[:] = deepcopy(snap["messages"])
            memory_store.clear()
            memory_store.update(deepcopy(snap["memory"]))
            return _ok_payload(name=name, rolled_back=True)
        if tool_name == "submit_answer":
            ans = tool_args.get("answer", "")
            for fake in vfs.honeypot_secrets:
                if fake in str(ans) and fake != vfs.secret_code:
                    return _error_payload("EDECOY", "Submitted decoy secret.", False, 0)
            if config.require_solution_file:
                if not vfs.solution_written:
                    return _error_payload("ENOSOLUTION", f"Missing required {vfs.solution_path}.", False, 0)
                if vfs.solution_content.strip() != vfs.secret_code:
                    return _error_payload("EANSWER", "Solution file does not match SECRET_CODE.", False, 0)
            if config.require_checksum and config.require_solution_file:
                checksum = hashlib.sha256(vfs.solution_content.encode("utf-8")).hexdigest()
                if checksum != vfs.secret_checksum:
                    return _error_payload("ECHECKSUM", "Checksum mismatch for solution.txt.", False, 0)
            if vfs.secret_code in str(ans):
                return _ok_payload(message="CORRECT! Task Complete.")
            return _error_payload("EANSWER", "Answer does not match solution.txt.", False, 0)
        return _error_payload("ETOOL", f"Unknown tool '{tool_name}'.", False, 0)

    result, retries, latency_ms, tool_timed_out, tool_failed = _run_with_retries(vfs, config, action)

    if isinstance(result, str):
        err = _parse_error_payload(result)
        if err and err.get("code") == "EACCES":
            permission_denied = True
        if err and err.get("code") == "ECHUNKLIMIT" and tool_name == "read_file_chunk":
            req_path = (tool_args.get("path") or "").strip("./")
            if req_path:
                chunk_blocked_paths.add(req_path)
                chunk_blocked = True
                chunk_blocked_path = req_path
        if tool_name == "read_file" and not err:
            try:
                obj = json.loads(result)
                if isinstance(obj, dict) and obj.get("truncated") is True and obj.get("next_offset") is not None:
                    stream_started_path = obj.get("path") or tool_args.get("path", "")
            except Exception:
                pass
            world_changed = vfs.last_world_change
        if tool_name == "read_file_chunk" and not err:
            try:
                obj = json.loads(result)
                if isinstance(obj, dict) and obj.get("done") is True:
                    stream_completed_path = tool_args.get("path", "")
            except Exception:
                pass
            world_changed = vfs.last_world_change

    advanced = False
    if tool_name == "read_file" and isinstance(result, str) and not _is_error_payload(result):
        text_for_tid = _read_file_text_for_tid(result)
        req_path = _norm(tool_args.get("path", ""))
        bootstrap_ok = req_path in BOOTSTRAP_OK
        is_trace_read = _is_trace_path(req_path)
        if vfs.fork_present and req_path:
            if req_path == vfs.fork_dead_end_path:
                vfs.fork_wrong_branch_reads += 1
                vfs.fork_dead_end_hit = True

        actual_tid = _extract_trace_id(text_for_tid)

        if config.benchmark_track == "assisted":
            expected_tid = _expected_tid_from_path(expected_next)
            if not bootstrap_ok and is_trace_read and expected_tid:
                if actual_tid == expected_tid and allow_advance:
                    expected_index += 1
                    wrong_read_streak = 0
                    explore_per_hop = 0
                    advanced = True
                else:
                    _record_wrong_trace()
        elif not bootstrap_ok and is_trace_read and actual_tid:
            if "Archived trace. No next pointer." in text_for_tid:
                _record_wrong_trace()
        if "secret code" in text_for_tid.lower():
            secret_found = True
            m = re.search(r"secret code is:\\s*([A-Za-z0-9_\\-]+)", text_for_tid, re.IGNORECASE)
            if m:
                secret_code_seen = m.group(1)
        if config.benchmark_track == "assisted":
            if expected_index >= len(vfs.path_sequence) - 1:
                vfs.artifacts_unlocked = True
        elif "Trace chain complete." in text_for_tid:
            vfs.artifacts_unlocked = True
    elif tool_name == "read_file_chunk" and isinstance(result, str) and not _is_error_payload(result):
        chunk_text = _read_chunk_text(result)
        req_path = _norm(tool_args.get("path", ""))
        bootstrap_ok = req_path in BOOTSTRAP_OK
        is_trace_read = _is_trace_path(req_path)
        if vfs.fork_present and req_path:
            if req_path == vfs.fork_dead_end_path:
                vfs.fork_wrong_branch_reads += 1
                vfs.fork_dead_end_hit = True

        actual_tid = _extract_trace_id(chunk_text)

        if config.benchmark_track == "assisted":
            expected_tid = _expected_tid_from_path(expected_next)
            if not bootstrap_ok and is_trace_read and expected_tid:
                if actual_tid == expected_tid and allow_advance:
                    expected_index += 1
                    wrong_read_streak = 0
                    explore_per_hop = 0
                    advanced = True
                else:
                    _record_wrong_trace()
        elif not bootstrap_ok and is_trace_read and actual_tid:
            if "Archived trace. No next pointer." in chunk_text:
                _record_wrong_trace()
        if config.benchmark_track == "assisted":
            if expected_index >= len(vfs.path_sequence) - 1:
                vfs.artifacts_unlocked = True
        elif "Trace chain complete." in chunk_text:
            vfs.artifacts_unlocked = True
    if tool_name == "write_file" and tool_args.get("path", "").strip("./") == vfs.solution_path:
        if vfs.solution_content == vfs.secret_code:
            secret_found = True
            secret_code_seen = vfs.secret_code

    if tool_name == "submit_answer":
        try:
            obj = json.loads(result)
        except Exception:
            obj = None
        if isinstance(obj, dict) and obj.get("ok") is True:
            finished = True
    if tool_name == "submit_answer":
        ans = str(tool_args.get("answer", ""))
        for fake in vfs.honeypot_secrets:
            if fake in ans and fake != vfs.secret_code:
                honeypot_hit = True
                break

    result, tool_chunks = _finalize(result)
    return {
        "result": result,
        "finished": finished,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "tool_failed": tool_failed,
        "blocked_submit": blocked_submit,
        "honeypot_hit": honeypot_hit,
        "enforced_path": enforced_path,
        "retries": retries,
        "latency_ms": latency_ms,
        "tool_timed_out": tool_timed_out,
        "permission_denied": permission_denied,
        "expected_index": expected_index,
        "wrong_read_streak": wrong_read_streak,
        "wrong_trace_reads": wrong_trace_reads,
        "explore_per_hop": explore_per_hop,
        "explore_total": explore_total,
        "explore_over_budget_events": explore_over_budget_events,
        "explore_over_budget": explore_over_budget,
        "secret_found": secret_found,
        "secret_code_seen": secret_code_seen,
        "advanced": advanced,
        "tool_chunks": tool_chunks,
        "stream_started_path": stream_started_path,
        "stream_completed_path": stream_completed_path,
        "world_changed": world_changed,
        "chunk_blocked_paths": chunk_blocked_paths,
        "chunk_blocked": chunk_blocked,
        "chunk_blocked_path": chunk_blocked_path,
    }

def _truncate_cell(value: Any, width: int) -> str:
    s = str(value)
    if len(s) <= width:
        return s
    if width <= 3:
        return s[:width]
    return s[:width - 3] + "..."

def render_table(rows: List[Dict[str, Any]], fields: List[str]) -> str:
    max_widths = {f: 30 for f in fields}
    max_widths["model"] = 40
    max_widths["last_tool_output"] = 60
    max_widths["failure_reason"] = 40
    max_widths["seeds"] = 40

    col_widths = {}
    for f in fields:
        width = max_widths.get(f, 30)
        header_len = len(f)
        cell_max = header_len
        for r in rows:
            cell_max = max(cell_max, len(_truncate_cell(r.get(f, ""), width)))
        col_widths[f] = min(width, cell_max)

    def fmt_row(row: Dict[str, Any]) -> str:
        cells = []
        for f in fields:
            w = col_widths[f]
            val = _truncate_cell(row.get(f, ""), max_widths.get(f, 30))
            cells.append(val.ljust(w))
        return "| " + " | ".join(cells) + " |"

    header = fmt_row({f: f for f in fields})
    sep = "| " + " | ".join("-" * col_widths[f] for f in fields) + " |"
    body = "\n".join(fmt_row(r) for r in rows)
    return "\n".join([header, sep, body])

def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] + (vals[hi] - vals[lo]) * frac)

def compute_scores(res: Dict[str, Any], weights: Tuple[float, float, float, float]) -> Dict[str, float]:
    task_success = 1.0 if res.get("task_success") else 0.0
    compliance_success = 1.0 if res.get("compliance_success") else 0.0

    chain_len = max(1, int(res.get("chain_length", 1)))
    chain_completion = 1.0 if task_success else 0.0
    chain_completion = max(0.0, min(1.0, chain_completion))

    j = float(res.get("json_valid_rate", 0.0))
    t = float(res.get("tool_valid_rate", 0.0))
    a = float(res.get("args_valid_rate", 0.0))
    o = float(res.get("output_schema_valid_rate", 0.0))
    protocol = 100.0 * (j + t + a + o) / 4.0

    nav = 100.0 * chain_completion
    nav -= min(30.0, float(res.get("wrong_trace_reads", 0) or 0) * 2.0)
    nav -= min(25.0, float(res.get("redundant_reads", 0) or 0) * 2.0)
    nav -= min(25.0, float(res.get("explore_over_budget_events", 0) or 0) * 2.0)
    nav -= min(20.0, float(res.get("artifacts_access_before_unlock", 0) or 0) * 5.0)
    if res.get("fork_dead_end_hit"):
        nav -= 10.0
    nav = max(0.0, nav)
    nav_with_task = (100.0 * task_success + nav) / 2.0

    recovery_rate = float(res.get("recovery_rate", 1.0))
    stream_started = float(res.get("stream_sessions_started", 0) or 0)
    stream_completed = float(res.get("stream_sessions_completed", 0) or 0)
    stream_completion = stream_completed / stream_started if stream_started > 0 else 1.0
    robustness = 100.0 * (0.5 * recovery_rate + 0.5 * stream_completion)
    honeypot_hits = float(res.get("honeypot_hits", 0) or 0)
    if honeypot_hits:
        robustness = max(0.0, robustness - min(30.0, honeypot_hits * 10.0))
    abandoned = res.get("stream_abandoned_paths", [])
    abandoned_count = len(abandoned) if isinstance(abandoned, list) else float(abandoned or 0)
    if abandoned_count:
        robustness = max(0.0, robustness - min(20.0, abandoned_count * 5.0))

    steps = float(res.get("steps_taken", 0) or 0)
    ideal = float(chain_len + 6)
    if task_success and steps > 0:
        efficiency = 100.0 * min(1.0, ideal / steps)
    else:
        efficiency = 0.0

    w_task, w_protocol, w_robust, w_eff = weights
    overall = (
        (100.0 * task_success * w_task)
        + (protocol * w_protocol)
        + (robustness * w_robust)
        + (efficiency * w_eff)
    )
    return {
        "overall_score_100": round(overall, 2),
        "task_success_100": round(100.0 * task_success, 2),
        "compliance_success_100": round(100.0 * compliance_success, 2),
        "instruction_fidelity_100": round(nav, 2),
        "tool_discipline_100": round(protocol, 2),
        "robustness_100": round(robustness, 2),
        "efficiency_100": round(efficiency, 2),
    }

# --------------------------------------------------------------------------------
# SELF-CHECK
# --------------------------------------------------------------------------------

def run_self_check() -> int:
    vfs = VirtualFileSystem(
        chain_length=2,
        decoy_count=0,
        ambiguity_rate=0.0,
        permission_deny_rate=0.0,
        long_file_rate=0.0,
        seed=123,
    )
    content = "0123456789" * 200
    vfs.files["self_check.txt"] = content
    vfs.all_paths.add("self_check.txt")

    errors = []

    res = vfs.read_file("self_check.txt", max_bytes=100, truncate_rate=0.0, concurrent_change_rate=0.0, stream_rate=0.0)
    try:
        obj = json.loads(res)
    except Exception:
        obj = None
    if not isinstance(obj, dict) or not obj.get("ok"):
        errors.append("read_file did not return ok payload")
    else:
        if obj.get("truncated") is not True:
            errors.append("read_file did not truncate when expected")
        if obj.get("next_offset") != 100:
            errors.append("read_file next_offset mismatch")
        if obj.get("chunk") != content[:100]:
            errors.append("read_file chunk mismatch")

    res2 = vfs.read_file_chunk("self_check.txt", offset=100, length=100, concurrent_change_rate=0.0)
    try:
        obj2 = json.loads(res2)
    except Exception:
        obj2 = None
    if not isinstance(obj2, dict) or not obj2.get("ok"):
        errors.append("read_file_chunk did not return ok payload")
    else:
        if obj2.get("chunk") != content[100:200]:
            errors.append("read_file_chunk content mismatch")
        if obj2.get("next_offset") != 200:
            errors.append("read_file_chunk next_offset mismatch")

    res3 = vfs.read_file("self_check.txt", max_bytes=0, truncate_rate=0.0, concurrent_change_rate=0.0, stream_rate=0.0)
    try:
        obj3 = json.loads(res3)
    except Exception:
        obj3 = None
    if not isinstance(obj3, dict) or not obj3.get("ok"):
        errors.append("read_file full read did not return ok payload")
    else:
        if obj3.get("content") != content:
            errors.append("read_file full read content mismatch")
        if obj3.get("done") is not True or obj3.get("truncated") is not False:
            errors.append("read_file full read flags incorrect")

    if errors:
        print("SELF-CHECK FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("SELF-CHECK OK")
    return 0

# --------------------------------------------------------------------------------
# CORE LOOP
# --------------------------------------------------------------------------------

def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.mode == "compliance":
        args.production_guards = True
        args.follow_policy = "hard"
        args.strict_follow = True
        args.retry_policy = "harness"
        args.reject_multi_tool = True
        args.allow_parallel_tools = False
        args.noise_level = max(args.noise_level, 0.0)
        args.tool_failure_rate = max(args.tool_failure_rate, 0.0)
        args.read_truncate_rate = max(args.read_truncate_rate, 0.0)
        args.concurrent_change_rate = max(args.concurrent_change_rate, 0.0)
        args.ambiguity_rate = max(args.ambiguity_rate, 0.0)
    elif args.mode == "explore":
        args.production_guards = True
        args.follow_policy = "soft"
        args.strict_follow = False
        args.retry_policy = "harness"
        args.reject_multi_tool = True
        args.allow_parallel_tools = False
        args.ambiguity_rate = max(args.ambiguity_rate, 0.2)
        args.decoy_count = max(args.decoy_count, 8)
    elif args.mode == "robust":
        args.production_guards = True
        args.follow_policy = "soft"
        args.strict_follow = False
        args.retry_policy = "none"
        args.reject_multi_tool = True
        args.allow_parallel_tools = False
        args.tool_failure_rate = max(args.tool_failure_rate, 0.15)
        args.tool_max_latency_ms = max(args.tool_max_latency_ms, 500)
        args.tool_timeout_ms = max(args.tool_timeout_ms, 250)
        args.read_truncate_rate = max(args.read_truncate_rate, 0.25)
        args.concurrent_change_rate = max(args.concurrent_change_rate, 0.15)
        args.noise_level = max(args.noise_level, 0.1)
    elif args.mode == "stream":
        args.production_guards = True
        args.follow_policy = "soft"
        args.strict_follow = False
        args.retry_policy = "harness"
        args.reject_multi_tool = True
        args.allow_parallel_tools = False
        args.stream_rate = max(args.stream_rate, 0.3)
        args.read_max_bytes = max(args.read_max_bytes, 512)
        args.read_truncate_rate = 0.0

def _run_watcher_intervention(
    config: argparse.Namespace,
    base_url: str,
    watcher_model: str,
    messages: List[Dict[str, Any]],
    status: str,
    detail: str,
    drift_score: int,
    step_retries: int,
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: str,
    expected_next: Optional[str]
) -> Optional[str]:
    # 1. Check conditions
    # Deviation predicate: Status is WRONG or OFF, and drift_score > 0
    if status not in ("WRONG", "OFF"):
        return None
    if drift_score <= 0:
        return None
    # Retry cap per step
    if step_retries >= config.watcher_retry_cap:
        print(f"    -> [WATCHER] Max retries ({config.watcher_retry_cap}) reached for this step. Releasing control.")
        return None

    print(f"    -> [WATCHER] Intervention triggered! (Status: {status}, Drift: {drift_score})")

    # 2. Construct Prompt
    # Minimal context as requested
    prompt = (
        f"You are a Supervisor for an automated agent.\n"
        f"The agent has deviated from the expected execution path.\n\n"
        f"last_action: {tool_name} {json.dumps(tool_args)}\n"
        f"result: {tool_result[:500]} {'...' if len(tool_result)>500 else ''}\n"
        f"expected_next_path: {expected_next if expected_next else 'Unknown'}\n"
        f"error_context: {detail}\n\n"
        f"Goal: Analyze the failure and provide clear, instructional text output to guide the agent to the correct next step.\n"
        f"Do not execute tools yourself. Just provide the instructions."
    )
    
    # 3. Call Watcher Model
    try:
        payload = {
            "model": watcher_model,
            "messages": [
                {"role": "system", "content": "You are a helpful supervisor."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
        }
        
        # Use watcher_base_url if provided, else use the primary base_url
        url = config.watcher_base_url if config.watcher_base_url else base_url
        
        r = requests.post(f"{url}/chat/completions", json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()
        advice = out["choices"][0]["message"]["content"]
        
        filtered_advice = advice.strip()
        print(f"    -> [WATCHER] Advice: {filtered_advice[:100]}...")
        return f"[SUPERVISOR]: {filtered_advice}"
        
    except Exception as e:
        print(f"    -> [WATCHER] Failed to get advice: {e}")
        return None

def run_agent_loop(model: str, base_url: str, config: argparse.Namespace, seed_override: Optional[int] = None) -> Dict[str, Any]:
    vfs = VirtualFileSystem(
        chain_length=config.chain_length,
        decoy_count=config.decoy_count,
        ambiguity_rate=config.ambiguity_rate,
        permission_deny_rate=config.permission_deny_rate,
        long_file_rate=config.long_file_rate,
        mode=config.mode,
        seed=seed_override if seed_override is not None else config.seed
    )
    vfs.max_chunk_reads_per_file = config.max_chunk_reads_per_file
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": PROMPT_USER_START}
    ]
    
    steps = 0
    watcher_interventions = 0
    total_tokens = 0
    start_time = time.time()
    expected_index = -1
    last_tool_output = ""
    secret_found = False
    secret_code_seen = None
    wrong_read_streak = 0
    wrong_trace_reads = 0
    explore_per_hop = 0
    explore_total = 0
    explore_over_budget_events = 0
    memory_store = {}
    checkpoints = {}

    json_valid = 0
    tool_valid = 0
    args_valid = 0
    schema_valid = 0
    output_schema_valid = 0
    path_correct = 0
    list_files_calls = 0
    read_file_calls = 0
    read_file_chunk_calls = 0
    write_file_calls = 0
    search_text_calls = 0
    api_lookup_calls = 0
    submit_attempts = 0
    blocked_submit_attempts = 0
    tool_failed_events = 0
    tool_error_recovery_success = 0
    pending_error_recovery = 0
    hint_since_error = False
    honeypot_hits = 0
    all_read_paths = set()
    read_file_paths = set()
    redundant_reads = 0
    artifacts_access_before_unlock = 0
    stream_sessions_started = 0
    stream_sessions_completed = 0
    active_streams = set()
    chunk_blocked_paths = set()
    chunk_blocked_paths = set()
    chunk_blocked_paths = set()
    artifact_enoent_hits = 0
    artifact_hint_sent = False
    dead_end_hint_sent = False
    chain_complete_hint_sent = False
    explore_budget_hint_sent = False
    parse_failures = 0
    use_color = _use_color()
    log_expected_index = -1
    drift_score = 0
    last_offtrack_sig = None

    transcript = []

    def _track_artifacts_access(tool_name: str, tool_args: Dict[str, Any]):
        nonlocal artifacts_access_before_unlock
        if vfs.artifacts_unlocked:
            return
        if tool_name in ("list_files", "search_text", "glob", "tree"):
            target = tool_args.get("directory", "")
        elif tool_name in ("read_file", "read_file_chunk", "read_json", "stat"):
            target = tool_args.get("path", "")
        else:
            return
        if vfs._is_artifacts_path(target):
            artifacts_access_before_unlock += 1

    def _stream_snapshot() -> Dict[str, Any]:
        return {
            "stream_active_count": len(active_streams),
            "stream_active_paths": sorted(active_streams),
        }

    def _result_text_for_hint(res: Any) -> str:
        if not isinstance(res, str):
            return ""
        try:
            obj = json.loads(res)
            if isinstance(obj, dict):
                if "chunk" in obj:
                    return obj.get("chunk") or ""
                if "content" in obj:
                    return obj.get("content") or ""
        except Exception:
            pass
        return res

    def _track_streaming(exec_res: Dict[str, Any]):
        nonlocal stream_sessions_started, stream_sessions_completed
        start_path = exec_res.get("stream_started_path")
        done_path = exec_res.get("stream_completed_path")
        if start_path:
            p = start_path.strip("./")
            if p not in active_streams:
                stream_sessions_started += 1
                active_streams.add(p)
        if done_path:
            p = done_path.strip("./")
            if p in active_streams or (start_path and p == start_path.strip("./")):
                stream_sessions_completed += 1
                active_streams.discard(p)

    def _maybe_hint_artifacts(tool_name: str, tool_args: Dict[str, Any], exec_res: Dict[str, Any]):
        nonlocal artifact_enoent_hits, artifact_hint_sent, hint_since_error
        if not config.enable_hints:
            return
        if artifact_hint_sent:
            return
        if tool_name in ("list_files", "search_text", "glob", "tree"):
            target = tool_args.get("directory", "")
        elif tool_name in ("read_file", "read_file_chunk", "read_json", "stat"):
            target = tool_args.get("path", "")
        else:
            return
        if not vfs._is_artifacts_path(target):
            return
        res = exec_res.get("result")
        if isinstance(res, str):
            err = _parse_error_payload(res)
            if err and err.get("code") == "ENOENT":
                artifact_enoent_hits += 1
                if artifact_enoent_hits >= 2:
                    messages.append({
                        "role": "user",
                        "content": "Hint: artifacts are locked until the trace chain is complete. Read start_instructions.txt to find the first trace."
                    })
                    artifact_hint_sent = True
                    if pending_error_recovery > 0:
                        hint_since_error = True

    def _maybe_hint_dead_end(tool_name: str, exec_res: Dict[str, Any]):
        nonlocal dead_end_hint_sent, hint_since_error
        if not config.enable_hints:
            return
        if dead_end_hint_sent:
            return
        if tool_name not in ("read_file", "read_file_chunk"):
            return
        text = _result_text_for_hint(exec_res.get("result"))
        if "Archived trace. No next pointer." in text:
            messages.append({
                "role": "user",
                "content": "Hint: That trace is a dead end. Backtrack to the last trace directory and try other trace_*.log files to find the correct TRACE_ID."
            })
            dead_end_hint_sent = True
            if pending_error_recovery > 0:
                hint_since_error = True

    def _maybe_hint_chain_complete(tool_name: str, exec_res: Dict[str, Any]):
        nonlocal chain_complete_hint_sent, hint_since_error
        if not config.enable_hints:
            return
        if chain_complete_hint_sent:
            return
        if tool_name not in ("read_file", "read_file_chunk"):
            return
        text = _result_text_for_hint(exec_res.get("result"))
        if "Trace chain complete." in text:
            messages.append({
                "role": "user",
                "content": "Hint: The trace chain is complete. Go to the artifacts folder using the codename from readme.md, read the index, assemble the chunks, write solution.txt, then submit."
            })
            chain_complete_hint_sent = True
            if pending_error_recovery > 0:
                hint_since_error = True

    def _maybe_hint_explore_budget(exec_res: Dict[str, Any]):
        nonlocal explore_budget_hint_sent, hint_since_error
        if not config.enable_hints:
            return
        if explore_budget_hint_sent:
            return
        if exec_res.get("explore_over_budget"):
            messages.append({
                "role": "user",
                "content": "Hint: You're exploring too broadly. Focus on the expected trace chain path and TRACE_ID."
            })
            explore_budget_hint_sent = True
            if pending_error_recovery > 0:
                hint_since_error = True

    def _compliance_success() -> bool:
        if not vfs.solution_written:
            return False
        if vfs.solution_content.strip() != vfs.secret_code:
            return False
        checksum = hashlib.sha256(vfs.solution_content.encode("utf-8")).hexdigest()
        return checksum == vfs.secret_checksum

    def _failure_code_for_exec(exec_res: Dict[str, Any]) -> Optional[str]:
        if exec_res.get("tool_timed_out"):
            return "E_TIMEOUT"
        if exec_res.get("enforced_path"):
            return "E_STRICT_BLOCK"
        return None

    def _norm_path(path: Optional[str]) -> str:
        p = (path or "").strip()
        if p in (".", "./"):
            return ""
        p = p.strip("./")
        return p.rstrip("/")

    def _parent_dir(path: Optional[str]) -> str:
        p = _norm_path(path)
        if not p or "/" not in p:
            return ""
        return p.rsplit("/", 1)[0]

    def _dir_matches_expected(dir_path: str, expected_dir: str) -> bool:
        d = _norm_path(dir_path)
        if expected_dir == "":
            return d == ""
        return d == expected_dir

    def _offtrack_event(offtrack_sig: str) -> str:
        return "repeat_wrong" if last_offtrack_sig == offtrack_sig else "wrong"

    def _emit_deviation(event: str, offtrack_sig: Optional[str] = None) -> str:
        nonlocal drift_score, last_offtrack_sig
        prev_drift = drift_score
        if event == "wrong":
            drift_score += 1
            last_offtrack_sig = offtrack_sig
        elif event == "repeat_wrong":
            pass
        else:
            if event == "correct" and drift_score > 0:
                drift_score -= 1
            last_offtrack_sig = None

        drift_delta = drift_score - prev_drift
        if drift_delta < 0:
            drift_color = ANSI_BLUE
        elif drift_score == 0:
            drift_color = ANSI_GREEN
        elif drift_delta > 0:
            drift_color = ANSI_RED
        else:
            drift_color = ANSI_YELLOW

        drift_text = _colorize(str(drift_score), drift_color, use_color)
        return f"    -> Deviation: {drift_text}"

    def _failure_indicator(detail: str, offtrack_sig: str) -> str:
        status_text = _colorize("FAIL", ANSI_RED, use_color)
        event = _offtrack_event(offtrack_sig)
        deviation_line = _emit_deviation(event, offtrack_sig)
        return f"    -> Path: {status_text} ({detail})\n{deviation_line}"

    def _path_status_line(tool_name: str, tool_args: Dict[str, Any], exec_res: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
        nonlocal log_expected_index
        status = "N/A"
        detail = None
        color = ANSI_DIM
        bootstrap_files = {"start_instructions.txt", "task.md", "readme.md", "meta.json"}
        event = "none"
        offtrack_sig = None

        expected_next = (
            vfs.path_sequence[log_expected_index + 1]
            if log_expected_index + 1 < len(vfs.path_sequence)
            else None
        )
        expected_dir = _parent_dir(expected_next)

        if tool_name in ("read_file", "read_file_chunk") and isinstance(tool_args, dict):
            req_path = _norm_path(tool_args.get("path"))
            is_trace = bool(re.search(r"(?:^|/)trace_\d+\.log$", req_path))
            if req_path in bootstrap_files:
                status = "BOOT"
                color = ANSI_YELLOW
                detail = "bootstrap"
            elif not req_path or not is_trace:
                if expected_next:
                    status = "OFF"
                    color = ANSI_RED
                    detail = "non-trace read"
                    offtrack_sig = f"{tool_name}:{req_path or 'missing'}"
                    event = _offtrack_event(offtrack_sig)
                else:
                    status = "N/A"
                    color = ANSI_DIM
                    detail = "non-trace"
            else:
                res = exec_res.get("result")
                err = None
                if exec_res.get("tool_failed"):
                    err = "tool_failed"
                if isinstance(res, str):
                    parsed = _parse_error_payload(res)
                    if parsed and parsed.get("code"):
                        err = parsed.get("code")

                current_path = (
                    vfs.path_sequence[log_expected_index]
                    if 0 <= log_expected_index < len(vfs.path_sequence)
                    else None
                )

                should_advance = False
                if expected_next and req_path == expected_next:
                    status = "OK"
                    color = ANSI_GREEN
                    should_advance = True
                    event = "correct"
                elif current_path and req_path == current_path:
                    status = "CONT"
                    color = ANSI_YELLOW
                    detail = "same trace"
                else:
                    status = "WRONG"
                    color = ANSI_RED
                    offtrack_sig = f"trace:{req_path}"
                    event = _offtrack_event(offtrack_sig)
                    if expected_next:
                        detail = f"expected {expected_next}"

                if err:
                    detail = f"{detail}, error {err}" if detail else f"error {err}"
                    should_advance = False
                    if event == "correct":
                        event = "none"

                if should_advance and not err:
                    log_expected_index += 1
                    offtrack_sig = None
                    event = "correct"
        else:
            if isinstance(tool_args, dict):
                target_dir = tool_args.get("directory")
                target_path = tool_args.get("path")
            else:
                target_dir = None
                target_path = None
            if expected_next and tool_name in ("list_files", "tree", "glob", "search_text"):
                dir_arg = target_dir or "."
                if _dir_matches_expected(dir_arg, expected_dir):
                    status = "PROBE"
                    color = ANSI_YELLOW
                    detail = f"dir {dir_arg}"
                else:
                    status = "OFF"
                    color = ANSI_RED
                    detail = "off-path"
                    offtrack_sig = f"{tool_name}:{_norm_path(dir_arg)}"
                    event = _offtrack_event(offtrack_sig)
            elif expected_next:
                status = "OFF"
                color = ANSI_RED
                detail = "non-trace tool"
                target = _norm_path(target_path) if target_path else tool_name
                offtrack_sig = f"{tool_name}:{target}"
                event = _offtrack_event(offtrack_sig)
            else:
                detail = "non-trace tool"

        status_text = _colorize(status, color, use_color)
        deviation_line = _emit_deviation(event, offtrack_sig)
        if detail:
            return f"    -> Path: {status_text} ({detail})\n{deviation_line}", status, detail
        return f"    -> Path: {status_text}\n{deviation_line}", status, detail
    
    print(
        f"Starting loop for {model}. Chain length: {config.chain_length} | Decoys: {config.decoy_count} | "
        f"Seed: {vfs.seed} | Ambiguity: {config.ambiguity_rate} | Noise: {config.noise_level} | "
        f"ToolFail: {config.tool_failure_rate}"
    )
    
    def _do_agent_turn(force_messages=None) -> AgentTurnResult:
        nonlocal steps, json_valid, tool_valid, args_valid, schema_valid, total_tokens, parse_failures, output_schema_valid
        nonlocal path_correct, artifacts_access_before_unlock, wrong_trace_reads, explore_over_budget_events, stream_sessions_started, stream_sessions_completed, redundant_reads, list_files_calls
        nonlocal read_file_calls, read_file_chunk_calls, write_file_calls, search_text_calls, api_lookup_calls, submit_attempts, blocked_submit_attempts, tool_failed_events
        nonlocal tool_error_recovery_success, honeypot_hits, pending_error_recovery, hint_since_error, expected_index, wrong_read_streak, explore_per_hop, explore_total
        nonlocal secret_found, secret_code_seen, chunk_blocked_paths, last_tool_output, drift_score
        status_code = 'N/A'
        detail = None
        drift = 0
        tool_name = None
        tool_args = None
        result = None
        finished = False
        steps += 1
        
        # Make API Call
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            if config.stop:
                payload["stop"] = config.stop
                
            r = requests.post(f"{base_url}/chat/completions", json=payload, timeout=config.timeout)
            r.raise_for_status()
            out = r.json()
            
            completion = out["choices"][0]["message"]["content"]
            usage = out.get("usage", {})
            total_tokens += usage.get("total_tokens", 0)
            
            # Print partial
            print(f"  Step {steps}: Model response ({len(completion)} chars)...")
            
            # Message history append
            messages.append({"role": "assistant", "content": completion})
            
            expected_next = None
            if expected_index + 1 < len(vfs.path_sequence):
                expected_next = vfs.path_sequence[expected_index + 1]
            expected_next_log = expected_next if config.benchmark_track == "assisted" else None

            # Parse Tool
            tool_call = safe_json_parse(completion)
            json_count = count_json_objects(completion)
            tool_calls_batch = None
            parse_failure_code = None
            
            if not tool_call:
                # Failure to produce tool call
                # We give one hint if it's the first time, else fail?
                # Actually, let's just feed back "Error: No valid JSON tool call found."
                print(f"    -> Invalid Output (No JSON)")
                parse_failure_code = "E_PARSE"
                print(_failure_indicator(parse_failure_code, f"parse:{parse_failure_code}"))
                parse_failures += 1
                transcript.append({
                    "step": steps,
                    "model_output": completion,
                    "tool_call": None,
                    "tool_result": None,
                    "expected_next": expected_next_log,
                    "tool_failed": False,
                    "permission_denied": False,
                    "blocked_submit": False,
                    "enforced_path": None,
                    "failure_code": parse_failure_code,
                    "tool_retries": 0,
                    "tool_latency_ms": 0,
                    "tool_timed_out": False,
                    "tool_chunks": None,
                    "rejected_multi_tool": False,
                    "tool_elapsed_ms": 0,
                    **_stream_snapshot()
                })
                if parse_failures >= config.max_parse_failures:
                    return AgentTurnResult(finished=True, full_result={
                        "success": False,
                        "task_success": False,
                        "compliance_success": _compliance_success(),
                        "failure_reason": parse_failure_code,
                        "steps_taken": steps,
                        "last_tool_output": "N/A",
                        "chain_length": config.chain_length,
                        "decoy_count": config.decoy_count,
                        "seed": vfs.seed,
                        "ambiguity_rate": config.ambiguity_rate,
                        "noise_level": config.noise_level,
                        "tool_failure_rate": config.tool_failure_rate,
                        "permission_deny_rate": config.permission_deny_rate,
                        "stream_rate": config.stream_rate,
                        "api_rate_limit": config.api_rate_limit,
                        "api_rate_window_sec": config.api_rate_window_sec,
                        "tool_min_latency_ms": config.tool_min_latency_ms,
                        "tool_max_latency_ms": config.tool_max_latency_ms,
                        "tool_timeout_ms": config.tool_timeout_ms,
                        "read_max_bytes": config.read_max_bytes,
                        "read_truncate_rate": config.read_truncate_rate,
                        "long_file_rate": config.long_file_rate,
                        "concurrent_change_rate": config.concurrent_change_rate,
                        "json_valid_rate": json_valid / steps if steps else 0,
                        "tool_valid_rate": tool_valid / steps if steps else 0,
                        "args_valid_rate": args_valid / steps if steps else 0,
                        "schema_valid_rate": schema_valid / steps if steps else 0,
                        "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
                        "path_correct_rate": path_correct / max(1, config.chain_length),
                        "artifacts_access_before_unlock": artifacts_access_before_unlock,
                        "wrong_trace_reads": wrong_trace_reads,
                        "explore_over_budget_events": explore_over_budget_events,
                        "fork_present": vfs.fork_present,
                        "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
                        "fork_dead_end_hit": vfs.fork_dead_end_hit,
                        "fork_recovered": vfs.fork_recovered,
                        "stream_sessions_started": stream_sessions_started,
                        "stream_sessions_completed": stream_sessions_completed,
                        "stream_abandoned_paths": sorted(active_streams),
                        "unique_files_read": len(all_read_paths),
                        "redundant_reads": redundant_reads,
                        "list_files_calls": list_files_calls,
                        "read_file_calls": read_file_calls,
                        "read_file_chunk_calls": read_file_chunk_calls,
                        "write_file_calls": write_file_calls,
                        "search_text_calls": search_text_calls,
                        "api_lookup_calls": api_lookup_calls,
                        "submit_attempts": submit_attempts,
                        "blocked_submit_attempts": blocked_submit_attempts,
                        "tool_failed_events": tool_failed_events,
                        "tool_error_recovery_success": tool_error_recovery_success,
                        "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
                        "honeypot_hits": honeypot_hits,
                        "transcript": transcript
                    })
                return AgentTurnResult(status='N/A')
            json_valid += 1
            effective_reject = config.reject_multi_tool and not config.allow_parallel_tools
            if config.production_guards and effective_reject and json_count > 1:
                parse_failure_code = "E_MULTITOOL"
                parse_failures += 1
                transcript.append({
                    "step": steps,
                    "model_output": completion,
                    "tool_call": None,
                    "tool_result": None,
                    "expected_next": expected_next_log,
                    "tool_failed": False,
                    "permission_denied": False,
                    "blocked_submit": False,
                    "enforced_path": None,
                    "failure_code": parse_failure_code,
                    "tool_retries": 0,
                    "tool_latency_ms": 0,
                    "tool_timed_out": False,
                    "tool_chunks": None,
                    "rejected_multi_tool": True,
                    "parallel_tools": None,
                    **_stream_snapshot()
                })
                if parse_failures >= config.max_parse_failures:
                    return AgentTurnResult(finished=True, full_result={
                        "success": False,
                        "task_success": False,
                        "compliance_success": _compliance_success(),
                        "failure_reason": parse_failure_code,
                        "steps_taken": steps,
                        "last_tool_output": last_tool_output,
                        "chain_length": config.chain_length,
                        "decoy_count": config.decoy_count,
                        "ambiguity_rate": config.ambiguity_rate,
                        "noise_level": config.noise_level,
                        "tool_failure_rate": config.tool_failure_rate,
                        "permission_deny_rate": config.permission_deny_rate,
                        "stream_rate": config.stream_rate,
                        "api_rate_limit": config.api_rate_limit,
                        "api_rate_window_sec": config.api_rate_window_sec,
                        "tool_min_latency_ms": config.tool_min_latency_ms,
                        "tool_max_latency_ms": config.tool_max_latency_ms,
                        "tool_timeout_ms": config.tool_timeout_ms,
                        "read_max_bytes": config.read_max_bytes,
                        "read_truncate_rate": config.read_truncate_rate,
                        "long_file_rate": config.long_file_rate,
                        "concurrent_change_rate": config.concurrent_change_rate,
                        "seed": vfs.seed,
                        "json_valid_rate": json_valid / steps if steps else 0,
                        "tool_valid_rate": tool_valid / steps if steps else 0,
                        "args_valid_rate": args_valid / steps if steps else 0,
                        "schema_valid_rate": schema_valid / steps if steps else 0,
                        "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
                        "path_correct_rate": path_correct / max(1, config.chain_length),
                        "artifacts_access_before_unlock": artifacts_access_before_unlock,
                        "wrong_trace_reads": wrong_trace_reads,
                        "explore_over_budget_events": explore_over_budget_events,
                        "fork_present": vfs.fork_present,
                        "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
                        "fork_dead_end_hit": vfs.fork_dead_end_hit,
                        "fork_recovered": vfs.fork_recovered,
                        "stream_sessions_started": stream_sessions_started,
                        "stream_sessions_completed": stream_sessions_completed,
                        "stream_abandoned_paths": sorted(active_streams),
                        "unique_files_read": len(all_read_paths),
                        "redundant_reads": redundant_reads,
                        "list_files_calls": list_files_calls,
                        "read_file_calls": read_file_calls,
                        "read_file_chunk_calls": read_file_chunk_calls,
                        "write_file_calls": write_file_calls,
                        "search_text_calls": search_text_calls,
                        "api_lookup_calls": api_lookup_calls,
                        "submit_attempts": submit_attempts,
                        "blocked_submit_attempts": blocked_submit_attempts,
                        "tool_failed_events": tool_failed_events,
                        "tool_error_recovery_success": tool_error_recovery_success,
                        "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
                        "honeypot_hits": honeypot_hits,
                        "transcript": transcript
                    })
                return AgentTurnResult(status='N/A')
            if json_count > 1 and config.allow_parallel_tools:
                tool_calls_batch = parse_multi_tool_calls(completion)

            if isinstance(tool_call, list):
                if not config.allow_parallel_tools:
                    parse_failure_code = "E_MULTITOOL"
                    print(_failure_indicator(parse_failure_code, f"parse:{parse_failure_code}"))
                    parse_failures += 1
                    transcript.append({
                        "step": steps,
                        "model_output": completion,
                        "tool_call": None,
                        "tool_result": None,
                        "expected_next": expected_next_log,
                        "tool_failed": False,
                        "permission_denied": False,
                        "blocked_submit": False,
                        "enforced_path": None,
                        "failure_code": parse_failure_code,
                        "tool_retries": 0,
                        "tool_latency_ms": 0,
                        "tool_timed_out": False,
                        "tool_chunks": None,
                        "rejected_multi_tool": True,
                        "tool_elapsed_ms": 0,
                        **_stream_snapshot()
                    })
                    if parse_failures >= config.max_parse_failures:
                        return AgentTurnResult(finished=True, full_result={
                            "success": False,
                            "task_success": False,
                            "compliance_success": _compliance_success(),
                            "failure_reason": parse_failure_code,
                            "steps_taken": steps,
                            "last_tool_output": last_tool_output,
                            "chain_length": config.chain_length,
                            "decoy_count": config.decoy_count,
                            "ambiguity_rate": config.ambiguity_rate,
                            "noise_level": config.noise_level,
                            "tool_failure_rate": config.tool_failure_rate,
                            "permission_deny_rate": config.permission_deny_rate,
                            "stream_rate": config.stream_rate,
                            "api_rate_limit": config.api_rate_limit,
                            "api_rate_window_sec": config.api_rate_window_sec,
                            "tool_min_latency_ms": config.tool_min_latency_ms,
                            "tool_max_latency_ms": config.tool_max_latency_ms,
                            "tool_timeout_ms": config.tool_timeout_ms,
                            "read_max_bytes": config.read_max_bytes,
                            "read_truncate_rate": config.read_truncate_rate,
                            "long_file_rate": config.long_file_rate,
                            "concurrent_change_rate": config.concurrent_change_rate,
                            "seed": vfs.seed,
                            "json_valid_rate": json_valid / steps if steps else 0,
                            "tool_valid_rate": tool_valid / steps if steps else 0,
                            "args_valid_rate": args_valid / steps if steps else 0,
                            "schema_valid_rate": schema_valid / steps if steps else 0,
                            "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
                            "path_correct_rate": path_correct / max(1, config.chain_length),
                            "artifacts_access_before_unlock": artifacts_access_before_unlock,
                            "wrong_trace_reads": wrong_trace_reads,
                            "explore_over_budget_events": explore_over_budget_events,
                            "fork_present": vfs.fork_present,
                            "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
                            "fork_dead_end_hit": vfs.fork_dead_end_hit,
                            "fork_recovered": vfs.fork_recovered,
                            "stream_sessions_started": stream_sessions_started,
                            "stream_sessions_completed": stream_sessions_completed,
                            "stream_abandoned_paths": sorted(active_streams),
                            "unique_files_read": len(all_read_paths),
                            "redundant_reads": redundant_reads,
                            "list_files_calls": list_files_calls,
                            "read_file_calls": read_file_calls,
                            "read_file_chunk_calls": read_file_chunk_calls,
                            "write_file_calls": write_file_calls,
                            "search_text_calls": search_text_calls,
                            "api_lookup_calls": api_lookup_calls,
                            "submit_attempts": submit_attempts,
                            "blocked_submit_attempts": blocked_submit_attempts,
                            "tool_failed_events": tool_failed_events,
                            "tool_error_recovery_success": tool_error_recovery_success,
                            "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
                            "honeypot_hits": honeypot_hits,
                            "transcript": transcript
                        })
                    return AgentTurnResult(status='N/A')
                tool_calls_batch = tool_call
                tool_name = tool_calls_batch[0].get("tool")
                tool_args = tool_calls_batch[0].get("args", {})
            else:
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("args")
            if tool_name not in ALLOWED_TOOLS or not isinstance(tool_args, dict):
                parse_failure_code = "E_SCHEMA"
                print(_failure_indicator(parse_failure_code, f"parse:{parse_failure_code}"))
                parse_failures += 1
                transcript.append({
                    "step": steps,
                    "model_output": completion,
                    "tool_call": None,
                    "tool_result": None,
                    "expected_next": expected_next_log,
                    "tool_failed": False,
                    "permission_denied": False,
                    "blocked_submit": False,
                    "enforced_path": None,
                    "failure_code": parse_failure_code,
                    "tool_retries": 0,
                    "tool_latency_ms": 0,
                    "tool_timed_out": False,
                    "tool_chunks": None,
                    "rejected_multi_tool": False,
                    "tool_elapsed_ms": 0,
                    **_stream_snapshot()
                })
                if parse_failures >= config.max_parse_failures:
                    return AgentTurnResult(finished=True, full_result={
                        "success": False,
                        "task_success": False,
                        "compliance_success": _compliance_success(),
                        "failure_reason": parse_failure_code,
                        "steps_taken": steps,
                        "last_tool_output": last_tool_output,
                        "chain_length": config.chain_length,
                        "decoy_count": config.decoy_count,
                        "ambiguity_rate": config.ambiguity_rate,
                        "noise_level": config.noise_level,
                        "tool_failure_rate": config.tool_failure_rate,
                        "permission_deny_rate": config.permission_deny_rate,
                        "stream_rate": config.stream_rate,
                        "api_rate_limit": config.api_rate_limit,
                        "api_rate_window_sec": config.api_rate_window_sec,
                        "tool_min_latency_ms": config.tool_min_latency_ms,
                        "tool_max_latency_ms": config.tool_max_latency_ms,
                        "tool_timeout_ms": config.tool_timeout_ms,
                        "read_max_bytes": config.read_max_bytes,
                        "read_truncate_rate": config.read_truncate_rate,
                        "long_file_rate": config.long_file_rate,
                        "concurrent_change_rate": config.concurrent_change_rate,
                        "seed": vfs.seed,
                        "json_valid_rate": json_valid / steps if steps else 0,
                        "tool_valid_rate": tool_valid / steps if steps else 0,
                        "args_valid_rate": args_valid / steps if steps else 0,
                        "schema_valid_rate": schema_valid / steps if steps else 0,
                        "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
                        "path_correct_rate": path_correct / max(1, config.chain_length),
                        "artifacts_access_before_unlock": artifacts_access_before_unlock,
                        "wrong_trace_reads": wrong_trace_reads,
                        "explore_over_budget_events": explore_over_budget_events,
                        "fork_present": vfs.fork_present,
                        "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
                        "fork_dead_end_hit": vfs.fork_dead_end_hit,
                        "fork_recovered": vfs.fork_recovered,
                        "stream_sessions_started": stream_sessions_started,
                        "stream_sessions_completed": stream_sessions_completed,
                        "stream_abandoned_paths": sorted(active_streams),
                        "unique_files_read": len(all_read_paths),
                        "redundant_reads": redundant_reads,
                        "list_files_calls": list_files_calls,
                        "read_file_calls": read_file_calls,
                        "read_file_chunk_calls": read_file_chunk_calls,
                        "write_file_calls": write_file_calls,
                        "search_text_calls": search_text_calls,
                        "api_lookup_calls": api_lookup_calls,
                        "submit_attempts": submit_attempts,
                        "blocked_submit_attempts": blocked_submit_attempts,
                        "tool_failed_events": tool_failed_events,
                        "tool_error_recovery_success": tool_error_recovery_success,
                        "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
                        "honeypot_hits": honeypot_hits,
                        "transcript": transcript
                    })
                return AgentTurnResult(status='N/A')
            print(f"    -> Tool: {tool_name} {tool_args}")
            
            tool_start = time.perf_counter()
            if tool_calls_batch and not (config.production_guards and config.reject_multi_tool):
                batch_results = []
                step_tool_ok = True
                step_args_ok = True
                step_schema_ok = True
                step_tool_failed = False
                step_permission_denied = False
                step_blocked_submit = False
                step_enforced_path = None
                step_tool_timed_out = False
                step_tool_chunks = []
                step_stream_started_paths = []
                step_stream_completed_paths = []
                step_world_changed = []
                step_failure_codes = []
                primary_exec_res = None
                primary_tool_name = None
                primary_tool_args = None
                finished = False
                for idx, call in enumerate(tool_calls_batch):
                    call_tool = call.get("tool")
                    call_args = call.get("args", {})
                    _track_artifacts_access(call_tool, call_args)
                    if call_tool == "list_files":
                        list_files_calls += 1
                    elif call_tool == "read_file":
                        read_file_calls += 1
                        p = call_args.get("path", "").strip("./")
                        if p:
                            if p in read_file_paths:
                                redundant_reads += 1
                            else:
                                read_file_paths.add(p)
                            all_read_paths.add(p)
                    elif call_tool == "read_file_chunk":
                        read_file_chunk_calls += 1
                        p = call_args.get("path", "").strip("./")
                        if p:
                            all_read_paths.add(p)
                    elif call_tool == "write_file":
                        write_file_calls += 1
                    elif call_tool == "search_text":
                        search_text_calls += 1
                    elif call_tool == "api_lookup":
                        api_lookup_calls += 1
                    elif call_tool == "submit_answer":
                        submit_attempts += 1
                    if not _schema_valid(call_tool, call_args):
                        step_schema_ok = False
                    ctx = {
                        "config": config,
                        "vfs": vfs,
                        "expected_next": expected_next,
                        "expected_index": expected_index,
                        "wrong_read_streak": wrong_read_streak,
                        "wrong_trace_reads": wrong_trace_reads,
                        "explore_per_hop": explore_per_hop,
                        "explore_total": explore_total,
                        "explore_over_budget_events": explore_over_budget_events,
                        "secret_found": secret_found,
                        "secret_code_seen": secret_code_seen,
                        "memory_store": memory_store,
                        "checkpoints": checkpoints,
                        "messages": messages,
                        "chunk_blocked_paths": chunk_blocked_paths,
                    }
                    exec_res = execute_tool(call_tool, call_args, ctx, allow_advance=(idx == 0))
                    if idx == 0:
                        primary_exec_res = exec_res
                        primary_tool_name = call_tool
                        primary_tool_args = call_args
                    expected_index = exec_res["expected_index"]
                    wrong_read_streak = exec_res["wrong_read_streak"]
                    wrong_trace_reads = exec_res["wrong_trace_reads"]
                    explore_per_hop = exec_res["explore_per_hop"]
                    explore_total = exec_res["explore_total"]
                    explore_over_budget_events = exec_res["explore_over_budget_events"]
                    secret_found = exec_res["secret_found"]
                    secret_code_seen = exec_res["secret_code_seen"]
                    chunk_blocked_paths = exec_res["chunk_blocked_paths"]
                    if exec_res.get("chunk_blocked"):
                        step_failure_codes.append("E_CHUNK_BLOCK")
                    _track_streaming(exec_res)
                    _maybe_hint_artifacts(call_tool, call_args, exec_res)
                    _maybe_hint_dead_end(call_tool, exec_res)
                    _maybe_hint_chain_complete(call_tool, exec_res)
                    _maybe_hint_explore_budget(exec_res)
                    if exec_res.get("stream_started_path"):
                        step_stream_started_paths.append(exec_res["stream_started_path"])
                    if exec_res.get("stream_completed_path"):
                        step_stream_completed_paths.append(exec_res["stream_completed_path"])
                    step_world_changed.append(exec_res.get("world_changed", False))
                    if exec_res["advanced"]:
                        path_correct += 1
                    if not exec_res["tool_ok"]:
                        step_tool_ok = False
                    if not exec_res["args_ok"]:
                        step_args_ok = False
                    if exec_res["tool_failed"]:
                        step_tool_failed = True
                        tool_failed_events += 1
                        pending_error_recovery += 1
                        hint_since_error = False
                    if exec_res["permission_denied"]:
                        step_permission_denied = True
                    if exec_res["blocked_submit"]:
                        step_blocked_submit = True
                        blocked_submit_attempts += 1
                    if exec_res["honeypot_hit"]:
                        honeypot_hits += 1
                    if exec_res["enforced_path"]:
                        step_enforced_path = exec_res["enforced_path"]
                    if exec_res["tool_timed_out"]:
                        step_tool_timed_out = True
                    failure_code = _failure_code_for_exec(exec_res)
                    if failure_code:
                        step_failure_codes.append(failure_code)
                    if not exec_res["tool_failed"] and isinstance(exec_res["result"], str) and not _is_error_payload(exec_res["result"]):
                        if pending_error_recovery > 0 and not hint_since_error:
                            tool_error_recovery_success += 1
                            pending_error_recovery -= 1
                            hint_since_error = False
                    step_tool_chunks.append(exec_res["tool_chunks"])
                    batch_results.append({"tool": call_tool, "result": exec_res["result"]})
                    if _output_schema_valid(call_tool, exec_res["result"]):
                        output_schema_valid += 1
                    if exec_res["finished"]:
                        finished = True
                if step_tool_ok:
                    tool_valid += 1
                if step_args_ok:
                    args_valid += 1
                if step_schema_ok:
                    schema_valid += 1
                tool_elapsed_ms = int((time.perf_counter() - tool_start) * 1000)
                print(f"    -> Result: {str(batch_results)[:100]}...")
                if primary_exec_res:
                    status_line, status_code, status_detail = _path_status_line(primary_tool_name, primary_tool_args, primary_exec_res)
                    print(status_line)
                    
                batch_payload = json.dumps(batch_results, ensure_ascii=False)
                noisy_batch_payload = apply_tool_noise(batch_payload, vfs.rng, config.noise_level, mode=config.noise_mode)
                last_tool_output = noisy_batch_payload
                messages.append({"role": "user", "content": noisy_batch_payload})
                transcript.append({
                    "step": steps,
                    "model_output": completion,
                    "tool_call": tool_calls_batch,
                    "tool_result": noisy_batch_payload,
                    "tool_chunks": step_tool_chunks,
                    "expected_next": expected_next_log,
                    "chunk_blocked": any(code == "E_CHUNK_BLOCK" for code in step_failure_codes),
                    "chunk_blocked_paths": sorted(chunk_blocked_paths),
                    "tool_failed": step_tool_failed,
                    "permission_denied": step_permission_denied,
                    "blocked_submit": step_blocked_submit,
                    "enforced_path": step_enforced_path,
                    "tool_retries": 0,
                    "tool_latency_ms": 0,
                    "tool_timed_out": step_tool_timed_out,
                    "rejected_multi_tool": False,
                    "failure_code": step_failure_codes or None,
                    "parallel_tools": True,
                    "tool_elapsed_ms": tool_elapsed_ms,
                    "stream_started_paths": step_stream_started_paths,
                    "stream_completed_paths": step_stream_completed_paths,
                    "world_changed": step_world_changed,
                    **_stream_snapshot()
                })
            else:
                ctx = {
                    "config": config,
                    "vfs": vfs,
                    "expected_next": expected_next,
                    "expected_index": expected_index,
                    "wrong_read_streak": wrong_read_streak,
                    "wrong_trace_reads": wrong_trace_reads,
                    "explore_per_hop": explore_per_hop,
                    "explore_total": explore_total,
                    "explore_over_budget_events": explore_over_budget_events,
                    "secret_found": secret_found,
                    "secret_code_seen": secret_code_seen,
                    "memory_store": memory_store,
                    "checkpoints": checkpoints,
                    "messages": messages,
                    "chunk_blocked_paths": chunk_blocked_paths,
                }
                _track_artifacts_access(tool_name, tool_args)
                exec_res = execute_tool(tool_name, tool_args, ctx, allow_advance=True)
                expected_index = exec_res["expected_index"]
                wrong_read_streak = exec_res["wrong_read_streak"]
                wrong_trace_reads = exec_res["wrong_trace_reads"]
                explore_per_hop = exec_res["explore_per_hop"]
                explore_total = exec_res["explore_total"]
                explore_over_budget_events = exec_res["explore_over_budget_events"]
                secret_found = exec_res["secret_found"]
                secret_code_seen = exec_res["secret_code_seen"]
                chunk_blocked_paths = exec_res["chunk_blocked_paths"]
                if exec_res.get("chunk_blocked"):
                    print(f"    -> [CHUNK-BLOCK] {exec_res.get('chunk_blocked_path')}")
                _track_streaming(exec_res)
                _maybe_hint_artifacts(tool_name, tool_args, exec_res)
                _maybe_hint_dead_end(tool_name, exec_res)
                _maybe_hint_chain_complete(tool_name, exec_res)
                _maybe_hint_explore_budget(exec_res)
                if tool_name == "list_files":
                    list_files_calls += 1
                elif tool_name == "read_file":
                    read_file_calls += 1
                    p = tool_args.get("path", "").strip("./")
                    if p:
                        if p in read_file_paths:
                            redundant_reads += 1
                        else:
                            read_file_paths.add(p)
                        all_read_paths.add(p)
                elif tool_name == "read_file_chunk":
                    read_file_chunk_calls += 1
                    p = tool_args.get("path", "").strip("./")
                    if p:
                        all_read_paths.add(p)
                elif tool_name == "write_file":
                    write_file_calls += 1
                elif tool_name == "search_text":
                    search_text_calls += 1
                elif tool_name == "api_lookup":
                    api_lookup_calls += 1
                elif tool_name == "submit_answer":
                    submit_attempts += 1
                if exec_res["advanced"]:
                    path_correct += 1
                if exec_res["tool_ok"]:
                    tool_valid += 1
                if exec_res["args_ok"]:
                    args_valid += 1
                if _schema_valid(tool_name, tool_args):
                    schema_valid += 1
                if _output_schema_valid(tool_name, exec_res["result"]):
                    output_schema_valid += 1
                if exec_res["tool_failed"]:
                    tool_failed_events += 1
                    pending_error_recovery += 1
                    hint_since_error = False
                if not exec_res["tool_failed"] and isinstance(exec_res["result"], str) and not _is_error_payload(exec_res["result"]):
                    if pending_error_recovery > 0 and not hint_since_error:
                        tool_error_recovery_success += 1
                        pending_error_recovery -= 1
                        hint_since_error = False
                if exec_res["blocked_submit"]:
                    blocked_submit_attempts += 1
                if exec_res["honeypot_hit"]:
                    honeypot_hits += 1
                tool_elapsed_ms = int((time.perf_counter() - tool_start) * 1000)
                result = exec_res["result"]
                tool_chunks = exec_res["tool_chunks"]
                finished = exec_res["finished"]
                failure_code = _failure_code_for_exec(exec_res)
                print(f"    -> Result: {str(result)[:100]}...")
                status_line, status_code, status_detail = _path_status_line(tool_name, tool_args, exec_res)
                print(status_line)
                
                noisy_result = apply_tool_noise(result, vfs.rng, config.noise_level, mode=config.noise_mode)
                last_tool_output = noisy_result
                messages.append({"role": "user", "content": noisy_result})
                transcript.append({
                    "step": steps,
                    "model_output": completion,
                    "tool_call": tool_call,
                    "tool_result": noisy_result,
                    "tool_chunks": tool_chunks,
                    "expected_next": expected_next_log,
                    "chunk_blocked": exec_res.get("chunk_blocked", False),
                    "chunk_blocked_path": exec_res.get("chunk_blocked_path"),
                    "tool_failed": exec_res["tool_failed"],
                    "permission_denied": exec_res["permission_denied"],
                    "blocked_submit": exec_res["blocked_submit"],
                    "enforced_path": exec_res["enforced_path"],
                    "failure_code": failure_code,
                    "tool_retries": exec_res["retries"],
                    "tool_latency_ms": exec_res["latency_ms"],
                    "tool_timed_out": exec_res["tool_timed_out"],
                    "rejected_multi_tool": False,
                    "tool_elapsed_ms": tool_elapsed_ms,
                    "stream_started_path": exec_res.get("stream_started_path"),
                    "stream_completed_path": exec_res.get("stream_completed_path"),
                    "world_changed": exec_res.get("world_changed", False),
                    **_stream_snapshot()
                })

            if config.auto_submit_on_secret and secret_code_seen and not finished:
                messages.append({
                    "role": "user",
                    "content": f"System: You found the secret code ({secret_code_seen}). Submit it using submit_answer."
                })

            if finished:
                return AgentTurnResult(finished=True, full_result={
                    "success": True,
                    "task_success": True,
                    "compliance_success": _compliance_success(),
                    "steps_taken": steps,
                    "time_sec": time.time() - start_time,
                    "total_tokens": total_tokens,
                    "failure_reason": None,
                    "chain_length": config.chain_length,
                    "decoy_count": config.decoy_count,
                    "ambiguity_rate": config.ambiguity_rate,
                    "noise_level": config.noise_level,
                    "tool_failure_rate": config.tool_failure_rate,
                    "permission_deny_rate": config.permission_deny_rate,
                    "stream_rate": config.stream_rate,
                    "api_rate_limit": config.api_rate_limit,
                    "api_rate_window_sec": config.api_rate_window_sec,
                    "tool_min_latency_ms": config.tool_min_latency_ms,
                    "tool_max_latency_ms": config.tool_max_latency_ms,
                    "tool_timeout_ms": config.tool_timeout_ms,
                    "read_max_bytes": config.read_max_bytes,
                    "read_truncate_rate": config.read_truncate_rate,
                    "long_file_rate": config.long_file_rate,
                    "concurrent_change_rate": config.concurrent_change_rate,
                    "seed": vfs.seed,
                    "json_valid_rate": json_valid / steps if steps else 0,
                    "tool_valid_rate": tool_valid / steps if steps else 0,
                    "args_valid_rate": args_valid / steps if steps else 0,
                    "schema_valid_rate": schema_valid / steps if steps else 0,
                    "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
                    "path_correct_rate": path_correct / max(1, config.chain_length),
                    "artifacts_access_before_unlock": artifacts_access_before_unlock,
                    "wrong_trace_reads": wrong_trace_reads,
                    "explore_over_budget_events": explore_over_budget_events,
                    "fork_present": vfs.fork_present,
                    "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
                    "fork_dead_end_hit": vfs.fork_dead_end_hit,
                    "fork_recovered": vfs.fork_recovered,
                    "stream_sessions_started": stream_sessions_started,
                    "stream_sessions_completed": stream_sessions_completed,
                    "stream_abandoned_paths": sorted(active_streams),
                    "unique_files_read": len(all_read_paths),
                    "redundant_reads": redundant_reads,
                    "list_files_calls": list_files_calls,
                    "read_file_calls": read_file_calls,
                    "read_file_chunk_calls": read_file_chunk_calls,
                    "write_file_calls": write_file_calls,
                    "search_text_calls": search_text_calls,
                    "api_lookup_calls": api_lookup_calls,
                    "submit_attempts": submit_attempts,
                    "blocked_submit_attempts": blocked_submit_attempts,
                    "tool_failed_events": tool_failed_events,
                    "tool_error_recovery_success": tool_error_recovery_success,
                    "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
                    "honeypot_hits": honeypot_hits,
                    "last_tool_output": last_tool_output,
                    "transcript": transcript
                })
            
        except Exception as e:
            print(f"Error in loop: {e}")
            return AgentTurnResult(finished=True, full_result={
                "success": False,
                "task_success": False,
                "compliance_success": _compliance_success(),
                "failure_reason": f"Exception: {str(e)}",
                "steps_taken": steps,
                "last_tool_output": last_tool_output,
                "chain_length": config.chain_length,
                "decoy_count": config.decoy_count,
                "ambiguity_rate": config.ambiguity_rate,
                "noise_level": config.noise_level,
                "tool_failure_rate": config.tool_failure_rate,
                "permission_deny_rate": config.permission_deny_rate,
                "stream_rate": config.stream_rate,
                "api_rate_limit": config.api_rate_limit,
                "api_rate_window_sec": config.api_rate_window_sec,
                "tool_min_latency_ms": config.tool_min_latency_ms,
                "tool_max_latency_ms": config.tool_max_latency_ms,
                "tool_timeout_ms": config.tool_timeout_ms,
                "read_max_bytes": config.read_max_bytes,
                "read_truncate_rate": config.read_truncate_rate,
                "long_file_rate": config.long_file_rate,
                "concurrent_change_rate": config.concurrent_change_rate,
                "seed": vfs.seed,
                "json_valid_rate": json_valid / steps if steps else 0,
                "tool_valid_rate": tool_valid / steps if steps else 0,
                "args_valid_rate": args_valid / steps if steps else 0,
                "schema_valid_rate": schema_valid / steps if steps else 0,
                "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
                "path_correct_rate": path_correct / max(1, config.chain_length),
                "artifacts_access_before_unlock": artifacts_access_before_unlock,
                "wrong_trace_reads": wrong_trace_reads,
                "explore_over_budget_events": explore_over_budget_events,
                "fork_present": vfs.fork_present,
                "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
                "fork_dead_end_hit": vfs.fork_dead_end_hit,
                "fork_recovered": vfs.fork_recovered,
                "stream_sessions_started": stream_sessions_started,
                "stream_sessions_completed": stream_sessions_completed,
                "stream_abandoned_paths": sorted(active_streams),
                "unique_files_read": len(all_read_paths),
                "redundant_reads": redundant_reads,
                "list_files_calls": list_files_calls,
                "read_file_calls": read_file_calls,
                "read_file_chunk_calls": read_file_chunk_calls,
                "write_file_calls": write_file_calls,
                "search_text_calls": search_text_calls,
                "api_lookup_calls": api_lookup_calls,
                "submit_attempts": submit_attempts,
                "blocked_submit_attempts": blocked_submit_attempts,
                "tool_failed_events": tool_failed_events,
                "tool_error_recovery_success": tool_error_recovery_success,
                "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
                "honeypot_hits": honeypot_hits,
                "transcript": transcript
            })
            

        return AgentTurnResult(
            status=status_code if 'status_code' in locals() else 'N/A',
            drift=drift_score,
            finished=finished if 'finished' in locals() else False,
            tool_name=tool_name if 'tool_name' in locals() else None,
            tool_args=tool_args if 'tool_args' in locals() else None,
            tool_result=last_tool_output, # Approx
            detail=status_detail if 'status_detail' in locals() else None,
            expected_next=expected_next if 'expected_next' in locals() else None
        )
    
    # Driver Loop
    while steps < MAX_STEPS:
        # steps += 1 (moved to _do_agent_turn)
        
        # 1. Primary Turn
        res = _do_agent_turn()
        
        if res.finished:
            res.full_result["watcher_interventions"] = watcher_interventions
            return res.full_result

        # Auto-submit secret logic (moved out of closure or kept in closure? It modifies messages. 
        # The closure should handle it, or we duplicate. Original code had it after loop body.
        # Actually in original code it was inside the loop. 
        # We'll rely on the closure handling it if it was inside.)
        
        # 2. Watcher Loop
        if config.enable_watcher and config.watcher_model and res.status in ("WRONG", "OFF") and res.drift > 0:
            intervention_streak = 0
            while res.status in ("WRONG", "OFF") and res.drift > 0 and intervention_streak < config.watcher_retry_cap and steps < MAX_STEPS:
                
                # Ask Watcher
                advice = _run_watcher_intervention(
                    config=config,
                    base_url=base_url,
                    watcher_model=config.watcher_model,
                    messages=messages,
                    status=res.status,
                    detail=res.detail,
                    drift_score=res.drift,
                    step_retries=intervention_streak,
                    tool_name=res.tool_name,
                    tool_args=res.tool_args,
                    tool_result=res.tool_result,
                    expected_next=res.expected_next
                )
                
                if not advice:
                    break
                    
                messages.append({"role": "user", "content": advice})
                
                # Retry Turn
                # steps += 1 (moved to _do_agent_turn)
                watcher_interventions += 1
                intervention_streak += 1
                
                res = _do_agent_turn()
                
                if res.status == "OK":
                    break
                if res.finished:
                    res.full_result["watcher_interventions"] = watcher_interventions
                    return res.full_result

    # End of MAX_STEPS

    return {
        "success": False,
        "task_success": False,
        "compliance_success": _compliance_success(),
        "failure_reason": "Max steps reached",
        "steps_taken": steps,
        "watcher_interventions": watcher_interventions,
        "last_tool_output": last_tool_output,
        "chain_length": config.chain_length,
        "decoy_count": config.decoy_count,
        "ambiguity_rate": config.ambiguity_rate,
        "noise_level": config.noise_level,
        "tool_failure_rate": config.tool_failure_rate,
        "permission_deny_rate": config.permission_deny_rate,
        "stream_rate": config.stream_rate,
        "api_rate_limit": config.api_rate_limit,
        "api_rate_window_sec": config.api_rate_window_sec,
        "tool_min_latency_ms": config.tool_min_latency_ms,
        "tool_max_latency_ms": config.tool_max_latency_ms,
        "tool_timeout_ms": config.tool_timeout_ms,
        "read_max_bytes": config.read_max_bytes,
        "read_truncate_rate": config.read_truncate_rate,
        "long_file_rate": config.long_file_rate,
        "concurrent_change_rate": config.concurrent_change_rate,
        "seed": vfs.seed,
        "json_valid_rate": json_valid / steps if steps else 0,
        "tool_valid_rate": tool_valid / steps if steps else 0,
        "args_valid_rate": args_valid / steps if steps else 0,
        "schema_valid_rate": schema_valid / steps if steps else 0,
        "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
        "path_correct_rate": path_correct / max(1, config.chain_length),
        "artifacts_access_before_unlock": artifacts_access_before_unlock,
        "wrong_trace_reads": wrong_trace_reads,
        "explore_over_budget_events": explore_over_budget_events,
        "fork_present": vfs.fork_present,
        "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
        "fork_dead_end_hit": vfs.fork_dead_end_hit,
        "fork_recovered": vfs.fork_recovered,
        "stream_sessions_started": stream_sessions_started,
        "stream_sessions_completed": stream_sessions_completed,
        "stream_abandoned_paths": sorted(active_streams),
        "unique_files_read": len(all_read_paths),
        "redundant_reads": redundant_reads,
        "list_files_calls": list_files_calls,
        "read_file_calls": read_file_calls,
        "read_file_chunk_calls": read_file_chunk_calls,
        "write_file_calls": write_file_calls,
        "search_text_calls": search_text_calls,
        "api_lookup_calls": api_lookup_calls,
        "submit_attempts": submit_attempts,
        "blocked_submit_attempts": blocked_submit_attempts,
        "tool_failed_events": tool_failed_events,
        "tool_error_recovery_success": tool_error_recovery_success,
        "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
        "honeypot_hits": honeypot_hits,
        "transcript": transcript
    }

# --------------------------------------------------------------------------------
# BASELINES
# --------------------------------------------------------------------------------

def run_baseline_oracle(config: argparse.Namespace, seed_override: Optional[int] = None) -> Dict[str, Any]:
    vfs = VirtualFileSystem(
        chain_length=config.chain_length,
        decoy_count=config.decoy_count,
        ambiguity_rate=config.ambiguity_rate,
        permission_deny_rate=config.permission_deny_rate,
        long_file_rate=config.long_file_rate,
        mode=config.mode,
        seed=seed_override if seed_override is not None else config.seed
    )
    vfs.max_chunk_reads_per_file = config.max_chunk_reads_per_file
    steps = 0
    total_tokens = 0
    start_time = time.time()
    expected_index = -1
    last_tool_output = ""
    secret_found = False
    secret_code_seen = None
    wrong_read_streak = 0
    wrong_trace_reads = 0
    explore_per_hop = 0
    explore_total = 0
    explore_over_budget_events = 0
    memory_store = {}
    checkpoints = {}

    json_valid = 0
    tool_valid = 0
    args_valid = 0
    schema_valid = 0
    output_schema_valid = 0
    path_correct = 0
    list_files_calls = 0
    read_file_calls = 0
    read_file_chunk_calls = 0
    write_file_calls = 0
    search_text_calls = 0
    api_lookup_calls = 0
    submit_attempts = 0
    blocked_submit_attempts = 0
    tool_failed_events = 0
    tool_error_recovery_success = 0
    pending_error_recovery = 0
    honeypot_hits = 0
    all_read_paths = set()
    read_file_paths = set()
    redundant_reads = 0
    artifacts_access_before_unlock = 0
    stream_sessions_started = 0
    stream_sessions_completed = 0
    active_streams = set()
    chunk_blocked_paths = set()

    transcript = []

    def _read_file_text(res: Any) -> str:
        if not isinstance(res, str):
            return ""
        try:
            obj = json.loads(res)
            if isinstance(obj, dict):
                if "content" in obj and isinstance(obj.get("content"), str):
                    return obj.get("content") or ""
                if "chunk" in obj and isinstance(obj.get("chunk"), str):
                    return obj.get("chunk") or ""
        except Exception:
            return res
        return res

    def _track_artifacts_access(tool_name: str, tool_args: Dict[str, Any]):
        nonlocal artifacts_access_before_unlock
        if vfs.artifacts_unlocked:
            return
        if tool_name in ("list_files", "search_text", "glob", "tree"):
            target = tool_args.get("directory", "")
        elif tool_name in ("read_file", "read_file_chunk", "read_json", "stat"):
            target = tool_args.get("path", "")
        else:
            return
        if vfs._is_artifacts_path(target):
            artifacts_access_before_unlock += 1

    def _track_streaming(exec_res: Dict[str, Any]):
        nonlocal stream_sessions_started, stream_sessions_completed
        start_path = exec_res.get("stream_started_path")
        done_path = exec_res.get("stream_completed_path")
        if start_path:
            p = start_path.strip("./")
            if p not in active_streams:
                stream_sessions_started += 1
                active_streams.add(p)
        if done_path:
            p = done_path.strip("./")
            if p in active_streams or (start_path and p == start_path.strip("./")):
                stream_sessions_completed += 1
                active_streams.discard(p)

    def _exec(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal steps, expected_index, wrong_read_streak, wrong_trace_reads, explore_per_hop, explore_total
        nonlocal explore_over_budget_events, secret_found, secret_code_seen, last_tool_output, json_valid
        nonlocal tool_valid, args_valid, schema_valid, output_schema_valid, tool_failed_events, path_correct
        nonlocal tool_error_recovery_success, pending_error_recovery, redundant_reads, chunk_blocked_paths
        nonlocal list_files_calls, read_file_calls, read_file_chunk_calls, write_file_calls, search_text_calls
        nonlocal api_lookup_calls, submit_attempts, blocked_submit_attempts, honeypot_hits

        steps += 1
        json_valid += 1
        tool_valid += 1
        args_valid += 1
        if _schema_valid(tool_name, tool_args):
            schema_valid += 1

        if tool_name == "list_files":
            list_files_calls += 1
        elif tool_name == "read_file":
            read_file_calls += 1
            p = tool_args.get("path", "").strip("./")
            if p:
                if p in read_file_paths:
                    redundant_reads += 1
                else:
                    read_file_paths.add(p)
                all_read_paths.add(p)
        elif tool_name == "read_file_chunk":
            read_file_chunk_calls += 1
            p = tool_args.get("path", "").strip("./")
            if p:
                all_read_paths.add(p)
        elif tool_name == "write_file":
            write_file_calls += 1
        elif tool_name == "search_text":
            search_text_calls += 1
        elif tool_name == "api_lookup":
            api_lookup_calls += 1
        elif tool_name == "submit_answer":
            submit_attempts += 1

        _track_artifacts_access(tool_name, tool_args)
        expected_next = None
        if expected_index + 1 < len(vfs.path_sequence):
            expected_next = vfs.path_sequence[expected_index + 1]
        ctx = {
            "config": config,
            "vfs": vfs,
            "expected_next": expected_next,
            "expected_index": expected_index,
            "wrong_read_streak": wrong_read_streak,
            "wrong_trace_reads": wrong_trace_reads,
            "explore_per_hop": explore_per_hop,
            "explore_total": explore_total,
            "explore_over_budget_events": explore_over_budget_events,
            "secret_found": secret_found,
            "secret_code_seen": secret_code_seen,
            "memory_store": memory_store,
            "checkpoints": checkpoints,
            "messages": [],
            "chunk_blocked_paths": chunk_blocked_paths,
        }
        exec_res = execute_tool(tool_name, tool_args, ctx, allow_advance=True)
        expected_index = exec_res["expected_index"]
        wrong_read_streak = exec_res["wrong_read_streak"]
        wrong_trace_reads = exec_res["wrong_trace_reads"]
        explore_per_hop = exec_res["explore_per_hop"]
        explore_total = exec_res["explore_total"]
        explore_over_budget_events = exec_res["explore_over_budget_events"]
        secret_found = exec_res["secret_found"]
        secret_code_seen = exec_res["secret_code_seen"]
        chunk_blocked_paths = exec_res["chunk_blocked_paths"]
        _track_streaming(exec_res)
        if exec_res["advanced"]:
            path_correct += 1
        if _output_schema_valid(tool_name, exec_res["result"]):
            output_schema_valid += 1
        if exec_res["tool_failed"]:
            tool_failed_events += 1
            pending_error_recovery += 1
        if not exec_res["tool_failed"] and isinstance(exec_res["result"], str) and not _is_error_payload(exec_res["result"]):
            if pending_error_recovery > 0:
                tool_error_recovery_success += 1
                pending_error_recovery -= 1
        if exec_res["blocked_submit"]:
            blocked_submit_attempts += 1
        if exec_res["honeypot_hit"]:
            honeypot_hits += 1
        last_tool_output = exec_res["result"]
        transcript.append({
            "step": steps,
            "tool_call": {"tool": tool_name, "args": tool_args},
            "tool_result": exec_res["result"],
            "expected_next": expected_next,
            "tool_failed": exec_res["tool_failed"],
            "permission_denied": exec_res["permission_denied"],
            "blocked_submit": exec_res["blocked_submit"],
            "enforced_path": exec_res["enforced_path"],
            "tool_retries": exec_res["retries"],
            "tool_latency_ms": exec_res["latency_ms"],
            "tool_timed_out": exec_res["tool_timed_out"],
            "tool_chunks": exec_res["tool_chunks"],
        })
        return exec_res

    _exec("read_file", {"path": "start_instructions.txt"})
    _exec("read_file", {"path": "task.md"})
    readme_res = _exec("read_file", {"path": "readme.md"})
    readme_text = _read_file_text(readme_res.get("result"))
    codename = None
    m = re.search(r"Codename:\s*([A-Za-z0-9_-]+)", readme_text)
    if m:
        codename = m.group(1).lower()
    for p in vfs.path_sequence:
        _exec("read_file", {"path": p})

    if codename:
        index_path = f"artifacts/{codename}/index.txt"
        index_res = _exec("read_file", {"path": index_path})
        index_text = _read_file_text(index_res.get("result"))
        chunk_paths = [line.strip() for line in index_text.splitlines() if line.strip()]
        chunks = []
        for cp in chunk_paths:
            chunk_res = _exec("read_file", {"path": cp})
            chunk_text = _read_file_text(chunk_res.get("result"))
            m = re.search(r"SECRET_CHUNK=([A-Za-z0-9_\\-]+)", chunk_text)
            if m:
                chunks.append(m.group(1))
        secret = "".join(chunks) if chunks else vfs.secret_code
        _exec("write_file", {"path": vfs.solution_path, "content": secret})
        submit_res = _exec("submit_answer", {"answer": secret})
    else:
        submit_res = _exec("submit_answer", {"answer": "UNKNOWN"})

    success = bool(submit_res.get("finished"))
    return {
        "success": success,
        "task_success": success,
        "compliance_success": success,
        "steps_taken": steps,
        "time_sec": time.time() - start_time,
        "total_tokens": total_tokens,
        "failure_reason": None if success else "baseline_incomplete",
        "chain_length": config.chain_length,
        "decoy_count": config.decoy_count,
        "ambiguity_rate": config.ambiguity_rate,
        "noise_level": config.noise_level,
        "tool_failure_rate": config.tool_failure_rate,
        "permission_deny_rate": config.permission_deny_rate,
        "stream_rate": config.stream_rate,
        "api_rate_limit": config.api_rate_limit,
        "api_rate_window_sec": config.api_rate_window_sec,
        "tool_min_latency_ms": config.tool_min_latency_ms,
        "tool_max_latency_ms": config.tool_max_latency_ms,
        "tool_timeout_ms": config.tool_timeout_ms,
        "read_max_bytes": config.read_max_bytes,
        "read_truncate_rate": config.read_truncate_rate,
        "long_file_rate": config.long_file_rate,
        "concurrent_change_rate": config.concurrent_change_rate,
        "seed": vfs.seed,
        "json_valid_rate": json_valid / steps if steps else 0,
        "tool_valid_rate": tool_valid / steps if steps else 0,
        "args_valid_rate": args_valid / steps if steps else 0,
        "schema_valid_rate": schema_valid / steps if steps else 0,
        "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
        "path_correct_rate": path_correct / max(1, config.chain_length),
        "artifacts_access_before_unlock": artifacts_access_before_unlock,
        "wrong_trace_reads": wrong_trace_reads,
        "explore_over_budget_events": explore_over_budget_events,
        "fork_present": vfs.fork_present,
        "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
        "fork_dead_end_hit": vfs.fork_dead_end_hit,
        "fork_recovered": vfs.fork_recovered,
        "stream_sessions_started": stream_sessions_started,
        "stream_sessions_completed": stream_sessions_completed,
        "stream_abandoned_paths": sorted(active_streams),
        "unique_files_read": len(all_read_paths),
        "redundant_reads": redundant_reads,
        "list_files_calls": list_files_calls,
        "read_file_calls": read_file_calls,
        "read_file_chunk_calls": read_file_chunk_calls,
        "write_file_calls": write_file_calls,
        "search_text_calls": search_text_calls,
        "api_lookup_calls": api_lookup_calls,
        "submit_attempts": submit_attempts,
        "blocked_submit_attempts": blocked_submit_attempts,
        "tool_failed_events": tool_failed_events,
        "tool_error_recovery_success": tool_error_recovery_success,
        "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
        "honeypot_hits": honeypot_hits,
        "last_tool_output": last_tool_output,
        "transcript": transcript
    }

def run_baseline_random(config: argparse.Namespace, seed_override: Optional[int] = None) -> Dict[str, Any]:
    vfs = VirtualFileSystem(
        chain_length=config.chain_length,
        decoy_count=config.decoy_count,
        ambiguity_rate=config.ambiguity_rate,
        permission_deny_rate=config.permission_deny_rate,
        long_file_rate=config.long_file_rate,
        mode=config.mode,
        seed=seed_override if seed_override is not None else config.seed
    )
    vfs.max_chunk_reads_per_file = config.max_chunk_reads_per_file
    steps = 0
    total_tokens = 0
    start_time = time.time()
    expected_index = -1
    last_tool_output = ""
    secret_found = False
    secret_code_seen = None
    wrong_read_streak = 0
    wrong_trace_reads = 0
    explore_per_hop = 0
    explore_total = 0
    explore_over_budget_events = 0
    memory_store = {}
    checkpoints = {}

    json_valid = 0
    tool_valid = 0
    args_valid = 0
    schema_valid = 0
    output_schema_valid = 0
    path_correct = 0
    list_files_calls = 0
    read_file_calls = 0
    read_file_chunk_calls = 0
    write_file_calls = 0
    search_text_calls = 0
    api_lookup_calls = 0
    submit_attempts = 0
    blocked_submit_attempts = 0
    tool_failed_events = 0
    tool_error_recovery_success = 0
    pending_error_recovery = 0
    honeypot_hits = 0
    all_read_paths = set()
    read_file_paths = set()
    redundant_reads = 0
    artifacts_access_before_unlock = 0
    stream_sessions_started = 0
    stream_sessions_completed = 0
    active_streams = set()
    chunk_blocked_paths = set()

    transcript = []

    all_paths = sorted(vfs.all_paths)
    dirs = sorted({p.split("/", 1)[0] for p in all_paths if "/" in p} | {""})

    def _track_artifacts_access(tool_name: str, tool_args: Dict[str, Any]):
        nonlocal artifacts_access_before_unlock
        if vfs.artifacts_unlocked:
            return
        if tool_name in ("list_files", "search_text", "glob", "tree"):
            target = tool_args.get("directory", "")
        elif tool_name in ("read_file", "read_file_chunk", "read_json", "stat"):
            target = tool_args.get("path", "")
        else:
            return
        if vfs._is_artifacts_path(target):
            artifacts_access_before_unlock += 1

    def _track_streaming(exec_res: Dict[str, Any]):
        nonlocal stream_sessions_started, stream_sessions_completed
        start_path = exec_res.get("stream_started_path")
        done_path = exec_res.get("stream_completed_path")
        if start_path:
            p = start_path.strip("./")
            if p not in active_streams:
                stream_sessions_started += 1
                active_streams.add(p)
        if done_path:
            p = done_path.strip("./")
            if p in active_streams or (start_path and p == start_path.strip("./")):
                stream_sessions_completed += 1
                active_streams.discard(p)

    def _exec(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal steps, expected_index, wrong_read_streak, wrong_trace_reads, explore_per_hop, explore_total
        nonlocal explore_over_budget_events, secret_found, secret_code_seen, last_tool_output, json_valid
        nonlocal tool_valid, args_valid, schema_valid, output_schema_valid, tool_failed_events, path_correct
        nonlocal tool_error_recovery_success, pending_error_recovery, redundant_reads, chunk_blocked_paths
        nonlocal list_files_calls, read_file_calls, read_file_chunk_calls, write_file_calls, search_text_calls
        nonlocal api_lookup_calls, submit_attempts, blocked_submit_attempts, honeypot_hits

        steps += 1
        json_valid += 1
        tool_valid += 1
        args_valid += 1
        if _schema_valid(tool_name, tool_args):
            schema_valid += 1

        if tool_name == "list_files":
            list_files_calls += 1
        elif tool_name == "read_file":
            read_file_calls += 1
            p = tool_args.get("path", "").strip("./")
            if p:
                if p in read_file_paths:
                    redundant_reads += 1
                else:
                    read_file_paths.add(p)
                all_read_paths.add(p)
        elif tool_name == "read_file_chunk":
            read_file_chunk_calls += 1
            p = tool_args.get("path", "").strip("./")
            if p:
                all_read_paths.add(p)
        elif tool_name == "write_file":
            write_file_calls += 1
        elif tool_name == "search_text":
            search_text_calls += 1
        elif tool_name == "api_lookup":
            api_lookup_calls += 1
        elif tool_name == "submit_answer":
            submit_attempts += 1

        _track_artifacts_access(tool_name, tool_args)
        expected_next = None
        if expected_index + 1 < len(vfs.path_sequence):
            expected_next = vfs.path_sequence[expected_index + 1]
        ctx = {
            "config": config,
            "vfs": vfs,
            "expected_next": expected_next,
            "expected_index": expected_index,
            "wrong_read_streak": wrong_read_streak,
            "wrong_trace_reads": wrong_trace_reads,
            "explore_per_hop": explore_per_hop,
            "explore_total": explore_total,
            "explore_over_budget_events": explore_over_budget_events,
            "secret_found": secret_found,
            "secret_code_seen": secret_code_seen,
            "memory_store": memory_store,
            "checkpoints": checkpoints,
            "messages": [],
            "chunk_blocked_paths": chunk_blocked_paths,
        }
        exec_res = execute_tool(tool_name, tool_args, ctx, allow_advance=True)
        expected_index = exec_res["expected_index"]
        wrong_read_streak = exec_res["wrong_read_streak"]
        wrong_trace_reads = exec_res["wrong_trace_reads"]
        explore_per_hop = exec_res["explore_per_hop"]
        explore_total = exec_res["explore_total"]
        explore_over_budget_events = exec_res["explore_over_budget_events"]
        secret_found = exec_res["secret_found"]
        secret_code_seen = exec_res["secret_code_seen"]
        chunk_blocked_paths = exec_res["chunk_blocked_paths"]
        _track_streaming(exec_res)
        if exec_res["advanced"]:
            path_correct += 1
        if _output_schema_valid(tool_name, exec_res["result"]):
            output_schema_valid += 1
        if exec_res["tool_failed"]:
            tool_failed_events += 1
            pending_error_recovery += 1
        if not exec_res["tool_failed"] and isinstance(exec_res["result"], str) and not _is_error_payload(exec_res["result"]):
            if pending_error_recovery > 0:
                tool_error_recovery_success += 1
                pending_error_recovery -= 1
        if exec_res["blocked_submit"]:
            blocked_submit_attempts += 1
        if exec_res["honeypot_hit"]:
            honeypot_hits += 1
        last_tool_output = exec_res["result"]
        transcript.append({
            "step": steps,
            "tool_call": {"tool": tool_name, "args": tool_args},
            "tool_result": exec_res["result"],
            "expected_next": expected_next,
            "tool_failed": exec_res["tool_failed"],
            "permission_denied": exec_res["permission_denied"],
            "blocked_submit": exec_res["blocked_submit"],
            "enforced_path": exec_res["enforced_path"],
            "tool_retries": exec_res["retries"],
            "tool_latency_ms": exec_res["latency_ms"],
            "tool_timed_out": exec_res["tool_timed_out"],
            "tool_chunks": exec_res["tool_chunks"],
        })
        return exec_res

    while steps < MAX_STEPS:
        tool_choice = vfs.rng.choice(["read_file", "list_files", "stat"])
        if tool_choice == "read_file":
            target = vfs.rng.choice(all_paths)
            _exec("read_file", {"path": target})
        elif tool_choice == "list_files":
            target = vfs.rng.choice(dirs)
            _exec("list_files", {"directory": target or "."})
        else:
            target = vfs.rng.choice(all_paths)
            _exec("stat", {"path": target})
        if vfs.solution_written and secret_found:
            break

    submit_res = _exec("submit_answer", {"answer": "OMEGA_000"})
    success = bool(submit_res.get("finished"))
    return {
        "success": success,
        "task_success": success,
        "compliance_success": success,
        "steps_taken": steps,
        "time_sec": time.time() - start_time,
        "total_tokens": total_tokens,
        "failure_reason": None if success else "baseline_incomplete",
        "chain_length": config.chain_length,
        "decoy_count": config.decoy_count,
        "ambiguity_rate": config.ambiguity_rate,
        "noise_level": config.noise_level,
        "tool_failure_rate": config.tool_failure_rate,
        "permission_deny_rate": config.permission_deny_rate,
        "stream_rate": config.stream_rate,
        "api_rate_limit": config.api_rate_limit,
        "api_rate_window_sec": config.api_rate_window_sec,
        "tool_min_latency_ms": config.tool_min_latency_ms,
        "tool_max_latency_ms": config.tool_max_latency_ms,
        "tool_timeout_ms": config.tool_timeout_ms,
        "read_max_bytes": config.read_max_bytes,
        "read_truncate_rate": config.read_truncate_rate,
        "long_file_rate": config.long_file_rate,
        "concurrent_change_rate": config.concurrent_change_rate,
        "seed": vfs.seed,
        "json_valid_rate": json_valid / steps if steps else 0,
        "tool_valid_rate": tool_valid / steps if steps else 0,
        "args_valid_rate": args_valid / steps if steps else 0,
        "schema_valid_rate": schema_valid / steps if steps else 0,
        "output_schema_valid_rate": output_schema_valid / steps if steps else 0,
        "path_correct_rate": path_correct / max(1, config.chain_length),
        "artifacts_access_before_unlock": artifacts_access_before_unlock,
        "wrong_trace_reads": wrong_trace_reads,
        "explore_over_budget_events": explore_over_budget_events,
        "fork_present": vfs.fork_present,
        "fork_wrong_branch_reads": vfs.fork_wrong_branch_reads,
        "fork_dead_end_hit": vfs.fork_dead_end_hit,
        "fork_recovered": vfs.fork_recovered,
        "stream_sessions_started": stream_sessions_started,
        "stream_sessions_completed": stream_sessions_completed,
        "stream_abandoned_paths": sorted(active_streams),
        "unique_files_read": len(all_read_paths),
        "redundant_reads": redundant_reads,
        "list_files_calls": list_files_calls,
        "read_file_calls": read_file_calls,
        "read_file_chunk_calls": read_file_chunk_calls,
        "write_file_calls": write_file_calls,
        "search_text_calls": search_text_calls,
        "api_lookup_calls": api_lookup_calls,
        "submit_attempts": submit_attempts,
        "blocked_submit_attempts": blocked_submit_attempts,
        "tool_failed_events": tool_failed_events,
        "tool_error_recovery_success": tool_error_recovery_success,
        "recovery_rate": (tool_error_recovery_success / tool_failed_events) if tool_failed_events else 1.0,
        "honeypot_hits": honeypot_hits,
        "last_tool_output": last_tool_output,
        "transcript": transcript
    }

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def run_bench(args):
    # 1. Model Selection
    models = []
    if args.active:
        try:
            print("Probing active model...")
            r = requests.get(f"{args.base_url}/models", timeout=5)
            # Actually just try chat to see what responds? Or assume LM Studio 'local-model' alias works?
            # Existing script probes with a dummy chat.
            probe = requests.post(
                f"{args.base_url}/chat/completions",
                json={"model": "local-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=10
            )
            probe.raise_for_status()
            mid = probe.json().get("model") or "unknown-model"
            print(f"Detected: {mid}")
            models = [mid]
        except Exception as e:
            print(f"Probe failed: {e}")
            sys.exit(1)
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        # List all
        try:
            r = requests.get(f"{args.base_url}/models", timeout=5)
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
        except:
            print("Failed to list models")
            sys.exit(1)

    results = []
    if isinstance(args.benchmark_track, str):
        if args.benchmark_track == "both":
            args.benchmark_track = ["realistic", "assisted"]
        else:
            args.benchmark_track = [args.benchmark_track]
    if args.baselines:
        models.extend(["baseline_oracle", "baseline_random"])
    weights_total = args.w_task + args.w_protocol + args.w_robust + args.w_eff
    weights = (
        args.w_task / weights_total,
        args.w_protocol / weights_total,
        args.w_robust / weights_total,
        args.w_eff / weights_total,
    )
    
    for m in models:
        for benchmark_track in args.benchmark_track:
            run_args = deepcopy(args)
            run_args.benchmark_track = benchmark_track
            if benchmark_track == "assisted":
                run_args.strict_follow = True
                run_args.enable_hints = True
                run_args.auto_submit_on_secret = True
            elif benchmark_track == "realistic":
                run_args.strict_follow = False
                run_args.enable_hints = False
                run_args.auto_submit_on_secret = False
                run_args.follow_policy = "none"

            print(f"\nExample Run: {m} [{benchmark_track}]")
            trial_results = []
            seeds = []
            for i in range(run_args.trials):
                seed = (run_args.seed + i) if run_args.seed is not None else random.randint(1, 1_000_000_000)
                seeds.append(seed)
                if m == "baseline_oracle":
                    res = run_baseline_oracle(run_args, seed_override=seed)
                elif m == "baseline_random":
                    res = run_baseline_random(run_args, seed_override=seed)
                else:
                    res = run_agent_loop(m, run_args.base_url, run_args, seed_override=seed)
                res["model"] = m
                res["benchmark_track"] = benchmark_track
                res["tokens_per_sec"] = res.get("total_tokens", 0) / res.get("time_sec", 1) if res.get("time_sec") else 0
                res.update(compute_scores(res, weights))
                res["robustness_stddev_overall"] = 0.0
                res["robustness_stddev_task"] = 0.0
                res["robustness_stddev_fidelity"] = 0.0
                res["robustness_stddev_tool"] = 0.0
                res["robustness_stddev_robustness"] = 0.0
                res["robustness_stddev_efficiency"] = 0.0
                res["trials"] = 1
                res["seeds"] = str(seed)
                trial_results.append(res)

            safe_name = sanitize_filename(m)
            out_dir = os.path.join(args.out_dir, benchmark_track, safe_name)
            os.makedirs(out_dir, exist_ok=True)

            if run_args.trials == 1:
                res = trial_results[0]
                results.append(res)
                results_file = os.path.join(out_dir, "results_longchain.csv")
                file_exists = os.path.exists(results_file)
                mode = "a" if run_args.append and file_exists else "w"
                with open(results_file, mode, newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    if mode == "w":
                        w.writeheader()
                    w.writerow({k: res.get(k, "") for k in CSV_FIELDS})
                table_file = os.path.join(out_dir, "results_longchain.table.txt")
                with open(table_file, "w", encoding="utf-8") as f:
                    f.write(render_table([res], CSV_FIELDS))
                trial_dir = os.path.join(out_dir, "trial_01")
                os.makedirs(trial_dir, exist_ok=True)
                with open(os.path.join(trial_dir, "details.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "model": m,
                        "time_sec": res.get("time_sec"),
                        "usage": {"total_tokens": res.get("total_tokens", 0)},
                        "result": res,
                    }, f, indent=2, ensure_ascii=False)
                with open(os.path.join(trial_dir, "transcript.json"), "w", encoding="utf-8") as f:
                    json.dump(res.get("transcript", []), f, indent=2, ensure_ascii=False)
                details = {
                    "model": m,
                    "benchmark_track": benchmark_track,
                    "weights": {
                        "task_success": weights[0],
                        "protocol": weights[1],
                        "robustness": weights[2],
                        "efficiency": weights[3],
                    },
                    "aggregate": res,
                    "trials": [
                        {
                            "trial": 1,
                            "seed": res.get("seed"),
                            "success": res.get("success"),
                            "steps_taken": res.get("steps_taken"),
                            "scores": {
                                "task_success_100": res.get("task_success_100"),
                                "compliance_success_100": res.get("compliance_success_100"),
                                "instruction_fidelity_100": res.get("instruction_fidelity_100"),
                                "tool_discipline_100": res.get("tool_discipline_100"),
                                "robustness_100": res.get("robustness_100"),
                                "efficiency_100": res.get("efficiency_100"),
                                "overall_score_100": res.get("overall_score_100"),
                            },
                            "paths": {
                                "details": os.path.join("trial_01", "details.json"),
                                "transcript": os.path.join("trial_01", "transcript.json"),
                            },
                        }
                    ],
                }
                with open(os.path.join(out_dir, "details.json"), "w", encoding="utf-8") as f:
                    json.dump(details, f, indent=2, ensure_ascii=False)
                print(
                    f"\nRESULT: Success={res['success']} Steps={res['steps_taken']} "
                    f"Time={res.get('time_sec',0):.2f}s Overall={res.get('overall_score_100')}"
                )
                continue

        task_scores = [t["task_success_100"] for t in trial_results]
        compliance_scores = [t["compliance_success_100"] for t in trial_results]
        fidelity_scores = [t["instruction_fidelity_100"] for t in trial_results]
        tool_scores = [t["tool_discipline_100"] for t in trial_results]
        robust_scores = [t["robustness_100"] for t in trial_results]
        eff_scores = [t["efficiency_100"] for t in trial_results]
        overall_scores = [t["overall_score_100"] for t in trial_results]
        time_vals = [t.get("time_sec", 0) or 0 for t in trial_results]
        steps_vals = [t.get("steps_taken", 0) or 0 for t in trial_results]

        agg = {
            "model": m,
            "benchmark_track": benchmark_track,
            "overall_score_100": round(sum(overall_scores) / len(overall_scores), 2),
            "task_success_100": round(sum(task_scores) / len(task_scores), 2),
            "compliance_success_100": round(sum(compliance_scores) / len(compliance_scores), 2),
            "instruction_fidelity_100": round(sum(fidelity_scores) / len(fidelity_scores), 2),
            "tool_discipline_100": round(sum(tool_scores) / len(tool_scores), 2),
            "robustness_100": round(sum(robust_scores) / len(robust_scores), 2),
            "efficiency_100": round(sum(eff_scores) / len(eff_scores), 2),
            "robustness_stddev_overall": round(statistics.pstdev(overall_scores), 2),
            "robustness_stddev_task": round(statistics.pstdev(task_scores), 2),
            "robustness_stddev_fidelity": round(statistics.pstdev(fidelity_scores), 2),
            "robustness_stddev_tool": round(statistics.pstdev(tool_scores), 2),
            "robustness_stddev_robustness": round(statistics.pstdev(robust_scores), 2),
            "robustness_stddev_efficiency": round(statistics.pstdev(eff_scores), 2),
            "trials": args.trials,
            "seeds": ",".join(str(s) for s in seeds),
            "success": any(t.get("success") for t in trial_results),
            "steps_taken": round(sum(t.get("steps_taken", 0) for t in trial_results) / len(trial_results), 2),
            "chain_length": args.chain_length,
            "decoy_count": args.decoy_count,
            "ambiguity_rate": args.ambiguity_rate,
            "noise_level": args.noise_level,
            "tool_failure_rate": args.tool_failure_rate,
            "permission_deny_rate": args.permission_deny_rate,
            "stream_rate": args.stream_rate,
            "api_rate_limit": args.api_rate_limit,
            "api_rate_window_sec": args.api_rate_window_sec,
            "tool_min_latency_ms": args.tool_min_latency_ms,
            "tool_max_latency_ms": args.tool_max_latency_ms,
            "tool_timeout_ms": args.tool_timeout_ms,
            "read_max_bytes": args.read_max_bytes,
            "read_truncate_rate": args.read_truncate_rate,
            "long_file_rate": args.long_file_rate,
            "concurrent_change_rate": args.concurrent_change_rate,
            "time_sec": round(sum(t.get("time_sec", 0) or 0 for t in trial_results) / len(trial_results), 3),
            "total_tokens": round(sum(t.get("total_tokens", 0) for t in trial_results) / len(trial_results), 2),
            "tokens_per_sec": round(sum(t.get("tokens_per_sec", 0) for t in trial_results) / len(trial_results), 2),
            "json_valid_rate": round(sum(t.get("json_valid_rate", 0) for t in trial_results) / len(trial_results), 3),
            "tool_valid_rate": round(sum(t.get("tool_valid_rate", 0) for t in trial_results) / len(trial_results), 3),
            "args_valid_rate": round(sum(t.get("args_valid_rate", 0) for t in trial_results) / len(trial_results), 3),
            "schema_valid_rate": round(sum(t.get("schema_valid_rate", 0) for t in trial_results) / len(trial_results), 3),
            "output_schema_valid_rate": round(sum(t.get("output_schema_valid_rate", 0) for t in trial_results) / len(trial_results), 3),
            "path_correct_rate": round(sum(t.get("path_correct_rate", 0) for t in trial_results) / len(trial_results), 3),
            "artifacts_access_before_unlock": round(sum(t.get("artifacts_access_before_unlock", 0) for t in trial_results) / len(trial_results), 2),
            "wrong_trace_reads": round(sum(t.get("wrong_trace_reads", 0) for t in trial_results) / len(trial_results), 2),
            "explore_over_budget_events": round(sum(t.get("explore_over_budget_events", 0) for t in trial_results) / len(trial_results), 2),
            "fork_present": round(sum(t.get("fork_present", 0) for t in trial_results) / len(trial_results), 2),
            "fork_wrong_branch_reads": round(sum(t.get("fork_wrong_branch_reads", 0) for t in trial_results) / len(trial_results), 2),
            "fork_dead_end_hit": round(sum(t.get("fork_dead_end_hit", 0) for t in trial_results) / len(trial_results), 2),
            "fork_recovered": round(sum(t.get("fork_recovered", 0) for t in trial_results) / len(trial_results), 2),
            "stream_sessions_started": round(sum(t.get("stream_sessions_started", 0) for t in trial_results) / len(trial_results), 2),
            "stream_sessions_completed": round(sum(t.get("stream_sessions_completed", 0) for t in trial_results) / len(trial_results), 2),
            "unique_files_read": round(sum(t.get("unique_files_read", 0) for t in trial_results) / len(trial_results), 2),
            "redundant_reads": round(sum(t.get("redundant_reads", 0) for t in trial_results) / len(trial_results), 2),
            "list_files_calls": round(sum(t.get("list_files_calls", 0) for t in trial_results) / len(trial_results), 2),
            "read_file_calls": round(sum(t.get("read_file_calls", 0) for t in trial_results) / len(trial_results), 2),
            "read_file_chunk_calls": round(sum(t.get("read_file_chunk_calls", 0) for t in trial_results) / len(trial_results), 2),
            "write_file_calls": round(sum(t.get("write_file_calls", 0) for t in trial_results) / len(trial_results), 2),
            "search_text_calls": round(sum(t.get("search_text_calls", 0) for t in trial_results) / len(trial_results), 2),
            "api_lookup_calls": round(sum(t.get("api_lookup_calls", 0) for t in trial_results) / len(trial_results), 2),
            "submit_attempts": round(sum(t.get("submit_attempts", 0) for t in trial_results) / len(trial_results), 2),
            "blocked_submit_attempts": round(sum(t.get("blocked_submit_attempts", 0) for t in trial_results) / len(trial_results), 2),
            "recovery_rate": round(sum(t.get("recovery_rate", 0) for t in trial_results) / len(trial_results), 3),
            "tool_failed_events": round(sum(t.get("tool_failed_events", 0) for t in trial_results) / len(trial_results), 2),
            "tool_error_recovery_success": round(sum(t.get("tool_error_recovery_success", 0) for t in trial_results) / len(trial_results), 2),
            "honeypot_hits": round(sum(t.get("honeypot_hits", 0) for t in trial_results) / len(trial_results), 2),
            "p50_time_sec": round(_quantile(time_vals, 0.5), 3),
            "p90_time_sec": round(_quantile(time_vals, 0.9), 3),
            "p50_steps": round(_quantile(steps_vals, 0.5), 2),
            "p90_steps": round(_quantile(steps_vals, 0.9), 2),
            "failure_reason": "",
            "last_tool_output": "",
        }
        results.append(agg)

        details = {
            "model": m,
            "benchmark_track": benchmark_track,
            "weights": {
                "task_success": weights[0],
                "protocol": weights[1],
                "robustness": weights[2],
                "efficiency": weights[3],
            },
            "aggregate": agg,
            "trials": [],
        }
        for i, t in enumerate(trial_results, start=1):
            trial_dir = os.path.join(out_dir, f"trial_{i:02d}")
            os.makedirs(trial_dir, exist_ok=True)
            with open(os.path.join(trial_dir, "details.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "model": m,
                    "time_sec": t.get("time_sec"),
                    "usage": {"total_tokens": t.get("total_tokens", 0)},
                    "result": t,
                }, f, indent=2, ensure_ascii=False)
            with open(os.path.join(trial_dir, "transcript.json"), "w", encoding="utf-8") as f:
                json.dump(t.get("transcript", []), f, indent=2, ensure_ascii=False)
            details["trials"].append({
                "trial": i,
                "seed": t.get("seed"),
                "success": t.get("success"),
                "steps_taken": t.get("steps_taken"),
                "scores": {
                    "task_success_100": t.get("task_success_100"),
                    "compliance_success_100": t.get("compliance_success_100"),
                    "instruction_fidelity_100": t.get("instruction_fidelity_100"),
                    "tool_discipline_100": t.get("tool_discipline_100"),
                    "robustness_100": t.get("robustness_100"),
                    "efficiency_100": t.get("efficiency_100"),
                    "overall_score_100": t.get("overall_score_100"),
                },
                "paths": {
                    "details": os.path.join(f"trial_{i:02d}", "details.json"),
                    "transcript": os.path.join(f"trial_{i:02d}", "transcript.json"),
                }
            })

        with open(os.path.join(out_dir, "details.json"), "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        results_file = os.path.join(out_dir, "results_longchain.csv")
        file_exists = os.path.exists(results_file)
        mode = "a" if run_args.append and file_exists else "w"
        with open(results_file, mode, newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if mode == "w":
                w.writeheader()
            for t in trial_results:
                w.writerow({k: t.get(k, "") for k in CSV_FIELDS})
        table_file = os.path.join(out_dir, "results_longchain.table.txt")
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(render_table(trial_results, CSV_FIELDS))
            print(
                f"\nRESULT: Success={agg['success']} Avg Steps={agg['steps_taken']} "
                f"Avg Time={agg.get('time_sec',0):.2f}s "
                f"p50/p90 Steps={agg.get('p50_steps')}/{agg.get('p90_steps')} "
                f"p50/p90 Time={agg.get('p50_time_sec')}/{agg.get('p90_time_sec')} "
                f"Overall={agg.get('overall_score_100')} "
                f"OverallStd={agg.get('robustness_stddev_overall')}"
            )
        
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    # Per-model CSVs are written alongside model details.
    if results:
        rank_fields = [
            "model",
            "benchmark_track",
            "overall_score_100",
            "task_success_100",
            "tool_discipline_100",
            "robustness_100",
            "efficiency_100",
            "steps_taken",
            "time_sec",
        ]
        by_track = {}
        for row in results:
            track = row.get("benchmark_track") or "unknown"
            by_track.setdefault(track, []).append(row)
        for track, rows in by_track.items():
            rows_sorted = sorted(rows, key=lambda r: (r.get("overall_score_100", 0)), reverse=True)
            table_file = os.path.join(args.out_dir, f"rankings_{track}.table.txt")
            with open(table_file, "w", encoding="utf-8") as f:
                f.write(render_table(rows_sorted, rank_fields))
            csv_file = os.path.join(args.out_dir, f"rankings_{track}.csv")
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rank_fields)
                w.writeheader()
                for row in rows_sorted:
                    w.writerow({k: row.get(k, "") for k in rank_fields})

if __name__ == "__main__":
    # SYNC_OK
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--models", help="Comma-separated model IDs")
    parser.add_argument("--active", action="store_true", help="Use active model")
    parser.add_argument("--chain-length", type=int, default=8, help="Number of hops in the file chain")
    parser.add_argument("--decoy-count", type=int, default=6, help="Number of decoy files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials with different seeds")
    parser.add_argument("--mode", choices=["compliance", "explore", "robust", "stream"], default="compliance", help="Preset configs for different benchmark modes")
    parser.add_argument("--benchmark-track", choices=["assisted", "realistic", "both"], default="realistic", help="Run assisted (debug) and/or realistic (unassisted) benchmarks")
    parser.add_argument("--w-task", dest="w_task", type=float, default=0.25, help="Weight for task success")
    parser.add_argument("--w-tool", dest="w_task", type=float, help="Alias for --w-task")
    parser.add_argument("--w-protocol", dest="w_protocol", type=float, default=0.25, help="Weight for protocol reliability")
    parser.add_argument("--w-core", dest="w_protocol", type=float, help="Alias for --w-protocol")
    parser.add_argument("--w-robust", dest="w_robust", type=float, default=0.25, help="Weight for robustness")
    parser.add_argument("--w-eff", dest="w_eff", type=float, default=0.25, help="Weight for efficiency")
    parser.add_argument("--ambiguity-rate", type=float, default=0.0, help="Chance to include a decoy path in clues")
    parser.add_argument("--noise-level", type=float, default=0.0, help="Chance to add noise around tool outputs")
    parser.add_argument("--noise-mode", choices=["debug", "realistic", "both"], default="debug", help="Noise style for tool outputs")
    parser.add_argument("--tool-failure-rate", type=float, default=0.0, help="Chance a tool call returns a transient error")
    parser.add_argument("--tool-min-latency-ms", type=int, default=0, help="Minimum tool latency in ms")
    parser.add_argument("--tool-max-latency-ms", type=int, default=0, help="Maximum tool latency in ms")
    parser.add_argument("--tool-timeout-ms", type=int, default=0, help="Tool timeout in ms (0 disables)")
    parser.add_argument("--read-max-bytes", type=int, default=4096, help="Max bytes before read_file truncates output")
    parser.add_argument("--read-truncate-rate", type=float, default=0.0, help="Chance to truncate even short reads")
    parser.add_argument("--long-file-rate", type=float, default=0.2, help="Chance a trace file is padded to be long")
    parser.add_argument("--concurrent-change-rate", type=float, default=0.05, help="Chance a file changes after first read")
    parser.add_argument("--max-chunk-reads-per-file", type=int, default=6, help="Limit chunk reads per file (0 disables)")
    parser.add_argument("--enable-hints", action=argparse.BooleanOptionalAction, default=False, help="Enable optional hint injections")
    parser.add_argument("--permission-deny-rate", type=float, default=0.0, help="Chance a decoy path is permission-denied")
    parser.add_argument("--stream-rate", type=float, default=0.0, help="Chance read_file returns a streamed chunk response")
    parser.add_argument("--reject-multi-tool", action=argparse.BooleanOptionalAction, default=True, help="Reject multiple tool calls in one turn")
    parser.add_argument("--api-rate-limit", type=int, default=0, help="Max API calls per window (0 disables)")
    parser.add_argument("--api-rate-window-sec", type=int, default=60, help="API rate limit window in seconds")
    parser.add_argument("--max-backoff-ms", type=int, default=4000, help="Max backoff for retries in ms")
    parser.add_argument("--allow-parallel-tools", action=argparse.BooleanOptionalAction, default=False, help="Allow multiple tool calls in one turn")
    parser.add_argument("--tool-debug", action="store_true", help="Print tool retry diagnostics")
    parser.add_argument("--auto-submit-on-secret", action=argparse.BooleanOptionalAction, default=False, help="Nudge submit_answer when secret is observed")
    parser.add_argument("--production-guards", action=argparse.BooleanOptionalAction, default=True, help="Enable production-style guardrails")
    parser.add_argument("--tool-retry-limit", type=int, default=2, help="Retries for transient tool failures")
    parser.add_argument("--retry-policy", choices=["none", "harness"], default="harness", help="Tool retry behavior")
    parser.add_argument("--max-parse-failures", type=int, default=4, help="Abort after this many tool-parse failures")
    parser.add_argument("--require-solution-file", action=argparse.BooleanOptionalAction, default=True, help="Require solution.txt before submit_answer")
    parser.add_argument("--require-checksum", action=argparse.BooleanOptionalAction, default=True, help="Require checksum verification before submit_answer")
    parser.add_argument("--follow-policy", choices=["hard", "soft", "none"], default="soft", help="Control expected-trace enforcement")
    parser.add_argument("--explore-budget-per-hop", type=int, default=4, help="Wrong-trace reads allowed per hop before penalties")
    parser.add_argument("--explore-budget-total", type=int, default=40, help="Total wrong-trace reads allowed before penalties")
    parser.add_argument("--strict-follow", action=argparse.BooleanOptionalAction, default=True, help="Block reads that deviate from expected next path")
    parser.add_argument("--strict-grace", type=int, default=1, help="Allow this many wrong reads before strict-follow blocks")
    parser.add_argument("--no-guess", action=argparse.BooleanOptionalAction, default=True, help="Block submit_answer before secret is found")
    parser.add_argument("--temperature", type=float, default=0.3, help="Low temp for tool use")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--stop", action="append", default=[], help="Stop sequences")
    parser.add_argument("--out-dir", default="advanced tool use bench", help="Output directory for results")
    parser.add_argument("--append", action="store_true", help="Append to existing results file if it exists")
    parser.add_argument("--self-check", action="store_true", help="Run a small VFS chunking sanity check and exit")
    parser.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True, help="Include baseline agents in the benchmark")
    
    # Watcher Arguments
    parser.add_argument("--watcher-model", default=None, help="Model ID for the Watcher supervisor")
    parser.add_argument("--watcher-base-url", default=None, help="Base URL for Watcher (defaults to --base-url)")
    parser.add_argument("--enable-watcher", action=argparse.BooleanOptionalAction, default=False, help="Enable Watcher intervention on deviation")
    parser.add_argument("--watcher-retry-cap", type=int, default=3, help="Max intervention retries per step/streak")
    
    args = parser.parse_args()
    if args.self_check:
        sys.exit(run_self_check())
    apply_mode_defaults(args)
    run_bench(args)
