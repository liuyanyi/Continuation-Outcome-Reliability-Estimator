import array
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import orjson
import xxhash
from tqdm import tqdm
from transformers import AutoTokenizer

from cognitive_probe.utils.math_utils import extract_answer, grade_answer_verl

# Global state for graceful shutdown
_shutdown_event = None
_executor_pool: ThreadPoolExecutor | None = None
_debug_mode = False

# Lazy load tokenizer (thread-safe, loaded in worker processes)
_tokenizer = None


def debug_print(msg: str):
    """Print debug message if debug mode is enabled."""
    if _debug_mode:
        print(f"[DEBUG] {msg}", file=sys.stderr)


class Timer:
    """Simple timer context manager for debugging."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        debug_print(f"Starting: {self.name}")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time # type: ignore
        debug_print(f"Finished: {self.name} ({elapsed:.2f}s)")


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(os.environ.get("PROBE_TOKENIZER_PATH", "Qwen/Qwen3-4B-Instruct-2507"))
    return _tokenizer


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"\nReceived signal {signum}, shutting down gracefully...", file=sys.stderr)
    global _shutdown_event, _executor_pool

    if _shutdown_event is not None:
        _shutdown_event.set()

    if _executor_pool is not None:
        _executor_pool.shutdown(wait=False, cancel_futures=True)

    sys.exit(1)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Configuration for concurrent processing
MAX_WORKERS = os.environ.get("MAX_WORKERS", None)  # None = CPU count


@dataclass
class NFResult:
    input_ids: list[int]
    token_contributions: list[float]
    contrib_weights: list[float]
    total_count: int


def correct_ratio(all_result, idx):
    idx_results = [res[idx] for res in all_result]
    correct_count = sum(1 for res in idx_results if res > 0.5)
    return (
        correct_count,
        len(idx_results),
        correct_count / len(idx_results) if len(idx_results) > 0 else 0.0,
    )


@lru_cache(maxsize=1000000)
def hash_func(tokens_tuple: tuple) -> str:
    """Hash a sequence of tokens using xxhash with LRU cache."""
    arr = array.array("q", tokens_tuple)
    return xxhash.xxh64(arr).hexdigest()


def hash_func_list(tokens: list[int]) -> str:
    """Wrapper to cache hash computations for lists."""
    return hash_func(tuple(tokens))


def precompute_hashes(token_ids: list[int]) -> list[str]:
    """
    Precompute all rolling hashes using TRUE incremental hashing.
    This uses xxhash's streaming API for O(1) updates per token.
    """
    hashes = []
    x = xxhash.xxh64()
    for token_id in token_ids:
        x.update(array.array("q", [token_id]))
        hashes.append(x.hexdigest())
    return hashes


def _process_sequence_for_map(args: tuple) -> dict:
    """
    Worker function: Process a single sequence to build token map entries.
    Returns a dict mapping positions to token_keys for this sequence.
    """
    seq_idx, token_ids = args
    seq_entries = {}

    # Precompute all hashes for this sequence in one pass
    hashes = precompute_hashes(token_ids)

    # Build entries using precomputed hashes (no tuple concatenation in loop!)
    for pos, (token_id, hash_val) in enumerate(zip(token_ids, hashes)):
        token_key = f"{token_id}:{hash_val}"
        if pos not in seq_entries:
            seq_entries[pos] = {}
        seq_entries[pos][token_key] = seq_idx

    return seq_entries


def build_position_token_map_parallel(
    all_tokenized: list[list[int]], max_workers: int = 1, progress: bool = True
) -> list[dict[str, list[int]]]:
    """
    Build position_token_map using parallel processing.
    Each worker processes a subset of sequences independently, then results are merged.
    """
    global _executor_pool
    # Create task arguments: (seq_idx, token_ids)
    tasks = list(enumerate(all_tokenized))

    position_token_map: list[dict[str, list[int]]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _executor_pool = executor
        # Submit all tasks
        futures = [executor.submit(_process_sequence_for_map, task) for task in tasks]

        # Merge results as they complete
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Building token map (parallel)",
            disable=not progress,
        ):
            seq_entries = future.result()

            # Merge this sequence's entries into the global map
            for pos, token_dict in seq_entries.items():
                # Ensure position_token_map has enough positions
                while len(position_token_map) <= pos:
                    position_token_map.append({})

                pos_map = position_token_map[pos]
                for token_key, seq_idx in token_dict.items():
                    if token_key not in pos_map:
                        pos_map[token_key] = []
                    pos_map[token_key].append(seq_idx)
        _executor_pool = None

    return position_token_map


def _compute_nf_result_for_sequence(args: tuple) -> "NFResult":
    """
    Worker function: Compute NF result for a single sequence.
    """
    seq_idx, token_ids, position_token_map, all_results, total_sequences = args

    seq_len = len(token_ids)
    seq_token_contribs: list[tuple[float, int]] = [(0.0, 0)] * seq_len

    # Precompute all hashes for this sequence
    hashes = precompute_hashes(token_ids)

    # Use precomputed hashes (no tuple concatenation in loop!)
    for pos, (token_id, hash_val) in enumerate(zip(token_ids, hashes)):
        current_token_key = f"{token_id}:{hash_val}"
        token_seq_indices = position_token_map[pos].get(current_token_key, [])

        if not token_seq_indices:
            # Token not found - skip or use default
            seq_token_contribs[pos] = (0.0, 0)
            continue

        # Calculate average contribution for this token
        contrib_avg = sum(all_results[i] for i in token_seq_indices) / len(
            token_seq_indices
        )
        contrib_count = len(token_seq_indices)
        seq_token_contribs[pos] = (contrib_avg, contrib_count)

    # Prepare result
    contrib_avgs = [c[0] for c in seq_token_contribs]
    contrib_counts = [c[1] for c in seq_token_contribs]
    contrib_weights = [
        c / total_sequences if total_sequences > 0 else 0.0 for c in contrib_counts
    ]

    return NFResult(
        input_ids=token_ids,
        token_contributions=contrib_avgs,
        contrib_weights=contrib_weights,
        total_count=total_sequences,
    )


def compute_nf_results_parallel(
    all_tokenized: list[list[int]],
    position_token_map: list[dict[str, list[int]]],
    all_results: list[float],
    max_workers: int = 1,
    progress: bool = True,
) -> list:
    """
    Compute NF results for all sequences in parallel.
    """
    global _executor_pool
    total_sequences = len(all_tokenized)
    tasks = [
        (seq_idx, token_ids, position_token_map, all_results, total_sequences)
        for seq_idx, token_ids in enumerate(all_tokenized)
    ]

    nf_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _executor_pool = executor
        futures = {
            executor.submit(_compute_nf_result_for_sequence, task): i
            for i, task in enumerate(tasks)
        }

        # Collect results in original order
        results_by_index = {}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing NF results (parallel)",
            disable=not progress,
        ):
            idx = futures[future]
            results_by_index[idx] = future.result()

        # Sort by index to maintain order
        nf_results = [results_by_index[i] for i in range(len(results_by_index))]
        _executor_pool = None

    return nf_results


def extract_outputs(data: dict) -> tuple[list[dict], list[str]]:
    """Extract branch samples and outputs from new tree generation format: branches[].text."""
    branches = data.get("branches")
    if not isinstance(branches, list):
        raise ValueError("Invalid format: missing `branches` list")

    branch_samples: list[dict] = []
    extracted_outputs: list[str] = []
    for idx, branch in enumerate(branches):
        if not isinstance(branch, dict):
            raise ValueError(f"Invalid format: branches[{idx}] is not an object")
        branch_samples.append(branch)
        text = branch.get("text", "")
        extracted_outputs.append(text if isinstance(text, str) else str(text))

    return branch_samples, extracted_outputs


def has_complete_scores(data: dict) -> bool:
    branches = data.get("branches")
    if not isinstance(branches, list) or not branches:
        return False
    return all(isinstance(branch, dict) and "score" in branch for branch in branches)


def attach_scores(
    data: dict,
    branch_samples: list[dict],
    outputs: list[str],
) -> tuple[list[float], list[str], str]:
    raw_ground_truth = data.get("answer", "")
    if isinstance(raw_ground_truth, str) and "\\boxed" in raw_ground_truth:
        ground_truth_answer = extract_answer(raw_ground_truth)
    else:
        ground_truth_answer = raw_ground_truth

    all_results = []
    extracted_answers = []
    for item in outputs:
        extracted_answers.append(extract_answer(item))
        score = grade_answer_verl(item, raw_ground_truth)
        all_results.append(score)

    accuracy = (
        sum(1 for score in all_results if score > 0.5) / len(all_results)
        if all_results
        else 0.0
    )
    data["accuracy"] = accuracy
    data["scores"] = all_results
    data["stats"] = {
        "avg_score": sum(all_results) / len(all_results) if all_results else 0.0,
        "accuracy": accuracy,
    }

    for branch, score, extracted_answer in zip(
        branch_samples, all_results, extracted_answers
    ):
        branch["score"] = score
        branch["extracted_answer"] = extracted_answer
        branch["ground_truth_answer"] = ground_truth_answer

    return all_results, extracted_answers, ground_truth_answer


def write_json_file(input_file: str, data: dict) -> None:
    with Timer("write_file"), open(input_file, "w") as f:
        f.write(
            orjson.dumps(
                data,
                option=orjson.OPT_INDENT_2,
            ).decode("utf-8")
        )


def process_single_file(
    input_file: str,
    max_workers: int = 1,
    progress: bool = True,
    score_only: bool = False,
    skip_scored: bool = False,
):
    """
    Process a single JSON file to compute scores and NF results.

    Args:
        input_file: Path to input JSON file
        max_workers: If >1, enable within-file parallelism; otherwise sequential
    """
    with Timer("load_file"), open(input_file) as f:
        data = orjson.loads(f.read())

    if skip_scored and has_complete_scores(data):
        debug_print(f"Skipping already scored file: {input_file}")
        return

    branch_samples, outputs = extract_outputs(data)
    data["outputs"] = outputs
    debug_print(f"Outputs: {len(outputs)}")

    # Score each output
    with Timer("scoring"):
        all_results, extracted_answers, ground_truth_answer = attach_scores(
            data, branch_samples, outputs
        )

    if score_only:
        write_json_file(input_file, data)
        return

    # Tokenize all outputs
    with Timer("tokenization"):
        all_tokenized: list[list[int]] = get_tokenizer()(outputs)["input_ids"]
    debug_print(f"Tokenized: {len(all_tokenized)} sequences")

    # Build position-token map
    with Timer("build_token_map"):
        if max_workers and max_workers > 1:
            position_token_map = build_position_token_map_parallel(
                all_tokenized, max_workers=max_workers, progress=progress
            )
        else:
            # Sequential version - removed nested progress bars for better performance
            position_token_map: list[dict[str, list[int]]] = []
            total_hash_time = 0
            total_dict_time = 0
            for seq_idx, token_ids in tqdm(
                enumerate(all_tokenized),
                desc="Building token map",
                total=len(all_tokenized),
                disable=not progress or not _debug_mode,  # Only show in debug mode
            ):
                # Precompute all hashes for this sequence
                t0 = time.time()
                hashes = precompute_hashes(token_ids)
                total_hash_time += time.time() - t0

                # Build entries using precomputed hashes (no tuple concatenation!)
                t0 = time.time()
                for pos, (token_id, hash_val) in enumerate(zip(token_ids, hashes)):
                    while len(position_token_map) <= pos:
                        position_token_map.append({})
                    pos_map = position_token_map[pos]
                    token_key = f"{token_id}:{hash_val}"
                    if token_key not in pos_map:
                        pos_map[token_key] = []
                    pos_map[token_key].append(seq_idx)
                total_dict_time += time.time() - t0

            if _debug_mode:
                debug_print(f"  Breakdown - hash computation: {total_hash_time:.2f}s, dict ops: {total_dict_time:.2f}s")
    debug_print(f"Token map positions: {len(position_token_map)}")

    # Compute NF results
    with Timer("compute_nf_results"):
        if max_workers and max_workers > 1:
            nf_results = compute_nf_results_parallel(
                all_tokenized,
                position_token_map,
                all_results,
                max_workers=max_workers,
                progress=progress,
            )
        else:
            # Sequential version - removed nested progress bars for better performance
            nf_results = []
            total_count = len(all_tokenized)
            for seq_idx, token_ids in tqdm(
                enumerate(all_tokenized),
                desc="Computing NF results",
                total=len(all_tokenized),
                disable=not progress or not _debug_mode,  # Only show in debug mode
            ):
                seq_len = len(token_ids)
                seq_token_contribs: list[tuple[float, int]] = [(0.0, 0)] * seq_len

                # Precompute all hashes for this sequence
                hashes = precompute_hashes(token_ids)

                # Use precomputed hashes (no tuple concatenation!)
                for pos, (token_id, hash_val) in enumerate(zip(token_ids, hashes)):
                    current_token_key = f"{token_id}:{hash_val}"
                    token_seq_indices = position_token_map[pos].get(current_token_key, [])
                    if not token_seq_indices:
                        seq_token_contribs[pos] = (0.0, 0)
                        continue
                    contrib_avg = sum(all_results[i] for i in token_seq_indices) / len(
                        token_seq_indices
                    )
                    contrib_count = len(token_seq_indices)
                    seq_token_contribs[pos] = (contrib_avg, contrib_count)

                contrib_avgs = [c[0] for c in seq_token_contribs]
                contrib_counts = [c[1] for c in seq_token_contribs]
                contrib_weights = [
                    c / total_count if total_count > 0 else 0.0 for c in contrib_counts
                ]
                nf_results.append(
                    NFResult(
                        input_ids=token_ids,
                        token_contributions=contrib_avgs,
                        contrib_weights=contrib_weights,
                        total_count=total_count,
                    )
                )

    # Serialize and attach results to each branch sample
    with Timer("serialization"):
        serialized_nf_results = [
            {
                "input_ids": orjson.dumps(res.input_ids).decode("utf-8"),
                "token_contributions": orjson.dumps(res.token_contributions).decode(
                    "utf-8"
                ),
                "contrib_weights": orjson.dumps(res.contrib_weights).decode("utf-8"),
                "total_count": res.total_count,
            }
            for res in nf_results
        ]

        if len(branch_samples) != len(serialized_nf_results):
            raise ValueError(
                f"Mismatch: {len(branch_samples)} branches vs {len(serialized_nf_results)} nf_results"
            )

        for branch, score, nf_result, extracted_answer in zip(
            branch_samples, all_results, serialized_nf_results, extracted_answers
        ):
            branch["score"] = score
            branch["nf_result"] = nf_result
            branch["extracted_answer"] = extracted_answer
            branch["ground_truth_answer"] = ground_truth_answer

    # Write back to file
    write_json_file(input_file, data)


def process_batch_files(
    input_dir: str,
    pattern: str = "*.json",
    max_workers: int = 1,
    max_file_workers: int = 1,
    max_file_processes: int = 0,
    score_only: bool = False,
    skip_scored: bool = False,
):
    """
    Process multiple JSON files in a directory.

    Args:
        input_dir: Directory containing JSON files to process
        pattern: Glob pattern for files (default: "*.json")
        max_workers: If >1, enable within-file parallelism; otherwise sequential
        max_file_workers: If >1, process multiple files concurrently; otherwise sequential
        max_file_processes: If >0, process files in separate processes
    """
    result_dir = Path(input_dir)
    all_files = list(result_dir.glob(pattern))

    if not all_files:
        raise FileNotFoundError(f"No files found matching {pattern} in {input_dir}")

    print(f"Found {len(all_files)} files to process")
    failures = []

    if max_file_processes and max_file_processes > 0:
        with ProcessPoolExecutor(max_workers=max_file_processes) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    str(file_path),
                    max_workers,
                    False,
                    score_only,
                    skip_scored,
                ): file_path
                for file_path in all_files
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                file_path = futures[future]
                try:
                    future.result()
                    tqdm.write(f"✓ Completed: {file_path.name}")
                except Exception as e:
                    failures.append(str(file_path))
                    tqdm.write(f"✗ Error processing {file_path.name}: {e}")
    elif max_file_workers and max_file_workers > 1:
        # Process files concurrently using ThreadPoolExecutor (I/O-bound)
        with ThreadPoolExecutor(max_workers=max_file_workers) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    str(file_path),
                    max_workers,
                    False,
                    score_only,
                    skip_scored,
                ): file_path
                for file_path in all_files
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                file_path = futures[future]
                try:
                    future.result()
                    tqdm.write(f"✓ Completed: {file_path.name}")
                except Exception as e:
                    failures.append(str(file_path))
                    tqdm.write(f"✗ Error processing {file_path.name}: {e}")
    else:
        # Process files sequentially
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                process_single_file(
                    str(file_path),
                    max_workers=max_workers,
                    progress=True,
                    score_only=score_only,
                    skip_scored=skip_scored,
                )
                tqdm.write(f"✓ Completed: {file_path.name}")
            except Exception as e:
                failures.append(str(file_path))
                tqdm.write(f"✗ Error processing {file_path.name}: {e}")

    if failures:
        raise RuntimeError(f"Scoring failed for {len(failures)} file(s): {failures}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score batch files with optional parallel processing"
    )
    parser.add_argument("input", help="Input file or directory")
    # Parallelism toggled by worker counts (>1 enables parallelism)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Workers for within-file parallelism (>1 enables parallel)",
    )
    parser.add_argument(
        "--max-file-workers",
        type=int,
        default=1,
        help="Workers for file-level parallelism (>1 enables concurrent files)",
    )
    parser.add_argument(
        "--max-file-processes",
        type=int,
        default=0,
        help="Processes for file-level parallelism; overrides --max-file-workers when >0",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern for batch processing (default: *.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with timing information",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Only compute branch scores; skip legacy nf_result token maps",
    )
    parser.add_argument(
        "--skip-scored",
        action="store_true",
        help="Skip files where every branch already has a score",
    )

    parser.add_argument("--tokenizer-path", default=os.environ.get("PROBE_TOKENIZER_PATH", "Qwen/Qwen3-4B-Instruct-2507"), help="Tokenizer for legacy token maps; unused with --score-only")
    args = parser.parse_args()
    os.environ["PROBE_TOKENIZER_PATH"] = args.tokenizer_path

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Set global debug mode (module level, no need for global keyword)
    _debug_mode = args.debug

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file mode
        print(f"Processing single file: {input_path}")
        process_single_file(
            str(input_path),
            max_workers=args.max_workers,
            score_only=args.score_only,
            skip_scored=args.skip_scored,
        )
    elif input_path.is_dir():
        # Batch mode
        print(f"Processing directory: {input_path}")
        process_batch_files(
            str(input_path),
            pattern=args.pattern,
            max_workers=args.max_workers,
            max_file_workers=args.max_file_workers,
            max_file_processes=args.max_file_processes,
            score_only=args.score_only,
            skip_scored=args.skip_scored,
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
