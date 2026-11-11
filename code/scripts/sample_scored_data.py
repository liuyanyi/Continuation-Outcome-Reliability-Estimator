#!/usr/bin/env python3
"""Sample scored data based on scores: all-correct, all-wrong, and diverse."""

import argparse
import random
from pathlib import Path

import orjson
from tqdm import tqdm


def load_json_file(file_path: str) -> dict:
    """Load a JSON file."""
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def is_positive(scores: list[bool]) -> bool:
    """Check if accuracy > 90%."""
    if not scores:
        return False
    return sum(scores) / len(scores) > 0.9


def is_negative(scores: list[bool]) -> bool:
    """Check if accuracy < 10%."""
    if not scores:
        return False
    return sum(scores) / len(scores) < 0.1


def calculate_score_ratio(scores: list[bool]) -> float:
    """Calculate ratio of True scores."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def sample_data(
    input_dir: str,
    output_dir: str,
    num_positive: int = 100,
    num_negative: int = 100,
    num_diverse: int = 200,
    test_positive: int = 10,
    test_negative: int = 10,
    test_diverse: int = 20,
    pattern: str = "*.json",
    seed: int = 42,
) -> dict[str, dict[str, list[dict]]]:
    """
    Sample data from input directory based on accuracy.

    Args:
        input_dir: Directory containing scored JSON files
        output_dir: Output directory for sampled data
        num_positive: Number of positive samples for train (accuracy > 90%)
        num_negative: Number of negative samples for train (accuracy < 10%)
        num_diverse: Number of diverse samples for train (normal distribution around 50%)
        test_positive: Number of positive samples for test
        test_negative: Number of negative samples for test
        test_diverse: Number of diverse samples for test
        pattern: Glob pattern for input files
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train' and 'test', each containing 'positive', 'negative', 'diverse' samples
    """
    random.seed(seed)
    input_path = Path(input_dir)
    all_files = list(input_path.glob(pattern))

    if not all_files:
        print(f"No files found matching {pattern} in {input_dir}")
        return {"train": {"positive": [], "negative": [], "diverse": []},
                "test": {"positive": [], "negative": [], "diverse": []}}

    print(f"Found {len(all_files)} files to process")

    # Categories
    positive: list[dict] = []
    negative: list[dict] = []
    diverse: list[dict] = []

    # Track score ratios for diverse sampling
    diverse_pool: list[tuple[float, dict]] = []

    # First pass: categorize all files
    print("\nCategorizing files...")
    for file_path in tqdm(all_files):
        try:
            data = load_json_file(str(file_path))
            scores = data.get("scores", [])

            if not scores:
                continue

            ratio = calculate_score_ratio(scores)

            if is_positive(scores):
                positive.append(data)
            elif is_negative(scores):
                negative.append(data)
            else:
                # For diverse samples, track by score ratio
                diverse_pool.append((ratio, data))
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")

    print("\nCategorization complete:")
    print(f"  Positive (>90%): {len(positive)} files")
    print(f"  Negative (<10%): {len(negative)} files")
    print(f"  Diverse candidates (10%-90%): {len(diverse_pool)} files")

    # Sample positive
    total_positive_needed = num_positive + test_positive
    if len(positive) >= total_positive_needed:
        sampled_positive_all = random.sample(positive, total_positive_needed)
        test_positive_samples = sampled_positive_all[:test_positive]
        train_positive_samples = sampled_positive_all[test_positive:]
        print(f"\n✓ Sampled {len(test_positive_samples)} test + {len(train_positive_samples)} train positive files")
    elif len(positive) >= test_positive:
        test_positive_samples = random.sample(positive, test_positive)
        train_positive_samples = [p for p in positive if p not in test_positive_samples][:num_positive]
        print(f"\n⚠ Only {len(positive)} positive files available (requested {total_positive_needed})")
        print(f"  Test: {len(test_positive_samples)}, Train: {len(train_positive_samples)}")
    else:
        test_positive_samples = positive
        train_positive_samples = []
        print(f"\n⚠ Only {len(positive)} positive files available for test (requested {test_positive})")

    # Sample negative
    total_negative_needed = num_negative + test_negative
    if len(negative) >= total_negative_needed:
        sampled_negative_all = random.sample(negative, total_negative_needed)
        test_negative_samples = sampled_negative_all[:test_negative]
        train_negative_samples = sampled_negative_all[test_negative:]
        print(f"✓ Sampled {len(test_negative_samples)} test + {len(train_negative_samples)} train negative files")
    elif len(negative) >= test_negative:
        test_negative_samples = random.sample(negative, test_negative)
        train_negative_samples = [n for n in negative if n not in test_negative_samples][:num_negative]
        print(f"⚠ Only {len(negative)} negative files available (requested {total_negative_needed})")
        print(f"  Test: {len(test_negative_samples)}, Train: {len(train_negative_samples)}")
    else:
        test_negative_samples = negative
        train_negative_samples = []
        print(f"\n⚠ Only {len(negative)} negative files available for test (requested {test_negative})")

    # Sample diverse using normal distribution (biased toward 50%)
    total_diverse_needed = num_diverse + test_diverse
    if len(diverse_pool) >= total_diverse_needed:
        sampled_diverse_all: list[dict] = []

        # Sort diverse_pool by ratio for binary search
        diverse_pool.sort(key=lambda x: x[0])
        ratios = [item[0] for item in diverse_pool]

        # Track sampled indices to avoid duplicates
        sampled_indices = set()

        # Sample using normal distribution centered at 0.5
        import bisect
        for _ in range(total_diverse_needed):
            # Generate target ratio from normal distribution (mean=0.5, std=0.15)
            # Clip to [0.1, 0.9] to stay within diverse range
            target_ratio = random.gauss(0.5, 0.15)
            target_ratio = max(0.1, min(0.9, target_ratio))

            # Find closest item using binary search
            idx = bisect.bisect_left(ratios, target_ratio)

            # Choose closest among nearby indices, avoiding already sampled
            best_idx = None
            best_dist = float('inf')
            for offset in [-3, -2, -1, 0, 1, 2, 3]:
                check_idx = idx + offset
                if 0 <= check_idx < len(ratios) and check_idx not in sampled_indices:
                    dist = abs(ratios[check_idx] - target_ratio)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = check_idx

            # Pick best and add to samples
            if best_idx is not None:
                sampled_indices.add(best_idx)
                sampled_diverse_all.append(diverse_pool[best_idx][1])

        test_diverse_samples = sampled_diverse_all[:test_diverse]
        train_diverse_samples = sampled_diverse_all[test_diverse:]
        print(f"✓ Sampled {len(test_diverse_samples)} test + {len(train_diverse_samples)} train diverse files")
    else:
        test_diverse_samples = [item[1] for item in diverse_pool][:test_diverse]
        train_diverse_samples = [item[1] for item in diverse_pool][test_diverse:num_diverse]
        print(f"⚠ Only {len(diverse_pool)} diverse files available (requested {total_diverse_needed})")
        print(f"  Test: {len(test_diverse_samples)}, Train: {len(train_diverse_samples)}")

    # Print score ratio statistics for diverse samples
    for split_name, diverse_split in [("train", train_diverse_samples), ("test", test_diverse_samples)]:
        if diverse_split:
            ratios = [calculate_score_ratio(d.get("scores", [])) for d in diverse_split]
            print(f"\n{split_name.capitalize()} diverse score ratio statistics:")
            print(f"  Min: {min(ratios):.3f}")
            print(f"  Max: {max(ratios):.3f}")
            print(f"  Mean: {sum(ratios) / len(ratios):.3f}")

    # Save samples - each sample as individual file (same format as input)
    # Use flat structure with category in filename
    output_path = Path(output_dir)

    samples = {
        "train": {
            "positive": train_positive_samples,
            "negative": train_negative_samples,
            "diverse": train_diverse_samples,
        },
        "test": {
            "positive": test_positive_samples,
            "negative": test_negative_samples,
            "diverse": test_diverse_samples,
        },
    }

    for split_name, split_data in samples.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for category, data_list in split_data.items():
            for data in tqdm(data_list, desc=f"Saving {split_name}/{category}"):
                # Use original filename from 'id' field if available
                file_id = data.get("id", f"sample_{random.randint(1000, 9999)}")
                output_file = split_dir / f"{file_id}.{category}.json"
                output_file.write_bytes(orjson.dumps(data))

            print(f"✓ Saved {len(data_list)} {split_name}/{category} samples")

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Train: {len(train_positive_samples)} pos + {len(train_negative_samples)} neg + {len(train_diverse_samples)} diverse = {len(train_positive_samples) + len(train_negative_samples) + len(train_diverse_samples)} total")
    print(f"  Test:  {len(test_positive_samples)} pos + {len(test_negative_samples)} neg + {len(test_diverse_samples)} diverse = {len(test_positive_samples) + len(test_negative_samples) + len(test_diverse_samples)} total")
    print(f"{'=' * 60}")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample scored data based on correctness patterns"
    )
    parser.add_argument(
        "input",
        help="Input directory containing scored JSON files",
    )
    parser.add_argument(
        "output",
        help="Output directory for sampled data files",
    )
    parser.add_argument(
        "--num-positive",
        type=int,
        default=100,
        help="Number of positive samples for train (accuracy > 90%%)",
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=100,
        help="Number of negative samples for train (accuracy < 10%%)",
    )
    parser.add_argument(
        "--num-diverse",
        type=int,
        default=200,
        help="Number of diverse samples for train (normal distribution around 50%%)",
    )
    parser.add_argument(
        "--test-positive",
        type=int,
        default=10,
        help="Number of positive samples for test (accuracy > 90%%)",
    )
    parser.add_argument(
        "--test-negative",
        type=int,
        default=10,
        help="Number of negative samples for test (accuracy < 10%%)",
    )
    parser.add_argument(
        "--test-diverse",
        type=int,
        default=20,
        help="Number of diverse samples for test (normal distribution around 50%%)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern for matching input files (default: *.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate counts
    if any(x < 0 for x in [args.num_positive, args.num_negative, args.num_diverse,
                          args.test_positive, args.test_negative, args.test_diverse]):
        print("Error: Sample counts must be non-negative")
        exit(1)

    sample_data(
        args.input,
        args.output,
        args.num_positive,
        args.num_negative,
        args.num_diverse,
        args.test_positive,
        args.test_negative,
        args.test_diverse,
        args.pattern,
        args.seed,
    )
