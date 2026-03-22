"""
Data Preprocessing: Convert KAKA22/CodeRM-UnitTest parquet files to OpenRLHF JSONL format.

Usage:
    python data/preprocess.py --input_dir /path/to/parquet_files --output_dir /path/to/output
"""
import argparse, json, os, random
import pandas as pd

SYSTEM_PROMPT = (
    "You are a Python code generation assistant. Think step by step inside "
    "<think>...</think> tags, then provide your final Python code inside "
    "<answer>...</answer> tags."
)

def process_parquet(input_dir, output_dir, test_ratio=0.1, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    samples = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith('.parquet'):
            continue
        path = os.path.join(input_dir, fname)
        df = pd.read_parquet(path)
        print(f"Loading {fname}: {len(df)} rows, columns: {list(df.columns)}")

        for _, row in df.iterrows():
            problem = str(row.get("problem", row.get("prompt", "")))
            unit_tests = row.get("unit_tests", row.get("test_cases", []))
            if isinstance(unit_tests, str):
                try:
                    unit_tests = json.loads(unit_tests)
                except json.JSONDecodeError:
                    unit_tests = [unit_tests]
            if not problem or not unit_tests:
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ]
            label = json.dumps({"unit_tests": unit_tests, "problem": problem})
            samples.append({"input": json.dumps(messages), "label": label})

    print(f"\nTotal valid samples: {len(samples)}")
    random.shuffle(samples)
    split = int(len(samples) * (1 - test_ratio))

    for path, data in [
        (os.path.join(output_dir, "train.jsonl"), samples[:split]),
        (os.path.join(output_dir, "test.jsonl"), samples[split:]),
    ]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Written {len(data)} samples to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    process_parquet(args.input_dir, args.output_dir, args.test_ratio)
