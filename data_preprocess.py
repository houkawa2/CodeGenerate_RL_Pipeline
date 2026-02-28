import os, json
import pandas as pd
from datasets import load_dataset

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def make_prefix(question):
    return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Write a Python function to solve the following problem. Show your reasoning in <think> </think> tags. Return only the function code in <answer> </answer> tags.

{question}
Assistant: Let me solve this step by step.
<think>"""

ds = load_dataset("KAKA22/CodeRM-UnitTest", split="train")
ds = ds.select(range(3000))

def process(item, idx):
    tests = json.loads(item['unit_tests'])
    top_tests = [t['code'] for t in tests[:3]]
    return {
        "data_source": "humaneval",
        "prompt": [{"role": "user", "content": make_prefix(item['question'])}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "unit_tests": top_tests,
                "ground_truth_code": item['code_ground_truth']
            }
        },
        "extra_info": {"split": "train", "index": idx}
    }

data = [process(ds[i], i) for i in range(len(ds))]
train = data[:2500]
test = data[2500:]

os.makedirs("/root/TinyZero/data/coderl", exist_ok=True)
pd.DataFrame(train).to_parquet("/root/TinyZero/data/coderl/train.parquet")
pd.DataFrame(test).to_parquet("/root/TinyZero/data/coderl/test.parquet")
print(f"Done: {len(train)} train, {len(test)} test")
