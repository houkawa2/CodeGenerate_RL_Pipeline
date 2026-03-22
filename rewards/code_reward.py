import re, os, json, random, subprocess, tempfile, torch
from typing import List
def extract_code(response):
    matches = list(re.finditer(r'<answer>(.*?)</answer>', response, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None
def extract_response(query, prompt):
    if prompt and query.startswith(prompt):
        return query[len(prompt):]
    if "Assistant:" in query:
        return query.split("Assistant:", 1)[1]
    return query
def run_unit_tests(code, unit_tests, timeout=10):
    passed = 0
    for test_code in unit_tests:
        full_code = code + "\n\n" + test_code + "\n\nif __name__ == '__main__':\n    import unittest\n    unittest.main()\n"
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
                f.write(full_code)
                tmp_path = f.name
            result = subprocess.run(["python", tmp_path], timeout=timeout, capture_output=True, text=True)
            if result.returncode == 0:
                passed += 1
        except:
            pass
        finally:
            try: os.unlink(tmp_path)
            except: pass
    return passed, len(unit_tests)
def reward_func(queries, prompts, labels):
    batch_size = len(queries)
    rewards = []
    scores = []
    total_passed = 0
    total_tests = 0
    for i in range(batch_size):
        response = extract_response(queries[i], prompts[i])
        code = extract_code(response)
        try:
            ground_truth = json.loads(labels[i])
            unit_tests = ground_truth.get("unit_tests", [])
        except:
            unit_tests = []
        if code is None:
            rewards.append(0.1)
            scores.append(0.0)
            continue
        passed, total = run_unit_tests(code, unit_tests)
        total_passed += passed
        total_tests += total
        pass_rate = passed / total if total > 0 else 0.0
        reward = pass_rate if pass_rate > 0 else 0.1
        rewards.append(reward)
        scores.append(pass_rate)
    if random.randint(1, 16) == 1:
        idx = random.randint(0, batch_size - 1)
        print(f"[Reward] sample {idx}: reward={rewards[idx]:.3f}")
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "extra_logs": {
            "reward/mean": sum(rewards) / len(rewards),
            "test/pass_rate": total_passed / max(total_tests, 1),
        },
    }
