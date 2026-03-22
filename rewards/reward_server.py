import json, re, os, subprocess, tempfile, random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

LABEL_MAP = {}
with open("/root/coderl_data/train.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        messages = json.loads(item["input"])
        for msg in messages:
            if msg["role"] == "user":
                key = msg["content"][:150]
                label = json.loads(item["label"])
                LABEL_MAP[key] = label.get("unit_tests", [])
                break
print(f"Loaded {len(LABEL_MAP)} mappings")

def extract_code(text):
    # 1. 先尝试 <answer> 标签
    matches = list(re.finditer(r'<answer>(.*?)</answer>', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    
    # 2. 尝试 markdown 代码块 ```python ... ```
    matches = list(re.finditer(r'```[Pp]ython\s*(.*?)```', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    
    # 3. 尝试普通代码块 ``` ... ```
    matches = list(re.finditer(r'```\s*(.*?)```', text, re.DOTALL))
    if matches:
        code = matches[-1].group(1).strip()
        if 'def ' in code or 'import ' in code or 'class ' in code:
            return code
    
    # 4. 直接找 def 开头的代码块（到 endoftext 或末尾）
    matches = list(re.finditer(r'((?:import .*\n)*def \w+.*?)(?:<\|endoftext\|>|\Z)', text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    
    return None

def run_unit_tests(code, unit_tests, timeout=10):
    passed = 0
    for test_code in unit_tests:
        full_code = code + "\n\n" + test_code
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

app = FastAPI()

@app.post("/get_reward")
async def get_reward(request: Request):
    data = await request.json()
    queries = data.get("query", [])
    rewards = []
    matched = 0
    extracted = 0
    tested = 0
    for i, query in enumerate(queries):
        code = extract_code(query)
        unit_tests = []
        for key, tests in LABEL_MAP.items():
            if key[:80] in query:
                unit_tests = tests
                matched += 1
                break
        
        if code is None:
            rewards.append(0.0)
            continue
        
        extracted += 1
        
        if not unit_tests:
            rewards.append(0.05)
            continue
        
        passed, total = run_unit_tests(code, unit_tests)
        tested += 1
        pass_rate = passed / total if total > 0 else 0.0
        # 更好的 reward 分级
        if pass_rate >= 0.8:
            reward = 1.0
        elif pass_rate > 0:
            reward = 0.3 + 0.7 * pass_rate
        else:
            reward = 0.1  # 代码能提取但没通过测试
        rewards.append(reward)
    
    if random.randint(1, 4) == 1:
        print(f"[Reward] batch={len(queries)}, matched={matched}, extracted={extracted}, tested={tested}, mean={sum(rewards)/len(rewards):.3f}")
    
    return JSONResponse(content={"rewards": rewards})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)
