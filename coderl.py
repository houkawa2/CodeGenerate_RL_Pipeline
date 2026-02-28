import re, random, subprocess, tempfile, os, json

def extract_code(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None

def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    unit_tests = ground_truth['unit_tests']
    code = extract_code(solution_str)
    do_print = random.randint(1, 32) == 1
    if code is None:
        return 0
    passed = 0
    for test_code in unit_tests:
        full_code = code + "\n\n" + test_code + "\n\nif __name__ == '__main__':\n    unittest.main()"
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                tmp = f.name
            result = subprocess.run(["python", tmp], timeout=5, capture_output=True)
            os.unlink(tmp)
            if result.returncode == 0:
                passed += 1
        except:
            pass
    reward = passed / len(unit_tests) if len(unit_tests) > 0 else 0
    if do_print:
        print(f"Passed {passed}/{len(unit_tests)} tests, reward={reward}")
    return reward if reward > 0 else format_score
