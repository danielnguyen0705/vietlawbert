import json

INPUT_BENCHMARK = "/home/dell/Desktop/vietlawbert/benchmark/vietlawbench_1000.jsonl"
OUTPUT_PROMPTFOO = "vietlawbench_promptfoo_tests.json"

tests = []
with open(INPUT_BENCHMARK, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        tests.append({
            "description": f"Test case {data.get('benchmark_id')}",
            "vars": {
                "query": data.get("query"),
                "ground_truth_chunk": data.get("ground_truth_chunk"),
                "ground_truth_doc_number": data.get("ground_truth_doc_number"),
                "ground_truth_article": data.get("ground_truth_article"),
            }
        })

with open(OUTPUT_PROMPTFOO, "w", encoding="utf-8") as f:
    json.dump(tests, f, ensure_ascii=False, indent=2)

print(f"✓ Đã chuyển đổi thành công {len(tests)} testcases sang {OUTPUT_PROMPTFOO}")