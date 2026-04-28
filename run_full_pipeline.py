from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "law_dataset" / "data"


def run_step(name: str, command: list[str]) -> None:
    print("\n" + "=" * 80)
    print(f"START STEP: {name}")
    print("COMMAND:", " ".join(command))
    print("=" * 80)

    start = time.time()
    result = subprocess.run(command, cwd=ROOT)

    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[FAILED] {name} after {elapsed:.2f}s")
        raise SystemExit(result.returncode)

    print(f"\n[SUCCESS] {name} finished in {elapsed:.2f}s")


def print_data_state() -> None:
    files_to_check = [
        DATA_DIR / "raw_links.json",
        DATA_DIR / "raw_documents.jsonl",
        DATA_DIR / "cleaned_documents.jsonl",
        DATA_DIR / "documents.jsonl",
        DATA_DIR / "chunks.jsonl",
        DATA_DIR / "domain_datasets" / "traffic_law" / "documents.jsonl",
        DATA_DIR / "domain_datasets" / "traffic_law" / "chunks.jsonl",
        DATA_DIR / "domain_datasets" / "traffic_law" / "summary.json",
    ]

    print("\nCurrent data state:")
    for path in files_to_check:
        if path.exists():
            print(f"  [EXISTS] {path} ({path.stat().st_size} bytes)")
        else:
            print(f"  [MISSING] {path}")


def main() -> None:
    python_exec = sys.executable

    reset_first = input("Reset database và dữ liệu cũ trước khi chạy? (y/n): ").strip().lower()
    if reset_first == "y":
        run_step("Reset Project State", [python_exec, "reset_project_state.py", "--yes"])

    print("Running full pipeline from scratch...")
    print_data_state()

    steps = [
        ("Phase 1 - Collect Links", [python_exec, "phase1_collect_links.py"]),
        ("Phase 2 - Parse Documents", [python_exec, "phase2_parse_documents.py"]),
        ("Phase 3 - Chunk Documents", [python_exec, "phase3_chunk_documents.py"]),
        ("Init MongoDB Indexes", [python_exec, "-m", "app.db.init_indexes"]),
        ("Phase 4 - Load to MongoDB", [python_exec, "phase4_load_to_mongodb.py"]),
        (
            "Phase 5 - Build Traffic Law Dataset",
            [
            python_exec,
            "phase5_build_domain_dataset.py",
            "--domain-config",
            "configs/domains/traffic_law.json",
            ],
        ),
    ]

    for step_name, cmd in steps:
        run_step(step_name, cmd)

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETED")
    print("=" * 80)
    print_data_state()


if __name__ == "__main__":
    main()