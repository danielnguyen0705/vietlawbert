# RUN_INFO

- timestamp: 2026-08-12T16:35:45.436155+00:00
- git branch: cuong-ubutu
- git commit SHA at run start/reference: e8992010bf189f63a2e225075b8765ec814ef9ec
- python version: 3.10.12
- crawler command used: python law_dataset/src/crawler/pilot_crawl_vbpl.py --manifest law_dataset/artifacts/vbpl_pilot_120_rerun_20260812_232140/sample_manifest.jsonl --output law_dataset/artifacts/vbpl_pilot_120_rerun_20260812_232140 --delay 0.75 --timeout 90 --retries 2
- harness command used: python -m harness.checks.validate_pipeline --input law_dataset/artifacts/vbpl_pilot_120_rerun_20260812_232140/documents.jsonl.gz --report law_dataset/artifacts/vbpl_pilot_120_rerun_20260812_232140/harness_validation_report.json
- manifest SHA-256: 5d71ae2689b965f29b9a5d3ad5c321e348e26e9f0e77f9e61119a8de95c310ce
- manifest records: 120
- checkpoint records: 120
- final record count: 120
- network used: yes
- resumed run: yes
- known limitations: harness currently rejects PDF_ONLY and CONTENT_TOO_SHORT by policy; language heuristic still flags English translation records in official BD group.
