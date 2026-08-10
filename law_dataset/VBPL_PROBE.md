# VBPL verification probe

Run this read-only probe before changing the production crawler:

```bash
cd law_dataset
python src/crawler/probe_vbpl.py --sample-size 24
```

For a quick smoke test using only the first three advertised sitemap files:

```bash
python src/crawler/probe_vbpl.py \
  --max-sitemaps 3 \
  --sample-size 6 \
  --output-dir artifacts/vbpl_probe_smoke
```

The probe produces:

- `REPORT.md`: human-readable findings;
- `report.json`: machine-readable counts and field coverage;
- `sitemap_inventory.jsonl.gz`: all unique public document URLs and ID formats;
- `sample_pages.jsonl`: metadata and structural markers from sampled pages;
- `robots.txt`: the policy observed during that run.

## Safety and interpretation

- Discovery comes from the sitemap declared in `robots.txt`.
- The probe uses a delay and retries transient failures.
- It does not call VBPL paths disallowed by `robots.txt`.
- It does not assume that agency filters contain only current ministries.
- Document IDs remain strings because VBPL uses numeric, UUID, and legacy-prefixed IDs.
- Full text is a separate concern from public metadata. A page can expose valid
  `Legislation` JSON-LD while loading its full legal content dynamically.

Run tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Run a stratified metadata/content-readiness audit

After generating the full sitemap inventory:

```bash
python src/crawler/audit_vbpl.py \
  --inventory artifacts/vbpl_probe_full/sitemap_inventory.jsonl.gz \
  --sample-size 120 \
  --workers 2 \
  --output-dir artifacts/vbpl_content_audit_120
```

`GROUP_AND_CONTENT_CHECK_REQUIRED` means that the public metadata is suitable for
later legal-group and full-text checks. It does not mean that the record is already
safe to embed.
