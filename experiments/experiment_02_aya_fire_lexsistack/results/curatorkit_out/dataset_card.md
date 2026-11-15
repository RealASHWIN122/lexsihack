# Dataset Card: curatorkit_run

## Summary

- **Run timestamp**: 2026-11-15T13:04:31.929740Z
- **Wall clock time**: 0.0s
- **Pipeline config hash**: `5249142583518f21`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| CSVReader | 8 | 8 | — | 0 | 8 |
| SchemaGate | 8 | 8 | 0 | 0 | 8 |
| ExactDeduplicator | 8 | 8 | — | — | 8 |
| TextCleaner | 8 | 8 | — | — | 8 |

## Gate Pass Rates and Rejection Breakdown

- **CSVReader**: 8/0 passed (N/A), 0 rejected
- **SchemaGate**: 8/8 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.28
- **python**: 3.12.16
