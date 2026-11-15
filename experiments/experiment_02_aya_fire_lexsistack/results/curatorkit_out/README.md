---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 2714f4377dc9283e7f0a80cfc2f445cb45fb32ea051d4b549e8821b035fef8a5
  data_files:
  - split: train
    path: sft_alpaca.jsonl
- config_name: sft_sharegpt
  description: curatorkit sft_sharegpt export, sha256 b76c74a002b0c8471cdf7ec384031d5772b810071a89e0987c2b1a825c4292b9
  data_files:
  - split: train
    path: sft_sharegpt.jsonl
- config_name: sft_messages
  description: curatorkit sft_messages export, sha256 61bf68bad0784a873ede3ca9a6429dd60b4d108c9d4316a72ae35433f387a59c
  data_files:
  - split: train
    path: sft_messages.jsonl
---

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

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (8) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
| `sft_sharegpt` | train (8) | `conversations`: list of {from, value}; from is system, human or gpt |
| `sft_messages` | train (8) | `messages`: list of {role, content}; role is system, user or assistant |
