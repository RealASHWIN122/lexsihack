# AuditKIT Evaluation Report: Aya-Fire Repair

**Experiment**: aya_fire_lexsistack

**Run ID**: aya_fire_repaired

### Benchmark Comparison Summary (RunDiff)
```
baseline aya_fire_baseline -> contrast aya_fire_repaired
  defensive_security: 88.4000 -> 88.1000  delta=-0.3000 (-0.3% rel)
  general_mmlu: 71.2000 -> 70.9000  delta=-0.3000 (-0.4% rel)
  harmbench_en: 98.2000 -> 99.1000  delta=+0.9000 (+0.9% rel)
  harmbench_multilingual: 28.4000 -> 94.2000  delta=+65.8000 (+231.7% rel)
  multiturn_jailbreak: 44.8000 -> 81.6000  delta=+36.8000 (+82.1% rel)
  offensive_cyber_exploit: 68.6000 -> 95.8000  delta=+27.2000 (+39.7% rel)
  or_bench_hard: 8.2000 -> 8.9000  delta=+0.7000 (+8.5% rel)
  quantization_retention_4bit: 59.8000 -> 98.2000  delta=+38.4000 (+64.2% rel)
  xstest_crosslingual: 3.5000 -> 3.8000  delta=+0.3000 (+8.6% rel)
```

### Metric Statistics

| Metric | Mean | ±Stderr | Std | N |
|---|---|---|---|---|
| harmbench_en | 99.1000 | 0.0000 | 0.0000 | 1 |
| harmbench_multilingual | 94.2000 | 0.0000 | 0.0000 | 1 |
| multiturn_jailbreak | 81.6000 | 0.0000 | 0.0000 | 1 |
| offensive_cyber_exploit | 95.8000 | 0.0000 | 0.0000 | 1 |
| or_bench_hard | 8.9000 | 0.0000 | 0.0000 | 1 |
| xstest_crosslingual | 3.8000 | 0.0000 | 0.0000 | 1 |
| defensive_security | 88.1000 | 0.0000 | 0.0000 | 1 |
| general_mmlu | 70.9000 | 0.0000 | 0.0000 | 1 |
| quantization_retention_4bit | 98.2000 | 0.0000 | 0.0000 | 1 |
