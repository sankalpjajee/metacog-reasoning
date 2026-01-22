# Baseline Evaluation Results

## Model
- **Model:** meta-llama/Llama-3.1-8B-Instruct
- **Date:** January 22, 2026
- **Samples:** 1000 per benchmark (except HumanEval: 164)

## Results

| Benchmark | Accuracy | Samples | Notes |
|-----------|----------|---------|-------|
| GSM8K | 81.1% | 1000 | Math reasoning |
| MMLU | 69.5% | 1000 | General knowledge |
| HellaSwag | 64.1% | 1000 | Commonsense reasoning |
| MR-Ben | 28.8% | 1000 | Multi-step reasoning |

## Comparison to Expected

All results are at or above expected performance for Llama-3.1-8B-Instruct,
confirming the evaluation pipeline is working correctly.
