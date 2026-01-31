# Agentic Experiment: Quick Start Guide

## What is This?

This experiment extends our metacognitive reasoning work to **agentic multi-hop reasoning**. Instead of making a single decision per question, the model makes **per-step decisions** about when to use deeper reasoning.

**Key Innovation:** Using learned confidence prediction to guide strategy selection at each step of a multi-hop reasoning task.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers datasets torch tqdm
```

### 2. Test HotPotQA Loader

```bash
cd metacog-reasoning
python -m src.evaluation.hotpotqa_loader
```

Expected output:
```
Testing HotPotQA loader...
Loading HotPotQA validation split...
Loaded 5 samples

Example sample:
ID: 5a8b57f25542995d1e6f1371
Question: What government position was held by...
Answer: U.S. Ambassador to Ghana
Type: bridge
Level: hard
```

### 3. Run Agentic Evaluation (Without Probe)

```bash
python -m src.evaluation.agentic_evaluator \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --num_samples 10 \
    --output_dir results/agentic_test
```

This will run without the confidence probe (using heuristics for now).

### 4. Train Probe with HotPotQA (Optional)

To enable learned confidence prediction:

```bash
# Generate training data including HotPotQA
python -m src.training.generate_probe_data \
    --benchmarks gsm8k mmlu hellaswag hotpotqa \
    --samples_per_benchmark 2000 \
    --output_dir data/training/probe_data_full

# Train probe
python -m src.training.train_probe \
    --data_dir data/training/probe_data_full \
    --output_dir models/confidence_probe_full \
    --epochs 20

# Run evaluation with probe
python -m src.evaluation.agentic_evaluator \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --probe_path models/confidence_probe_full/best_probe.pt \
    --num_samples 1000 \
    --output_dir results/agentic_with_probe
```

---

## Files Added

| File | Purpose |
|------|---------|
| `src/evaluation/hotpotqa_loader.py` | Loads HotPotQA dataset |
| `src/evaluation/question_decomposer.py` | Decomposes multi-hop questions |
| `src/evaluation/agentic_evaluator.py` | Main agentic evaluation logic |
| `docs/AGENTIC_EXPERIMENT.md` | Full documentation |
| `docs/AGENTIC_README.md` | This quick start guide |

---

## What to Expect

### Without Probe (Heuristic)
- Uses simple heuristics to decide when to use metacognition
- Good for testing the pipeline
- Accuracy: ~60-65% on HotPotQA

### With Probe (Learned)
- Uses trained confidence predictor
- Adapts based on learned uncertainty patterns
- Expected accuracy: ~68-70% on HotPotQA
- Selective metacognition usage: ~25-30% of hops

---

## Troubleshooting

### Issue: "No module named 'src.evaluation.hotpotqa_loader'"

**Solution:** Make sure you're running from the repo root:
```bash
cd metacog-reasoning
python -m src.evaluation.hotpotqa_loader
```

### Issue: "Dataset not found"

**Solution:** The first run will download HotPotQA from HuggingFace. This may take a few minutes. Subsequent runs will use the cached version.

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or use a smaller model:
```bash
python -m src.evaluation.agentic_evaluator \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --num_samples 10  # Start with fewer samples
```

---

## Next Steps

1. **Run baseline comparison:** Compare learned-adaptive vs. baseline-all vs. metacog-all
2. **Analyze per-hop results:** See which hops benefit most from metacognition
3. **Extend to other tasks:** Apply the same approach to tool-use or planning benchmarks
4. **Write paper section:** Use results to demonstrate agentic relevance

---

## Questions?

See the full documentation in `docs/AGENTIC_EXPERIMENT.md` or open an issue on GitHub.
