#!/bin/bash

# Quick test script to validate probe accuracy on small dataset
# Run this before full training to ensure everything works

echo "=========================================="
echo "Testing Error Prediction Probe Pipeline"
echo "=========================================="

# Step 1: Generate small training dataset (100 samples per benchmark)
echo ""
echo "Step 1: Generating training data (100 samples per benchmark)..."
python -m src.training.generate_error_prediction_data_v2 \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --benchmarks gsm8k mmlu hellaswag \
    --samples_per_benchmark 100 \
    --output_dir data/training/error_prediction_test

# Step 2: Train probe (10 epochs)
echo ""
echo "Step 2: Training probe (10 epochs)..."
python -m src.training.train_error_predictor_v2 \
    --data_dir data/training/error_prediction_test \
    --output_dir models/error_predictor_test \
    --benchmarks gsm8k mmlu hellaswag \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-3

# Step 3: Evaluate on small test set (50 samples)
echo ""
echo "Step 3: Evaluating learned adaptive approach (50 samples)..."
python -m src.evaluation.learned_adaptive_evaluator_v2 \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --probe_path models/error_predictor_test/best_probe.pt \
    --probe_config models/error_predictor_test/config.json \
    --benchmarks gsm8k \
    --num_samples 50 \
    --error_threshold 0.5 \
    --output_dir results/learned_adaptive_test

echo ""
echo "=========================================="
echo "Test complete! Check results in:"
echo "  - data/training/error_prediction_test/"
echo "  - models/error_predictor_test/"
echo "  - results/learned_adaptive_test/"
echo "=========================================="
