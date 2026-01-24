# Benchmark Overview

This document provides detailed information about all six benchmarks used in the evaluation framework.

## Benchmark Summary

| Benchmark | Type | Samples | Focus | Difficulty |
|:----------|:-----|:--------|:------|:-----------|
| **GSM8k** | Math | 1,319 | Grade school math word problems | Easy-Medium |
| **MATH** | Math | 5,000 | Competition mathematics | Hard |
| **MMLU** | Knowledge | 14,042 | Multitask language understanding | Medium |
| **HellaSwag** | Reasoning | 10,042 | Commonsense reasoning | Medium |
| **BIG-Bench** | Reasoning | Varies | Diverse reasoning tasks | Medium-Hard |
| **HumanEval** | Code | 164 | Python code generation | Medium-Hard |

## Detailed Descriptions

### 1. GSM8k (Grade School Math)

**Full Name:** Grade School Math 8K  
**Paper:** [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)  
**Dataset:** `gsm8k` on HuggingFace

**Description:**  
GSM8k contains 8,500 grade school math word problems that require multi-step reasoning. Problems involve basic arithmetic operations and require understanding of real-world contexts.

**Example:**
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold half 
as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: 72
```

**Evaluation Metric:** Exact match accuracy

**Why it matters:** Tests basic mathematical reasoning and multi-step problem solving.

---

### 2. MATH (Competition Mathematics)

**Full Name:** Mathematics Aptitude Test of Heuristics  
**Paper:** [Measuring Mathematical Problem Solving](https://arxiv.org/abs/2103.03874)  
**Dataset:** `hendrycks/competition_math` on HuggingFace

**Description:**  
MATH contains 12,500 competition-level mathematics problems from AMC 10, AMC 12, and AIME competitions. Problems span 7 categories (Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus) and 5 difficulty levels.

**Example:**
```
Question: Find the sum of all positive integers n such that, given an unlimited 
supply of stamps of denominations 5, n, and n+1 cents, 91 cents is the greatest 
postage that cannot be formed.

Answer: 18
Category: Number Theory
Level: 5
```

**Evaluation Metrics:**
- Overall accuracy
- Per-category accuracy
- Per-difficulty accuracy (Level 1-5)

**Why it matters:** Tests advanced mathematical reasoning and problem-solving skills.

---

### 3. MMLU (Massive Multitask Language Understanding)

**Full Name:** Massive Multitask Language Understanding  
**Paper:** [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)  
**Dataset:** `cais/mmlu` on HuggingFace

**Description:**  
MMLU is a benchmark covering 57 subjects across STEM, humanities, social sciences, and other areas. Each question is a multiple-choice question with 4 options.

**Example:**
```
Question: What is the embryological origin of the hyoid bone?

A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches

Answer: D
Subject: anatomy
```

**Evaluation Metrics:**
- Overall accuracy
- Per-subject accuracy
- Per-category accuracy (STEM, Humanities, Social Sciences, Other)

**Why it matters:** Tests broad knowledge and reasoning across diverse domains.

---

### 4. HellaSwag (Commonsense Reasoning)

**Full Name:** HellaSwag: Can a Machine Really Finish Your Sentence?  
**Paper:** [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)  
**Dataset:** `Rowan/hellaswag` on HuggingFace

**Description:**  
HellaSwag is a benchmark for commonsense natural language inference. Given a context, the model must select the most plausible continuation from 4 options. Problems are derived from ActivityNet and WikiHow.

**Example:**
```
Context: A woman is outside with a bucket and a dog. The dog is running around 
trying to avoid a bath. She...

A. rinses the bucket off with soap and blow dries the dog's head.
B. uses a hose to keep it from getting soapy.
C. gets the dog wet, then it runs away again.
D. gets into a bathtub with the dog.

Answer: C
```

**Evaluation Metrics:**
- Overall accuracy
- Per-activity accuracy

**Why it matters:** Tests commonsense reasoning and understanding of everyday scenarios.

---

### 5. BIG-Bench (Beyond the Imitation Game)

**Full Name:** Beyond the Imitation Game Benchmark  
**Paper:** [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615)  
**Dataset:** `bigbench` on HuggingFace

**Description:**  
BIG-Bench is a collaborative benchmark with 200+ diverse tasks designed to probe language model capabilities. Tasks cover linguistics, mathematics, common sense reasoning, and more.

**Popular Tasks:**
- **Logical Deduction:** Determine ordering from constraints
- **Causal Judgment:** Identify causal relationships
- **Formal Fallacies:** Detect logical fallacies
- **Temporal Sequences:** Reason about time
- **Tracking Shuffled Objects:** Track object positions

**Example (Logical Deduction):**
```
Question: The following paragraphs each describe a set of three objects arranged 
in a fixed order. The statements are logically consistent within each paragraph. 
On a shelf, there are three books: a white book, a green book, and an orange book. 
The green book is to the right of the white book. The orange book is the rightmost.

Options:
A. The white book is the leftmost.
B. The green book is the leftmost.
C. The orange book is the leftmost.

Answer: A
```

**Evaluation Metrics:**
- Overall accuracy
- Per-task accuracy

**Why it matters:** Tests diverse reasoning capabilities beyond standard benchmarks.

---

### 6. HumanEval (Code Generation)

**Full Name:** Evaluating Large Language Models Trained on Code  
**Paper:** [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)  
**Dataset:** `openai_humaneval` on HuggingFace

**Description:**  
HumanEval consists of 164 hand-written programming problems that test code generation and problem-solving abilities. Each problem includes a function signature, docstring, body, and unit tests.

**Example:**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
```

**Evaluation Metrics:**
- Pass@k: Percentage of problems solved with k attempts
- Functional correctness

**Why it matters:** Tests code generation, algorithmic thinking, and programming skills.

---

## Benchmark Statistics

### Sample Counts

| Benchmark | Train | Validation | Test | Total |
|:----------|:------|:-----------|:-----|:------|
| GSM8k | 7,473 | - | 1,319 | 8,792 |
| MATH | 7,500 | - | 5,000 | 12,500 |
| MMLU | 99,842 | 1,531 | 14,042 | 115,415 |
| HellaSwag | 39,905 | 10,042 | 10,003 | 59,950 |
| BIG-Bench | Varies by task | - | Varies | ~200 tasks |
| HumanEval | - | - | 164 | 164 |

### Reasoning Types Covered

| Reasoning Type | Benchmarks |
|:---------------|:-----------|
| **Mathematical** | GSM8k, MATH |
| **Logical** | BIG-Bench (logical deduction, formal fallacies) |
| **Commonsense** | HellaSwag, BIG-Bench |
| **Causal** | BIG-Bench (causal judgment) |
| **Temporal** | BIG-Bench (temporal sequences) |
| **Spatial** | BIG-Bench (tracking objects) |
| **Algorithmic** | HumanEval |
| **Knowledge** | MMLU |

## Expected Performance Ranges

Based on published results for Llama-3.1-8B and similar models:

| Benchmark | Baseline (Expected) | Strong Model | Human |
|:----------|:-------------------|:-------------|:------|
| GSM8k | 60-70% | 80-90% | 95%+ |
| MATH | 20-30% | 40-50% | 90%+ |
| MMLU | 65-75% | 80-85% | 90%+ |
| HellaSwag | 75-85% | 90-95% | 95%+ |
| BIG-Bench | 40-60% | 70-80% | 85%+ |
| HumanEval | 20-40% | 60-80% | 90%+ |

## Usage in Evaluation

### Download All Benchmarks

```bash
bash scripts/download_benchmarks.sh
```

### Evaluate on Specific Benchmarks

```bash
# Single benchmark
python scripts/evaluate_baseline.py --benchmarks gsm8k

# Multiple benchmarks
python scripts/evaluate_baseline.py --benchmarks gsm8k,math,hellaswag

# All benchmarks
python scripts/evaluate_baseline.py --benchmarks gsm8k,math,mmlu,hellaswag,bigbench,humaneval
```

### BIG-Bench Task Selection

For BIG-Bench, specify the task:

```python
from src.evaluation.benchmarks import load_benchmark

# Load specific task
samples = load_benchmark("bigbench", split="default", task="logical_deduction")
```

Popular BIG-Bench tasks for reasoning:
- `logical_deduction`
- `causal_judgment`
- `formal_fallacies`
- `temporal_sequences`
- `tracking_shuffled_objects`

## References

1. **GSM8k:** Cobbe et al. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
2. **MATH:** Hendrycks et al. (2021). [Measuring Mathematical Problem Solving](https://arxiv.org/abs/2103.03874)
3. **MMLU:** Hendrycks et al. (2020). [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
4. **HellaSwag:** Zellers et al. (2019). [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)
5. **BIG-Bench:** Srivastava et al. (2022). [Beyond the Imitation Game](https://arxiv.org/abs/2206.04615)
6. **HumanEval:** Chen et al. (2021). [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
