# Technical Specifications and Implementation Details

## 1. Project Structure

```
metacog-reasoning/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── teacher_config.yaml
│   ├── student_config.yaml
│   └── eval_config.yaml
├── data/
│   ├── raw/                    # Original benchmarks
│   ├── processed/              # Pre-processed data
│   └── generated/              # Generated datasets
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py         # Data loading utilities
│   │   ├── translation.py     # Translation pipeline
│   │   └── preprocessing.py   # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── teacher.py         # Teacher model wrapper
│   │   ├── student.py         # Student model wrapper
│   │   └── reward_model.py    # Reward computation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── teacher_trainer.py # Phase 1 training
│   │   ├── student_trainer.py # Phase 2 training
│   │   └── grpo.py            # GRPO implementation
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── teacher_prompts.py # Teacher prompts
│   │   └── student_prompts.py # Student prompts
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── answer_reward.py   # R_answer
│   │   ├── strategy_reward.py # R_strategy
│   │   ├── process_reward.py  # R_process
│   │   └── plan_reward.py     # R_plan
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmarks.py      # Benchmark evaluation
│   │   └── metrics.py         # Metric computation
│   └── analysis/
│       ├── __init__.py
│       ├── strategy_analysis.py
│       ├── error_taxonomy.py
│       └── visualizations.py
├── scripts/
│   ├── setup_environment.sh
│   ├── download_data.sh
│   ├── train_teacher.py
│   ├── train_student.py
│   ├── evaluate.py
│   └── generate_dataset.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_strategy_analysis.ipynb
│   └── 03_results_visualization.ipynb
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_rewards.py
```

## 2. Core Components Specification

### 2.1. Reasoning Strategy Taxonomy

```python
from enum import Enum

class ReasoningStrategy(Enum):
    DECOMPOSITION = "decomposition"
    DEDUCTIVE = "deductive_reasoning"
    INDUCTIVE = "inductive_reasoning"
    CAUSAL = "causal_inference"
    ANALOGICAL = "analogical_reasoning"
    BACKWARD_CHAINING = "backward_chaining"
    PROOF_BY_CONTRADICTION = "proof_by_contradiction"
    HYPOTHESIS_TESTING = "hypothesis_testing"
```

### 2.2. Data Structures

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ReasoningProblem:
    id: str
    text: str
    language: str
    category: str  # math, logic, commonsense
    difficulty: int  # 1-5
    target_answer: Optional[str] = None

@dataclass
class ReasoningStep:
    step_number: int
    strategy: ReasoningStrategy
    text: str

@dataclass
class ReasoningSolution:
    problem_id: str
    selected_strategy: ReasoningStrategy
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float

@dataclass
class MetaCognitiveTrace:
    problem: ReasoningProblem
    solution: ReasoningSolution
    is_correct: bool
    rewards: dict  # {answer, strategy, process, plan}
```

### 2.3. Model Configuration

**Teacher Model (Llama-3.1-8B):**
```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  device_map: "auto"
  torch_dtype: "bfloat16"
  
training:
  learning_rate: 1e-5
  batch_size: 4
  gradient_accumulation_steps: 8
  num_epochs: 3
  warmup_steps: 100
  max_grad_norm: 1.0
  
generation:
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  do_sample: true
```

**Student Model (Qwen2.5-7B):**
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  device_map: "auto"
  torch_dtype: "bfloat16"
  
training:
  learning_rate: 5e-6
  batch_size: 4
  gradient_accumulation_steps: 8
  num_epochs: 2
  warmup_steps: 50
  max_grad_norm: 1.0
  distillation_ratio: 0.7  # 70% distillation, 30% self-play
  
generation:
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.9
  do_sample: true
```

## 3. Key Implementation Details

### 3.1. Phase 1: Teacher Training Loop

```python
def teacher_training_loop(model, config):
    """
    Meta-cognitive self-play training loop for teacher model.
    """
    for iteration in range(config.num_iterations):
        # 1. Generate problem
        problem = model.generate_problem(
            prompt=PROBLEM_GENERATION_PROMPT,
            category=sample_category()
        )
        
        # 2. Select strategy
        strategy = model.select_strategy(
            problem=problem,
            prompt=STRATEGY_SELECTION_PROMPT
        )
        
        # 3. Generate annotated solution
        solution = model.generate_solution(
            problem=problem,
            strategy=strategy,
            prompt=ANNOTATED_SOLUTION_PROMPT
        )
        
        # 4. Self-evaluate
        is_correct, evaluation = model.self_evaluate(
            problem=problem,
            solution=solution
        )
        
        # 5. Compute rewards
        rewards = {
            'answer': 1.0 if is_correct else 0.0,
            'strategy': evaluate_strategy_appropriateness(problem, strategy),
            'plan': evaluate_plan_consistency(solution),
        }
        
        # 6. Update model via GRPO
        loss = grpo_update(model, solution, rewards)
        
        # 7. Log progress
        log_iteration(iteration, problem, solution, rewards, loss)
```

### 3.2. Phase 2: Student Training Loop

```python
def student_training_loop(teacher_model, student_model, config):
    """
    Hybrid distillation + self-play training loop for student model.
    """
    for iteration in range(config.num_iterations):
        # Decide: distillation or self-play?
        if random.random() < config.distillation_ratio:
            # DISTILLATION MODE
            
            # 1. Get English problem
            english_problem = sample_problem(language='en')
            
            # 2. Teacher generates reference trace
            teacher_trace = teacher_model.generate_full_trace(english_problem)
            
            # 3. Translate problem to Indic language
            indic_language = sample_indic_language()
            indic_problem = translate(english_problem, target_lang=indic_language)
            
            # 4. Student attempts problem
            student_trace = student_model.generate_full_trace(indic_problem)
            
            # 5. Compute multi-component rewards
            rewards = {
                'answer': compute_answer_reward(student_trace, teacher_trace),
                'strategy': compute_strategy_reward(student_trace, teacher_trace),
                'process': compute_process_reward(student_trace, teacher_trace),
                'plan': compute_plan_reward(student_trace),
            }
            
        else:
            # SELF-PLAY MODE
            
            # 1. Student generates Indic problem
            indic_language = sample_indic_language()
            indic_problem = student_model.generate_problem(language=indic_language)
            
            # 2. Student solves problem
            student_trace = student_model.generate_full_trace(indic_problem)
            
            # 3. Verify with teacher (translate to English)
            english_problem = translate(indic_problem, target_lang='en')
            teacher_answer = teacher_model.solve(english_problem)
            
            # 4. Compute rewards
            rewards = {
                'answer': compute_answer_reward(student_trace, teacher_answer),
                'strategy': evaluate_strategy_appropriateness(indic_problem, student_trace.strategy),
                'plan': compute_plan_reward(student_trace),
                'process': 0.5,  # Neutral for self-play
            }
        
        # 6. Update student model via GRPO
        loss = grpo_update(student_model, student_trace, rewards)
        
        # 7. Log progress
        log_iteration(iteration, indic_problem, student_trace, rewards, loss)
```

### 3.3. Reward Functions

```python
def compute_answer_reward(student_trace, teacher_trace):
    """R_answer: Exact match or numerical equivalence."""
    student_answer = student_trace.final_answer
    teacher_answer = teacher_trace.final_answer
    
    # Try exact match
    if student_answer.strip() == teacher_answer.strip():
        return 1.0
    
    # Try numerical equivalence
    try:
        if abs(float(student_answer) - float(teacher_answer)) < 1e-6:
            return 1.0
    except:
        pass
    
    return 0.0

def compute_strategy_reward(student_trace, teacher_trace):
    """R_strategy: Strategy alignment."""
    return 1.0 if student_trace.selected_strategy == teacher_trace.selected_strategy else 0.0

def compute_process_reward(student_trace, teacher_trace):
    """R_process: BERTScore similarity of reasoning chains."""
    from bert_score import score
    
    # Translate student's reasoning to English
    student_reasoning = translate_to_english(student_trace.get_reasoning_text())
    teacher_reasoning = teacher_trace.get_reasoning_text()
    
    # Compute BERTScore
    P, R, F1 = score([student_reasoning], [teacher_reasoning], lang='en')
    
    return F1.item()

def compute_plan_reward(student_trace):
    """R_plan: LLM-as-a-judge for plan consistency."""
    prompt = f"""
    Evaluate whether the following solution follows its stated plan.
    
    Selected Strategy: {student_trace.selected_strategy}
    Solution Steps: {student_trace.get_reasoning_text()}
    
    Rate the consistency from 0.0 to 1.0.
    """
    
    response = reward_model.generate(prompt)
    score = extract_score(response)  # Parse the score from response
    
    return score
```

## 4. Computational Requirements

### 4.1. Hardware Specifications

**Minimum:**
- 4 × NVIDIA A100 (40GB) GPUs
- 256GB RAM
- 2TB SSD storage

**Recommended:**
- 8 × NVIDIA A100 (80GB) GPUs
- 512GB RAM
- 4TB NVMe SSD storage

### 4.2. Estimated Training Time

| Phase | Model | GPUs | Batch Size | Iterations | Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Phase 1 | Llama-3.1-8B | 8 × A100 | 32 (4×8) | 50,000 | 2-3 weeks |
| Phase 2 | Qwen2.5-7B | 8 × A100 | 32 (4×8) | 100,000 | 3-4 weeks |

### 4.3. Storage Requirements

- **Raw benchmarks:** ~5GB
- **Translated datasets:** ~10GB
- **Generated traces (Phase 1):** ~50GB
- **Generated traces (Phase 2):** ~100GB
- **Model checkpoints:** ~100GB
- **Total:** ~265GB

## 5. Dependencies

```txt
# Core ML libraries
torch>=2.1.0
transformers>=4.36.0
accelerate>=0.25.0
trl>=0.7.0
peft>=0.7.0

# Data processing
datasets>=2.16.0
pandas>=2.0.0
numpy>=1.24.0

# Translation
googletrans>=4.0.0
indicnlp>=0.1.0

# Evaluation metrics
bert-score>=0.3.13
sentence-transformers>=2.2.0
sacrebleu>=2.3.0

# Utilities
wandb>=0.16.0
pyyaml>=6.0
tqdm>=4.66.0
jupyter>=1.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
```

## 6. Next Steps

With this technical specification, you can now:

1. **Set up the project structure** following the directory layout
2. **Implement the core components** starting with data loaders and model wrappers
3. **Test individual components** before integrating into the full training loop
4. **Launch Phase 1 training** once the teacher training loop is validated
5. **Monitor and iterate** based on initial results

This specification provides a complete blueprint for implementation. Each component is modular and can be developed and tested independently before integration.
