"""Benchmark dataset loaders for evaluation."""

import json
import os
from typing import List, Dict, Optional
from datasets import load_dataset
from dataclasses import dataclass


@dataclass
class BenchmarkSample:
    """A single sample from a benchmark dataset."""
    id: str
    question: str
    answer: str
    category: Optional[str] = None
    difficulty: Optional[int] = None
    metadata: Optional[Dict] = None


class BenchmarkLoader:
    """Base class for benchmark loaders."""
    
    def __init__(self, data_dir: str = "data/benchmarks"):
        self.data_dir = data_dir
    
    def load(self, split: str = "test") -> List[BenchmarkSample]:
        """Load benchmark samples."""
        raise NotImplementedError
    
    def save_cache(self, samples: List[BenchmarkSample], cache_path: str):
        """Save samples to cache file."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            for sample in samples:
                json.dump({
                    'id': sample.id,
                    'question': sample.question,
                    'answer': sample.answer,
                    'category': sample.category,
                    'difficulty': sample.difficulty,
                    'metadata': sample.metadata,
                }, f)
                f.write('\n')
    
    def load_cache(self, cache_path: str) -> List[BenchmarkSample]:
        """Load samples from cache file."""
        samples = []
        with open(cache_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append(BenchmarkSample(**data))
        return samples


class GSM8kLoader(BenchmarkLoader):
    """Loader for GSM8k (Grade School Math) benchmark."""
    
    def load(self, split: str = "test") -> List[BenchmarkSample]:
        """Load GSM8k dataset."""
        cache_path = os.path.join(self.data_dir, f"gsm8k/{split}.jsonl")
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading GSM8k from cache: {cache_path}")
            return self.load_cache(cache_path)
        
        # Download from HuggingFace
        print(f"Downloading GSM8k {split} split...")
        dataset = load_dataset("gsm8k", "main", split=split)
        
        samples = []
        for idx, item in enumerate(dataset):
            # Extract answer from the solution
            answer = self._extract_answer(item['answer'])
            
            samples.append(BenchmarkSample(
                id=f"gsm8k_{idx}",
                question=item['question'],
                answer=answer,
                category="math",
                difficulty=None,
                metadata={'full_solution': item['answer']}
            ))
        
        # Save to cache
        self.save_cache(samples, cache_path)
        print(f"Saved {len(samples)} samples to {cache_path}")
        
        return samples
    
    @staticmethod
    def _extract_answer(solution: str) -> str:
        """Extract the final answer from GSM8k solution."""
        # GSM8k answers are in format: "... #### 42"
        if "####" in solution:
            return solution.split("####")[-1].strip()
        return solution.strip()


class MATHLoader(BenchmarkLoader):
    """Loader for MATH (Competition Mathematics) benchmark."""
    
    def load(self, split: str = "test") -> List[BenchmarkSample]:
        """Load MATH dataset."""
        cache_path = os.path.join(self.data_dir, f"math/{split}.jsonl")
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading MATH from cache: {cache_path}")
            return self.load_cache(cache_path)
        
        # Download from HuggingFace
        print(f"Downloading MATH {split} split...")
        dataset = load_dataset("hendrycks/competition_math", split=split)
        
        samples = []
        for idx, item in enumerate(dataset):
            samples.append(BenchmarkSample(
                id=f"math_{idx}",
                question=item['problem'],
                answer=item['solution'],
                category=item['type'],  # algebra, geometry, etc.
                difficulty=int(item['level']),  # 1-5
                metadata={'type': item['type'], 'level': item['level']}
            ))
        
        # Save to cache
        self.save_cache(samples, cache_path)
        print(f"Saved {len(samples)} samples to {cache_path}")
        
        return samples


class MMLULoader(BenchmarkLoader):
    """Loader for MMLU (Massive Multitask Language Understanding) benchmark."""
    
    def load(self, split: str = "test", subjects: Optional[List[str]] = None) -> List[BenchmarkSample]:
        """Load MMLU dataset."""
        cache_path = os.path.join(self.data_dir, f"mmlu/{split}.jsonl")
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading MMLU from cache: {cache_path}")
            samples = self.load_cache(cache_path)
            if subjects:
                samples = [s for s in samples if s.category in subjects]
            return samples
        
        # Download from HuggingFace
        print(f"Downloading MMLU {split} split...")
        dataset = load_dataset("cais/mmlu", "all", split=split)
        
        samples = []
        for idx, item in enumerate(dataset):
            # Format multiple choice question
            question = self._format_mcq(
                item['question'],
                item['choices']
            )
            
            # Answer is A, B, C, or D
            answer = ['A', 'B', 'C', 'D'][item['answer']]
            
            samples.append(BenchmarkSample(
                id=f"mmlu_{idx}",
                question=question,
                answer=answer,
                category=item['subject'],
                difficulty=None,
                metadata={'subject': item['subject'], 'choices': item['choices']}
            ))
        
        # Save to cache
        self.save_cache(samples, cache_path)
        print(f"Saved {len(samples)} samples to {cache_path}")
        
        if subjects:
            samples = [s for s in samples if s.category in subjects]
        
        return samples
    
    @staticmethod
    def _format_mcq(question: str, choices: List[str]) -> str:
        """Format multiple choice question."""
        formatted = f"{question}\n\n"
        for i, choice in enumerate(choices):
            formatted += f"{chr(65+i)}. {choice}\n"
        return formatted.strip()


def load_benchmark(
    benchmark_name: str,
    split: str = "test",
    data_dir: str = "data/benchmarks",
    **kwargs
) -> List[BenchmarkSample]:
    """
    Load a benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark (gsm8k, math, mmlu)
        split: Dataset split (train, test, validation)
        data_dir: Directory to store benchmark data
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        List of benchmark samples
    """
    loaders = {
        'gsm8k': GSM8kLoader,
        'math': MATHLoader,
        'mmlu': MMLULoader,
    }
    
    if benchmark_name not in loaders:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(loaders.keys())}")
    
    loader = loaders[benchmark_name](data_dir=data_dir)
    return loader.load(split=split, **kwargs)


if __name__ == "__main__":
    # Test benchmark loaders
    print("Testing GSM8k loader...")
    gsm8k_samples = load_benchmark("gsm8k", split="test")
    print(f"Loaded {len(gsm8k_samples)} GSM8k samples")
    print(f"Example: {gsm8k_samples[0]}")
    
    print("\nTesting MATH loader...")
    math_samples = load_benchmark("math", split="test")
    print(f"Loaded {len(math_samples)} MATH samples")
    print(f"Example: {math_samples[0]}")
    
    print("\nTesting MMLU loader...")
    mmlu_samples = load_benchmark("mmlu", split="test")
    print(f"Loaded {len(mmlu_samples)} MMLU samples")
    print(f"Example: {mmlu_samples[0]}")
