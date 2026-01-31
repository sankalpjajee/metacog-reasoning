"""HotPotQA benchmark loader for multi-hop reasoning evaluation."""
import json
import os
from typing import List, Dict, Optional
from datasets import load_dataset
from dataclasses import dataclass, field

@dataclass
class HotPotQASample:
    """A single sample from HotPotQA benchmark."""
    id: str
    question: str
    answer: str
    type: str  # 'comparison' or 'bridge'
    level: str  # 'easy', 'medium', 'hard'
    supporting_facts: List[Dict[str, any]] = field(default_factory=list)
    context: List[Dict[str, any]] = field(default_factory=list)
    
    def to_benchmark_sample(self):
        """Convert to standard BenchmarkSample format."""
        from .benchmarks import BenchmarkSample
        return BenchmarkSample(
            id=self.id,
            question=self.question,
            answer=self.answer,
            category=f"hotpotqa_{self.type}",
            difficulty={"easy": 1, "medium": 2, "hard": 3}.get(self.level, 2),
            metadata={
                'type': self.type,
                'level': self.level,
                'supporting_facts': self.supporting_facts,
                'context': self.context
            }
        )


class HotPotQALoader:
    """Loader for HotPotQA multi-hop reasoning benchmark."""
    
    def __init__(self, data_dir: str = "data/benchmarks"):
        self.data_dir = data_dir
    
    def load(self, split: str = "validation", setting: str = "distractor") -> List[HotPotQASample]:
        """Load HotPotQA dataset.
        
        Args:
            split: 'train' or 'validation'
            setting: 'distractor' (10 paragraphs) or 'fullwiki' (all Wikipedia)
        
        Returns:
            List of HotPotQASample objects
        """
        cache_path = os.path.join(self.data_dir, f"hotpotqa/{split}_{setting}.jsonl")
        
        # Try to load from cache
        if os.path.exists(cache_path):
            print(f"Loading HotPotQA from cache: {cache_path}")
            return self.load_cache(cache_path)
        
        # Download from HuggingFace
        print(f"Downloading HotPotQA {split} split ({setting} setting)...")
        dataset = load_dataset("hotpotqa/hotpot_qa", setting, split=split)
        
        samples = []
        for idx, item in enumerate(dataset):
            samples.append(HotPotQASample(
                id=item['id'],
                question=item['question'],
                answer=item['answer'],
                type=item['type'],
                level=item['level'],
                supporting_facts=item['supporting_facts'],
                context=item['context']
            ))
        
        # Save to cache
        self.save_cache(samples, cache_path)
        print(f"Saved {len(samples)} samples to {cache_path}")
        
        return samples
    
    def save_cache(self, samples: List[HotPotQASample], cache_path: str):
        """Save samples to cache file."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            for sample in samples:
                json.dump({
                    'id': sample.id,
                    'question': sample.question,
                    'answer': sample.answer,
                    'type': sample.type,
                    'level': sample.level,
                    'supporting_facts': sample.supporting_facts,
                    'context': sample.context,
                }, f)
                f.write('\n')
    
    def load_cache(self, cache_path: str) -> List[HotPotQASample]:
        """Load samples from cache file."""
        samples = []
        with open(cache_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append(HotPotQASample(**data))
        return samples
    
    def format_context(self, sample: HotPotQASample, max_paragraphs: int = 10) -> str:
        """Format context paragraphs for the model.
        
        Args:
            sample: HotPotQASample
            max_paragraphs: Maximum number of paragraphs to include
        
        Returns:
            Formatted context string
        """
        context_str = ""
        for i, (title, sentences) in enumerate(sample.context[:max_paragraphs]):
            context_str += f"\n[{i+1}] {title}\n"
            for sent in sentences:
                context_str += f"  {sent}\n"
        return context_str.strip()


def load_hotpotqa(split: str = "validation", 
                  setting: str = "distractor",
                  max_samples: Optional[int] = None,
                  data_dir: str = "data/benchmarks") -> List:
    """Convenience function to load HotPotQA.
    
    Args:
        split: 'train' or 'validation'
        setting: 'distractor' or 'fullwiki'
        max_samples: Maximum number of samples to load (None for all)
        data_dir: Directory for cached data
    
    Returns:
        List of BenchmarkSample objects
    """
    loader = HotPotQALoader(data_dir=data_dir)
    samples = loader.load(split=split, setting=setting)
    
    if max_samples is not None:
        samples = samples[:max_samples]
    
    # Convert to standard BenchmarkSample format
    return [s.to_benchmark_sample() for s in samples]


if __name__ == "__main__":
    # Test the loader
    print("Testing HotPotQA loader...")
    samples = load_hotpotqa(split="validation", setting="distractor", max_samples=5)
    
    print(f"\nLoaded {len(samples)} samples")
    print("\nExample sample:")
    sample = samples[0]
    print(f"ID: {sample.id}")
    print(f"Question: {sample.question}")
    print(f"Answer: {sample.answer}")
    print(f"Type: {sample.metadata['type']}")
    print(f"Level: {sample.metadata['level']}")
    print(f"Supporting facts: {len(sample.metadata['supporting_facts'])} facts")
    print(f"Context: {len(sample.metadata['context'])} paragraphs")
