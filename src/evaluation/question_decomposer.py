"""Question decomposer for multi-hop reasoning tasks."""
import re
from typing import List, Dict, Optional
import torch


class QuestionDecomposer:
    """Decomposes multi-hop questions into sequential sub-questions."""
    
    def __init__(self, model, tokenizer):
        """Initialize decomposer with model and tokenizer.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def decompose(self, question: str, max_hops: int = 3) -> List[str]:
        """Decompose a multi-hop question into sequential sub-questions.
        
        Args:
            question: The original multi-hop question
            max_hops: Maximum number of hops to extract
        
        Returns:
            List of sub-questions (hops)
        """
        prompt = self._get_decomposition_prompt(question)
        
        # Generate decomposition
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for consistent decomposition
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the decomposition from the response
        hops = self._extract_hops(response, max_hops)
        
        return hops
    
    def _get_decomposition_prompt(self, question: str) -> str:
        """Get the prompt for question decomposition."""
        prompt = f"""Break down this multi-hop question into sequential sub-questions. Each sub-question should build on the previous answer.

Question: {question}

Provide your decomposition in this exact format:
Hop 1: [first sub-question]
Hop 2: [second sub-question that uses the answer from Hop 1]

Important:
- Use exactly "Hop 1:", "Hop 2:", etc. as labels
- Each hop should be a complete question
- Later hops can reference earlier answers with [answer from Hop X]

Decomposition:
"""
        return prompt
    
    def _extract_hops(self, response: str, max_hops: int) -> List[str]:
        """Extract hop questions from the model's response.
        
        Args:
            response: The model's decomposition response
            max_hops: Maximum number of hops to extract
        
        Returns:
            List of hop questions
        """
        hops = []
        
        # Look for patterns like "Hop 1:", "Hop 2:", etc.
        pattern = r'Hop\s+(\d+):\s*(.+?)(?=Hop\s+\d+:|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        for hop_num, hop_text in matches[:max_hops]:
            # Clean up the hop text
            hop_text = hop_text.strip()
            # Remove any trailing punctuation or newlines
            hop_text = hop_text.split('\n')[0].strip()
            if hop_text:
                hops.append(hop_text)
        
        # If no hops found with pattern, try to extract manually
        if not hops:
            # Fallback: treat the entire question as a single hop
            hops = [response.strip()]
        
        return hops
    
    def format_hop_with_context(self, 
                                hop_question: str, 
                                previous_answers: List[Dict[str, str]]) -> str:
        """Format a hop question with context from previous hops.
        
        Args:
            hop_question: The current hop question
            previous_answers: List of dicts with 'hop' and 'answer' keys
        
        Returns:
            Formatted question with context
        """
        if not previous_answers:
            return hop_question
        
        context = "Previous information:\n"
        for prev in previous_answers:
            context += f"- {prev['hop']}: {prev['answer']}\n"
        
        # Replace placeholders in the hop question
        formatted_question = hop_question
        for i, prev in enumerate(previous_answers, 1):
            # Replace patterns like "[answer from Hop 1]" with actual answer
            patterns = [
                f"[answer from Hop {i}]",
                f"[Hop {i}]",
                f"{{answer from Hop {i}}}",
                f"{{Hop {i}}}"
            ]
            for pattern in patterns:
                formatted_question = formatted_question.replace(pattern, prev['answer'])
        
        return f"{context}\nQuestion: {formatted_question}"


class SimpleDecomposer:
    """A simple rule-based decomposer for testing without LLM calls."""
    
    def decompose(self, question: str, max_hops: int = 2) -> List[str]:
        """Simple decomposition based on question structure.
        
        For HotPotQA, most questions follow patterns like:
        - "What X did Y who Z?"  → Hop 1: "Who Z?", Hop 2: "What X did [answer]?"
        - "Where was X born who Y?" → Hop 1: "Who Y?", Hop 2: "Where was [answer] born?"
        
        Args:
            question: The original question
            max_hops: Maximum number of hops (default 2 for most HotPotQA)
        
        Returns:
            List of hop questions
        """
        # This is a simplified version - in practice, use the LLM-based decomposer
        # For now, just return the original question as a single hop
        return [question]
    
    def format_hop_with_context(self, hop_question: str, previous_answers: List[Dict[str, str]]) -> str:
        """Format hop with context (same as LLM version)."""
        if not previous_answers:
            return hop_question
        
        context = "Previous information:\n"
        for prev in previous_answers:
            context += f"- {prev['hop']}: {prev['answer']}\n"
        
        return f"{context}\nQuestion: {hop_question}"


if __name__ == "__main__":
    # Test the simple decomposer
    decomposer = SimpleDecomposer()
    
    test_questions = [
        "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "Where was the director of the film 'The Imitation Game' born?",
        "What is the birth year of the actor who played the lead role in 'The Shawshank Redemption'?"
    ]
    
    print("Testing Simple Decomposer:")
    for q in test_questions:
        print(f"\nQuestion: {q}")
        hops = decomposer.decompose(q)
        for i, hop in enumerate(hops, 1):
            print(f"  Hop {i}: {hop}")
