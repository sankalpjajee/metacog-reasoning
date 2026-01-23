# Exact Metacognitive Prompting Steps from Wang & Zhao (NAACL 2024)

**Source**: https://github.com/EternityYW/Metacognitive-Prompting/blob/main/prompts/zero_shot_prompts.pdf

## SST-2 (Sentiment Classification) - Metacognitive Prompting

For the sentence: "[sentence]", is the sentiment in this sentence positive or negative? Exclude the neutral case. As you perform this task, follow these steps:

1. **Clarify your understanding of the sentence.**
2. **Make a preliminary identification of the sentiment of the given text.**
3. **Critically assess your preliminary analysis. If you feel unsure about your initial sentiment classification, try to reassess it.**
4. **Confirm your final answer and explain the reasoning behind your choice.**
5. **Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level.**

Provide the answer in your final response as "The sentiment is {} (positive / negative)".

---

## Key Observations

### The 5 Steps Match the Metacognitive Framework:

1. **Self-understanding** (Clarify understanding)
2. **Preliminary Judgment** (Make preliminary identification)
3. **Reflection** (Critically assess and reassess)
4. **Final Decision** (Confirm final answer with reasoning)
5. **Self-regulation** (Evaluate confidence level)

### Important Differences from My Custom Prompt:

- **Much simpler and cleaner** - no detailed sub-instructions
- **No explicit "verify calculations" or "check each step"** - just "critically assess"
- **Confidence is 0-100%** not "high/medium/low"
- **Final answer format is clearly specified** at the end
- **Steps are numbered and concise**

### For Math/Reasoning Tasks (GSM8K):

Need to adapt this to math problems. The structure should be:

1. Clarify your understanding of the problem
2. Make a preliminary solution to the problem
3. Critically assess your preliminary solution. If you feel unsure, try to reassess it
4. Confirm your final answer and explain the reasoning behind your choice
5. Evaluate your confidence (0-100%) in your analysis and provide an explanation for this confidence level

---

## Next Steps

1. Update `metacognitive_evaluator.py` with this exact structure
2. Test on 10 samples first
3. If successful, run full evaluation
