# Metacognitive Prompting Paper Notes

**Paper**: Metacognitive Prompting Improves Understanding in Large Language Models  
**Authors**: Yuqing Wang, Yun Zhao  
**Venue**: NAACL 2024  
**ArXiv**: https://arxiv.org/abs/2308.05342

## Key Findings

- Metacognitive Prompting (MP) is inspired by human introspective reasoning processes
- LLMs undergo systematic series of structured, self-aware evaluations
- Tested on 4 LLMs: Llama2, PaLM2, GPT-3.5, GPT-4
- Tested on 10 NLU datasets from GLUE, SuperGLUE, BLUE, LexGLUE
- MP consistently outperforms chain-of-thought and its variants

## Metacognitive Process Stages (from Figure 1)

The paper shows alignment between human metacognition and MP stages:

1. **Self-understanding** (Recognition of knowledge)
2. **Preliminary Judgment** (Initial evaluation)
3. **Reflection** (Examination of initial judgment)
4. **Final Decision** (Consolidation of reasoning)
5. **Self-regulation** (Evaluation assessment of the entire process)
6. **Confidence Assessment** (How confident and reliable is the reasoning)

## Need to Find

- Exact prompt templates used for each stage
- How they apply this to reasoning/math tasks (GSM8K-like)
- Specific wording for each metacognitive stage

## Next Steps

1. Check GitHub repo for exact prompts: https://github.com/EternityYW/Metacognitive-Prompting
2. Read the methodology section for prompt details
3. Implement the exact prompts from the paper
