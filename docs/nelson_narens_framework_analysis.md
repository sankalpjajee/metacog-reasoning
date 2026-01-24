# Nelson & Narens (1990) Framework Analysis

## The Core Framework

From the original Nelson & Narens (1990) paper "Metamemory: A Theoretical Framework and New Findings":

```
┌─────────────────────────────────────┐
│           META-LEVEL                │
│                                     │
└─────────────────────────────────────┘
         ↑                    ↓
    Monitoring           Control
  (Flow of Info)     (Flow of Info)
         ↑                    ↓
┌─────────────────────────────────────┐
│          OBJECT-LEVEL               │
│                                     │
└─────────────────────────────────────┘
```

### Key Definitions:

**Monitoring**: Information flows FROM object-level TO meta-level
- The meta-level receives information about the state of the object-level
- Examples: "How confident am I?", "Did I understand this?", "Is this answer correct?"

**Control**: Information flows FROM meta-level TO object-level
- The meta-level modifies the object-level based on monitoring
- Examples: "Try a different strategy", "Check your work", "Allocate more time to this"

---

## Comparing Wang & Zhao (2024) to Nelson & Narens (1990)

### Wang & Zhao's 5 Steps:

1. **Clarify your understanding of the sentence** 
   → This is MONITORING (checking object-level understanding)

2. **Make a preliminary identification**
   → This is OBJECT-LEVEL (actual cognitive processing)

3. **Critically assess your preliminary analysis. If unsure, try to reassess it**
   → This is MONITORING + CONTROL (monitor confidence, then control by reassessing)

4. **Confirm your final answer and explain reasoning**
   → This is OBJECT-LEVEL + MONITORING (final decision + explanation)

5. **Evaluate your confidence (0-100%) and provide explanation**
   → This is MONITORING (assessing certainty about the process)

---

## CRITICAL ANALYSIS: Does Wang & Zhao Match Nelson & Narens?

### ✅ What Matches:

1. **Monitoring is present**: Steps 1, 3, and 5 involve monitoring
2. **Control is present**: Step 3 includes "if unsure, try to reassess" (control action)
3. **Two-level structure**: Distinguishes between doing the task vs. thinking about doing the task

### ⚠️ What's Missing or Weak:

1. **Control is MINIMAL**: Only one control action ("reassess if unsure")
   - Nelson & Narens emphasize control as a major component
   - Control should include: strategy selection, resource allocation, termination decisions
   - Wang & Zhao's prompt doesn't explicitly train these control mechanisms

2. **No explicit strategy adjustment**: 
   - Nelson & Narens: "If this approach isn't working, try another"
   - Wang & Zhao: Just "reassess" but doesn't say HOW or with WHAT different approach

3. **Monitoring is mostly POST-HOC**:
   - Step 5 evaluates confidence AFTER the answer
   - True metacognition includes DURING-TASK monitoring that triggers control
   - Example: "Wait, this calculation seems wrong [MONITORING] → Let me verify [CONTROL]"

4. **No termination control**:
   - Nelson & Narens framework includes "when to stop searching memory"
   - Wang & Zhao doesn't address "when have I thought enough?" or "when should I give up this approach?"

---

## What TRUE Nelson & Narens Metacognition Would Look Like

For a math problem like GSM8K:

```
Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast 
and bakes muffins with 4. She sells the remainder at $2 per egg. 
How much does she make?

OBJECT-LEVEL: Initial attempt
"She uses 3 + 4 = 7 eggs"

MONITORING: Check understanding
"Wait, am I clear on what the question asks? Yes, it's asking for money made."

OBJECT-LEVEL: Continue
"16 - 7 = 9 eggs to sell"

MONITORING: Confidence check
"I'm confident about this subtraction"

OBJECT-LEVEL: Calculate revenue
"9 × 2 = 18 dollars"

MONITORING: Verify calculation
"Let me double-check: 9 × 2 = 18. Yes, that's correct."

MONITORING: Overall confidence
"I'm 95% confident in this answer because the steps are straightforward"

CONTROL: Decide if more checking needed
"Given my high confidence and straightforward problem, no need for additional verification"

Final Answer: $18
```

**Key difference**: Monitoring and control happen THROUGHOUT the process, not just at the end.

---

## For Our Research: What Should We Do?

### Option 1: Use Wang & Zhao's Prompt As-Is
**Pros**: 
- Published in NAACL 2024 (peer-reviewed)
- Shows improvement on NLU tasks
- Simple and clean

**Cons**:
- Doesn't fully match Nelson & Narens framework
- Minimal control component
- May not be sufficient for our "explicit metacognition" claim

### Option 2: Enhance Wang & Zhao's Prompt
**Modify to include stronger control**:

1. Clarify your understanding of the problem
2. Make a preliminary solution
3. **[MONITORING]** Assess your confidence in each step
4. **[CONTROL]** If confidence is low, try a different approach or verify calculations
5. **[CONTROL]** Decide if you need to continue checking or if you're ready to finalize
6. Confirm your final answer with reasoning
7. **[MONITORING]** Evaluate overall confidence (0-100%)

**Pros**:
- Better alignment with Nelson & Narens
- Stronger control component
- Still grounded in published work (Wang & Zhao as foundation)

**Cons**:
- Not exactly from a paper (we're modifying it)
- Need to test if it works

### Option 3: Search for Other Metacognitive Prompting Papers
**Look for papers that explicitly implement control mechanisms**

**Pros**:
- Might find better alignment with Nelson & Narens
- Could discover more recent/relevant work

**Cons**:
- Time consuming
- May not exist (Wang & Zhao might be the best we have)

---

## My Recommendation

**Use Option 2: Enhanced Wang & Zhao Prompt**

**Justification**:
1. We can cite Wang & Zhao (2024) as the foundation
2. We explicitly state we're enhancing it to better align with Nelson & Narens (1990)
3. This becomes part of our contribution: "We adapt metacognitive prompting to include stronger control mechanisms"
4. For a NeurIPS paper, showing we understand the theoretical framework (Nelson & Narens) and adapting existing methods to better match it is a STRENGTH, not a weakness

**The enhanced prompt would be**:

```
For the problem: "[problem]", solve it step by step. As you work through this:

1. First, clarify your understanding of what the problem is asking.

2. Make a preliminary solution to the problem, showing your work.

3. As you solve, monitor your confidence in each step. If you notice uncertainty 
   or potential errors, pause and verify your work.

4. If your verification reveals issues, adjust your approach and try again.

5. Once you have a solution, decide whether you need additional verification 
   or if you're confident enough to finalize.

6. Provide your final answer with a clear explanation of your reasoning.

7. Rate your overall confidence (0-100%) in this answer and explain why you 
   have this confidence level.

Provide the answer in your final response as "The answer is [your answer]".
```

This maintains the spirit of Wang & Zhao but adds explicit control mechanisms that align with Nelson & Narens.
