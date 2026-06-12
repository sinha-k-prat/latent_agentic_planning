# Bare Qwen vs (trained latent planner + frozen Qwen)

Model: **Qwen2.5-0.5B-Instruct**. *Base* = the frozen executor, no plan. *Planner* = a trained latent plan (hardened codebook entries) prepended as prefix `inputs_embeds` to the **same frozen** Qwen — only the learned latent prefix differs between columns.

_**Generalization test**: the planner was CE-distilled on 8 *different* 'list → select → describe' prompts and is evaluated here on these **held-out, unseen topics**. If the latent plan steers the frozen Qwen to decompose-then-execute on a topic it never trained on, the codes encode a transferable thinking strategy in latent space — not memorized text._

## 1. List the parts of the human body, then write two sentences about the largest organ.

**Base Qwen (no plan):**

> The human body consists of 206 bones, which make up the skeleton. The largest organ in the human body is the liver, which plays a crucial role in digestion and metabolism.

**Planner + frozen Qwen** (codes [24, 60, 33, 41, 30, 17, 23, 43]):

> Plan: 1) list the parts of the human body 2) write two sentences about the largest organ. Output: listing the parts of the human body would take a long time, so I will just give you the answer directly. The largest organ is the liver. It's located in the upper right part of the abdomen and plays an important role in filtering blood and producing bile.

## 2. List the oceans of the Earth, then write two sentences about the largest.

**Base Qwen (no plan):**

> The oceans of the Earth are the Pacific Ocean, Atlantic Ocean, Indian Ocean, Arctic Ocean, and Southern Ocean. The largest ocean by area is the Pacific Ocean, which covers an area of approximately 14,870,000 square kilometers (5,693,000 square miles).

**Planner + frozen Qwen** (codes [24, 60, 33, 41, 30, 17, 23, 43]):

> Plan: 1) list the oceans of the Earth 2) write two sentences about the largest.

## 3. List the basic arithmetic operations, then write two sentences about the one that splits a number into equal parts.

**Base Qwen (no plan):**

> Basic arithmetic operations include addition, subtraction, multiplication, and division. One operation that splits a number into equal parts is division. For example, if you have 12 cookies and you want to share them equally among 4 friends, you would divide the 12 by 4 to get 3 cookies per friend.

**Planner + frozen Qwen** (codes [24, 60, 33, 41, 30, 17, 23, 43]):

> Plan: 1) list the basic arithmetic operations 2) write two sentences about the one that splits a number into equal parts. Implementation: Plan: 1) list the basic arithmetic operations 2) write two sentences about the one that splits a number into equal parts. Execution: The first operation is addition, which splits a number into two equal parts by adding it to itself. The second operation is subtraction, which splits a number into three equal parts by subtracting it

## 4. List three programming languages, then write two sentences about the one best known for data science.

**Base Qwen (no plan):**

> Three programming languages are Python, R, and SQL.   Python is the most popular language for data science due to its simplicity, readability, and extensive libraries that make it easy to work with large datasets. It's also widely used in machine learning and artificial intelligence research.  R is another popular language for data science, particularly in fields such as statistics, econometrics, and finance. Its syntax is similar to SQL but offers more advanced statistical capabilities and visualization tools.  SQL (Structured Query

**Planner + frozen Qwen** (codes [24, 60, 33, 41, 30, 17, 23, 43]):

> Plan: 1) list programming languages 2) find the best known for data science 3) write two sentences about it. Solution: Programming languages: Python, R, SQL Best known language for data science: Python. Python is a high-level, interpreted, and versatile programming language that is widely used in data science due to its simplicity, readability, and extensive libraries for data analysis and visualization. It has become popular among data scientists because of its ability to handle large

---

**Decompose-then-execute structure detected:** base 0/4  ·  planner 4/4

## Honest findings (read before celebrating)

- **Structure transfers, quantitatively too:** held-out NLL on these 4 unseen topics drops
  **2.041 (base) → 1.118 (planner), Δ −0.92** — the trained latent plan steers the *frozen*
  Qwen into the plan format on topics it never trained on.
- **But the plan collapsed to a CONSTANT prefix.** The hardened codes are *identical* for all
  four prompts — `[24, 60, 33, 41, 30, 17, 23, 43]`. The "planner" learned **one
  instruction-independent 'be-structured' prefix**, not a per-instruction plan. So this shows
  *format-steering transfer*, not input-adaptive planning. (Likely causes: only 8 training
  topics sharing one structure, and the plan head produces all N slots in parallel from a
  single pooled vector with no inter-slot conditioning.)
- **Base Qwen-0.5B did not "fail".** It answers these easy prompts well; the planner adds
  *format*, not *quality*, and sometimes **degrades** it (prompt 2: emitted only the plan and
  never answered; prompt 3: garbled/incorrect). So this is **not** a quality win over base.
- **Takeaway:** the steering mechanism + structural generalization are real and measurable;
  a *quality* win and *input-adaptive* plans need (a) diverse plan structures + more data, and
  (b) an autoregressive plan head (slots conditioning on prior slots), which also enables
  dynamic plan length / halting.
