# NLP Question Answering — Learning Roadmap & Modern Approaches

> A guide comparing the classic HuggingFace Transformers extractive QA approach with modern LLM-based alternatives, and where each fits in 2025/2026.

---

## Table of Contents

- [NLP Question Answering — Learning Roadmap \& Modern Approaches](#nlp-question-answering--learning-roadmap--modern-approaches)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [The 4 Main QA Approaches](#the-4-main-qa-approaches)
    - [1. Extractive QA (BERT/SQuAD)](#1-extractive-qa-bertsquad)
    - [2. RAG — Retrieval-Augmented Generation](#2-rag--retrieval-augmented-generation)
    - [3. Direct LLM API (Prompting)](#3-direct-llm-api-prompting)
    - [4. Fine-tuning LLMs](#4-fine-tuning-llms)
  - [Comparison Table](#comparison-table)
  - [Is the HuggingFace Transformers Course Still Worth It?](#is-the-huggingface-transformers-course-still-worth-it)
    - [✅ Concepts that are timeless](#-concepts-that-are-timeless)
    - [⚠️ What's outdated in that specific tutorial](#️-whats-outdated-in-that-specific-tutorial)
  - [Learning Path](#learning-path)

---

## Background

The [HuggingFace Transformers QA tutorial](https://huggingface.co/docs/transformers/main/en/tasks/question_answering) teaches **extractive question answering** — fine-tuning models like DistilBERT on the SQuAD dataset to highlight a span of text as an answer.

This approach was state-of-the-art around **2019–2021**. It's still worth understanding, but in 2025/2026 most production QA systems are built differently.

---

## The 4 Main QA Approaches

### 1. Extractive QA (BERT/SQuAD)

**How it works:** The model reads a passage and literally highlights a span of text as the answer. No text generation — purely extraction.

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Note: use the namespaced path (old "squad" path is deprecated)
squad = load_dataset("rajpurkar/squad", split="train[:5000]")

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

**Key preprocessing steps:**
- Truncate only the `context` with `truncation="only_second"`
- Map answer positions with `return_offsets_mapping=True`
- Use `sequence_ids()` to distinguish question tokens from context tokens

| | |
|---|---|
| ✅ **Pros** | Deterministic, fast inference, no hallucination, fully interpretable |
| ❌ **Cons** | Answer must exist verbatim in context, can't synthesize or reason, brittle on complex questions |
| 📌 **Best for** | Simple closed-document QA, legal span extraction, compliance use cases |

---

### 2. RAG — Retrieval-Augmented Generation

**How it works:** A retriever fetches relevant document chunks from a vector database, then an LLM generates an answer grounded in those chunks.

```
User Query
    │
    ▼
[Embedding Model]  →  Query Vector
    │
    ▼
[Vector DB Search]  →  Top-k Relevant Chunks   (FAISS, Chroma, Pinecone)
    │
    ▼
[LLM Prompt]  →  "Answer this question using the context below..."
    │
    ▼
Generated Answer (with sources)
```

**Modern RAG pipeline (3-stage):**
1. **BM25** — keyword-based initial retrieval
2. **Dense embeddings** — semantic similarity search
3. **Cross-encoder reranker** — precise relevance scoring

**Frameworks:**

| Framework | Best For | Notes |
|---|---|---|
| **LlamaIndex** | RAG-first applications | Faster retrieval (~0.8s), better out-of-box RAG accuracy |
| **LangChain** | Complex agent workflows | More flexible, great for multi-step chains and tool use |
| **Both together** | Production systems | LlamaIndex as knowledge layer, LangChain as orchestration |

```python
# LlamaIndex example
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the capital of France?")
print(response)
```

| | |
|---|---|
| ✅ **Pros** | Works on your own documents, updatable knowledge base, grounded answers with sources |
| ❌ **Cons** | Retrieval quality is a bottleneck, multi-hop questions are hard, extra latency |
| 📌 **Best for** | Document QA, internal knowledge bases, customer support bots |

---

### 3. Direct LLM API (Prompting)

**How it works:** Pass context + question directly to a powerful LLM via API. No retrieval infrastructure needed.

```python
from anthropic import Anthropic

client = Anthropic()

context = "..."  # your document or passage
question = "What does the document say about X?"

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above."
        }
    ]
)
print(response.content[0].text)
```

| | |
|---|---|
| ✅ **Pros** | Zero setup, handles reasoning & multi-step synthesis, great out of the box |
| ❌ **Cons** | Context window limits, API cost, knowledge cutoff, no private doc access without engineering |
| 📌 **Best for** | Prototyping, reasoning-heavy QA, when you control the context being passed in |

---

### 4. Fine-tuning LLMs

**How it works:** Train a pre-trained LLM (LLaMA, Mistral) on domain-specific QA pairs to adjust its weights for your use case.

**Modern approach uses PEFT / LoRA** — not full fine-tuning:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

lora_config = LoraConfig(
    r=16,                        # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 6,815,744 || all params: 8,036,564,992 || trainable%: 0.0848
```

| | |
|---|---|
| ✅ **Pros** | Best accuracy for domain-specific tasks, model learns your format and style |
| ❌ **Cons** | Expensive to train, stale knowledge requires retraining, needs labelled data |
| 📌 **Best for** | Specialized domains (medical, legal, finance) with stable and proprietary knowledge |

---

## Comparison Table

| Approach | Setup Effort | Knowledge Updates | Reasoning | Cost | Hallucination Risk |
|---|---|---|---|---|---|
| Extractive QA (BERT/SQuAD) | Low | Retrain required | ❌ None | Low | ✅ None (span-only) |
| RAG (LlamaIndex/LangChain) | Medium | Easy (re-index) | ✅ Good | Medium | ⚠️ Low-Medium |
| Direct LLM API | Very Low | Instant | ✅✅ Excellent | API costs | ⚠️ Medium |
| Fine-tuned LLM | High | Retrain required | ✅✅ Excellent | High | ⚠️ Low-Medium |

---

## Is the HuggingFace Transformers Course Still Worth It?

### ✅ Concepts that are timeless

- **Tokenization** — Every modern LLM (GPT-4, Claude, LLaMA) still uses tokenizers. Understanding `offset_mapping`, `truncation`, and `sequence_ids` is directly transferable
- **Transformer architecture** — Attention, encoders, decoders — this is the bedrock of everything in 2025 AI
- **Fine-tuning mental model** — Even if you use LlamaIndex or LangChain, knowing *why* a model behaves a certain way requires understanding what fine-tuning does to weights
- **HuggingFace ecosystem** — `datasets`, `transformers`, `PEFT`, `Trainer` are still industry-standard tools

### ⚠️ What's outdated in that specific tutorial

| What the tutorial shows | Reality in 2025/2026 |
|---|---|
| DistilBERT for QA | Replaced by generative LLMs for most QA tasks |
| Fine-tuning on SQuAD | SQuAD-style extractive QA is now a narrow use case |
| Full fine-tuning | Mostly replaced by **LoRA / QLoRA** (parameter-efficient) |
| Raw `Trainer` loop | Abstracted by **TRL**, **Axolotl**, **Unsloth** in practice |

---

## Learning Path

Think of it as layers — each builds on the one below:

```
┌──────────────────────────────────────────┐
│  Agents / RAG (LangChain, LlamaIndex)   │  ← Where the jobs are
├──────────────────────────────────────────┤
│  LLM APIs (OpenAI, Anthropic, HF)       │  ← Day-to-day building
├──────────────────────────────────────────┤
│  PEFT / LoRA fine-tuning                │  ← When APIs aren't enough
├──────────────────────────────────────────┤
│  HuggingFace Transformers course        │  ← Foundation ✅ Start here
├──────────────────────────────────────────┤
│  Attention / Transformer math           │  ← Deepens everything above
└──────────────────────────────────────────┘
```

**Recommended order:**

1. ✅ Finish the HuggingFace NLP course (tokenization, fine-tuning concepts)
2. 🔜 Learn `peft` + LoRA — the modern fine-tuning approach
3. 🔜 Build a RAG QA agent with LlamaIndex or LangChain
4. 🔜 Explore HuggingFace's newer courses:
   - [Agents Course](https://huggingface.co/learn/agents-course)
   - [Deep RL Course](https://huggingface.co/learn/deep-rl-course)

---
