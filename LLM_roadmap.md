# 🗺️ LLM Learning Roadmap

> Personal roadmap for learning Large Language Models, AI Agents, and the Model Context Protocol — following the HuggingFace course ecosystem and beyond.

**GitHub:** [sakib078/LLM](https://github.com/sakib078/LLM/tree/main)

---

## Table of Contents

- [🗺️ LLM Learning Roadmap](#️-llm-learning-roadmap)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [The Learning Stack](#the-learning-stack)
  - [Phase 1 — LLM Foundations *(In Progress)*](#phase-1--llm-foundations-in-progress)
    - [Syllabus](#syllabus)
    - [Core concepts covered](#core-concepts-covered)
    - [Modern context (what goes beyond the course)](#modern-context-what-goes-beyond-the-course)
  - [Phase 2 — AI Agents](#phase-2--ai-agents)
    - [What you'll learn](#what-youll-learn)
    - [Syllabus](#syllabus-1)
    - [Key frameworks you'll use](#key-frameworks-youll-use)
    - [Prerequisites](#prerequisites)
  - [Phase 3 — Model Context Protocol (MCP)](#phase-3--model-context-protocol-mcp)
    - [What is MCP?](#what-is-mcp)
    - [Syllabus](#syllabus-2)
    - [Prerequisites](#prerequisites-1)
    - [Certification](#certification)
  - [Phase 4 — Building \& Deploying](#phase-4--building--deploying)
  - [Course Index](#course-index)
  - [Progress Tracker](#progress-tracker)
    - [✅ Skills I Already Know](#-skills-i-already-know)
    - [🔄 Currently Learning](#-currently-learning)
    - [⬜ Structured Courses — Upcoming](#-structured-courses--upcoming)
    - [⬜ Skills to Explore — Phase 4](#-skills-to-explore--phase-4)

---

## Overview

This repo documents my journey through the modern LLM/AI stack — from understanding how transformers work to building production-ready AI agents that can use external tools and data via MCP.

**Learning philosophy:** understand the foundations, then build on top of them with modern frameworks. Don't stop at the HuggingFace Transformers tutorial — that's 2019-era. The real work is agents, RAG, and MCP.

---

## The Learning Stack

```
┌──────────────────────────────────────────────────────┐
│  Phase 4 — Deploy & Ship                             │
│  RAG pipelines · REST APIs · HF Spaces · LangSmith  │
├──────────────────────────────────────────────────────┤
│  Phase 3 — MCP (Model Context Protocol)              │  ← Next
│  Tools · Servers · Claude integration · End-to-end  │
├──────────────────────────────────────────────────────┤
│  Phase 2 — AI Agents                                 │  ← Next
│  smolagents · LlamaIndex · LangGraph · Tool use     │
├──────────────────────────────────────────────────────┤
│  Phase 1 — LLM Foundations                           │  ← HERE ✅
│  Transformers · Fine-tuning · NLP tasks · RAG basics │
└──────────────────────────────────────────────────────┘
```

---

## Phase 1 — LLM Foundations *(In Progress)*

**Course:** [🤗 LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)

The HuggingFace LLM course covers both traditional NLP and modern LLM techniques. It evolved from the original NLP course to now emphasize LLMs as the primary paradigm.

### Syllabus

| Chapters | Topic | Status |
|---|---|---|
| 1–4 | Transformers library fundamentals, HF Hub, fine-tuning, sharing models | 🔄 In Progress |
| 5–9 | Datasets, tokenizers, classic NLP tasks (classification, QA, summarization, translation) | ⬜ Upcoming |
| 10–12 | Advanced LLMs — fine-tuning, dataset curation, reasoning models (Open R1) | ⬜ Upcoming |

### Core concepts covered

- **Tokenization** — `offset_mapping`, `truncation`, `sequence_ids`, BPE vs WordPiece
- **Transformer architecture** — Encoder/Decoder/Encoder-Decoder, attention heads
- **NLP tasks** — Text classification, token classification, QA, summarization, translation
- **Fine-tuning** — Full fine-tuning with `Trainer`, evaluating with metrics
- **HuggingFace Hub** — Loading models, pushing checkpoints, `datasets` library

### Modern context (what goes beyond the course)

The course teaches the *foundations*. In practice these have evolved:

| Course teaches | 2025 reality |
|---|---|
| Full fine-tuning | **LoRA / QLoRA** (PEFT) — train 0.08% of parameters |
| SQuAD extractive QA | Generative QA via LLMs + RAG |
| `Trainer` from scratch | **TRL**, **Axolotl**, **Unsloth** in production |
| Single-task models | General-purpose instruction-tuned LLMs |

---

## Phase 2 — AI Agents

**Course:** [🤗 AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction)

> *"From beginner to expert in understanding, using, and building AI agents."*

### What you'll learn

An AI Agent is an LLM that can **reason, plan, and take actions** using tools — not just generate text.

```
User Query
    │
    ▼
[LLM — Think]  →  "I need to search the web to answer this"
    │
    ▼
[Tool Call]    →  web_search("latest AI news")
    │
    ▼
[Observation]  →  Results returned
    │
    ▼
[LLM — Act]    →  Final answer grounded in real data
```

### Syllabus

| Unit | Topic | Description |
|---|---|---|
| 0 | Onboarding | Setup, tools, Discord |
| 1 | Agent Fundamentals | Tools, Thoughts, Actions, Observations. LLMs, messages, special tokens, chat templates |
| 2 | Frameworks | **smolagents**, **LlamaIndex**, **LangGraph** — how fundamentals are implemented |
| 3 | Use Cases | Real-world Agentic RAG applications |
| 4 | Final Project | Build and certify an agent on the student leaderboard |
| Bonus 1 | Fine-tuning for Function Calling | Train an LLM to use tools reliably |
| Bonus 2 | Agent Observability & Evaluation | Monitor, trace, and evaluate agents in production |
| Bonus 3 | Agents in Games | Build a Pokémon battle agent 🎮 |

### Key frameworks you'll use

| Framework | Best for | Notes |
|---|---|---|
| **smolagents** (HF) | Lightweight, fast agent prototyping | HuggingFace native, minimal boilerplate |
| **LangGraph** | Complex stateful agent workflows | Graph-based, great for multi-step reasoning |
| **LlamaIndex** | RAG + agents over documents | Best retrieval accuracy, easy data connectors |

### Prerequisites

- ✅ Basic Python
- ✅ Basic LLM understanding (Phase 1 covers this)

---

## Phase 3 — Model Context Protocol (MCP)

**Course:** [🤗 MCP Course](https://huggingface.co/learn/mcp-course/unit0/introduction) *(built in partnership with Anthropic)*

> *"From beginner to informed in understanding, using, and building applications with MCP."*

### What is MCP?

MCP (Model Context Protocol) is an **open standard** that lets AI models connect to external data sources and tools in a standardized way — like a USB-C port for AI. Instead of every app reinventing how to give Claude/GPT access to your database, MCP creates one protocol they all speak.

```
Without MCP:                    With MCP:
┌─────────┐                    ┌─────────┐
│   LLM   │──custom──▶ Tool A  │   LLM   │──MCP──▶ Any MCP Server
│         │──custom──▶ Tool B  │         │
│         │──custom──▶ Tool C  └─────────┘
└─────────┘
```

### Syllabus

| Unit | Topic | Description |
|---|---|---|
| 0 | Onboarding | Setup, accounts, Discord |
| 1 | MCP Fundamentals | Core concepts, architecture, components. Simple use case |
| 2 | End-to-End Use Case | Build a shareable MCP application |
| 3 | Deployed Use Case | Deploy MCP app using HF ecosystem + partner services |
| 4 | Bonus Units | Partner libraries, advanced integrations |

### Prerequisites

- ✅ Basic AI/LLM understanding
- ✅ API concepts (REST, JSON)
- ✅ Python or TypeScript

### Certification

- **Fundamentals cert** → Complete Unit 1
- **Completion cert** → Complete Units 2 and 3

---

## Phase 4 — Building & Deploying

After completing the courses, the goal is to ship real projects:

| Project Idea | Stack | Concepts Used |
|---|---|---|
| Document QA bot | LlamaIndex + FAISS + Claude API | RAG, embeddings, retrieval |
| Personal research agent | smolagents + web search tool | Agents, tool use, reasoning |
| MCP server for GitHub | MCP SDK + HF Space | MCP, deployment, API |
| Fine-tuned domain model | LoRA + TRL + LLaMA 3 | PEFT, instruction tuning |

---

## Course Index

| Course | Platform | Level | Link |
|---|---|---|---|
| LLM Course | HuggingFace | Beginner → Advanced | [huggingface.co/learn/llm-course](https://huggingface.co/learn/llm-course/chapter1/1) |
| AI Agents Course | HuggingFace | Beginner → Expert | [huggingface.co/learn/agents-course](https://huggingface.co/learn/agents-course/unit0/introduction) |
| MCP Course | HuggingFace × Anthropic | Beginner → Informed | [huggingface.co/learn/mcp-course](https://huggingface.co/learn/mcp-course/unit0/introduction) |
| smol-course (fine-tuning) | HuggingFace | Intermediate | [huggingface.co/learn/smol-course](https://huggingface.co/learn/smol-course/unit0/1) |
| NLP QA deep dive | HuggingFace Transformers Docs | Intermediate | [Transformers QA Task](https://huggingface.co/docs/transformers/main/en/tasks/question_answering) |

----

## Progress Tracker
 
### ✅ Skills I Already Know
 
| Skill | Confidence | Demand | Notes |
|---|---|---|---|
| Transformers & tokenization | Solid | 🔥 Core requirement | HuggingFace `AutoTokenizer`, attention, BPE, WordPiece |
| NLP fundamentals | Solid | ✅ Foundation | Extractive QA, text/token classification, SQuAD format |
| RAG architecture | Conceptual | 🔥 65% of LLM job posts | Pipeline design, dense vs sparse retrieval, chunking |
| Vector databases | Conceptual | 🔥 High demand | Embeddings, similarity search, FAISS / Chroma / Pinecone |
| Prompt engineering | Conceptual | 📈 135% YoY growth | Chain-of-thought, few-shot, system prompts, structured output |
| AI agent frameworks | Conceptual | 🔥 High demand | smolagents, LlamaIndex, LangGraph — concepts |
| MCP | Conceptual | 📈 Emerging standard | Protocol design, clients/servers/tools — concepts |
 
### 🔄 Currently Learning
 
| Task | Course | Status |
|---|---|---|
| Started HuggingFace LLM Course | HF LLM Course | ✅ Done |
| Tokenization & preprocessing for QA | HF LLM Course | ✅ Done |
| Complete Chapters 1–4 (Transformers fundamentals) | HF LLM Course | 🔄 In progress |
| Complete Chapters 5–9 (NLP tasks) | HF LLM Course | 🔄 In progress |
| Complete Chapters 10–12 (Advanced LLMs, Open R1) | HF LLM Course | ⬜ Upcoming |
 
### ⬜ Structured Courses — Upcoming
 
| Milestone | Course | Phase |
|---|---|---|
| Unit 1 — Agent Fundamentals | HF Agents Course | Phase 2 |
| Build first agent with smolagents | HF Agents Course | Phase 2 |
| Unit 2 — Frameworks (smolagents, LlamaIndex, LangGraph) | HF Agents Course | Phase 2 |
| Unit 3 — Agentic RAG use cases | HF Agents Course | Phase 2 |
| Unit 1 — MCP Fundamentals | HF MCP Course | Phase 3 |
| Unit 2 — Build end-to-end MCP application | HF MCP Course | Phase 3 |
| Unit 3 — Deploy MCP app | HF MCP Course | Phase 3 |
 
### ⬜ Skills to Explore — Phase 4
 
| Task | Skill Area | Priority |
|---|---|---|
| Set up LangSmith tracing on a project | LLMOps | 🔴 High |
| Evaluate latency and cost of a RAG pipeline | LLMOps | 🔴 High |
| Fine-tune a small model on a custom dataset | LoRA / PEFT | 🔴 High |
| Complete Agents Bonus Unit 1 (function-calling fine-tune) | LoRA / PEFT | 🔴 High |
| Evaluate a RAG pipeline with RAGAS | Evaluation | 🟡 Medium |
| Write LLM unit tests with DeepEval | Evaluation | 🟡 Medium |
| Run inference with a VLM (Qwen-VL or Gemma 3) | Multimodal | 🟡 Medium |
| Build a multimodal RAG pipeline (images + text) | Multimodal | 🟡 Medium |
| Ship a project combining agents + MCP + evaluation | All | 🏁 Goal |
 
---
 
*Last updated: June 2026*