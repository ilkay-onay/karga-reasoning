# 🐦‍⬛ Karga-2B-Thinking: Turkish SLM with Chain-of-Thought

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/ilkayO/Karga-Thinking-Demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilkay-onay/karga-reasoning/blob/main/notebooks/Karga_Quickstart.ipynb)

This repository serves as the public showcase and inference codebase for **Karga-2B-Thinking**, a 2-Billion parameter Small Language Model (SLM) trained to perform rigorous step-by-step reasoning in Turkish.

> ⚠️ **Academic Publication Notice (Teaser Repository)** 
> *This repository currently contains the inference engine, evaluation scripts, and architectural overview. The core training pipelines (QLoRA/Unsloth), the synthetic dataset, and the novel **"Deterministic Tensor Injection Agent"** algorithm are temporarily withheld pending double-blind peer review for academic publication. The full source code (GPL v3.0) will be released upon publication.*

## 🎥 Hugging Face Space Demo
[![YıldızSezar Live Demo](https://img.youtube.com/vi/cKtHwEbcZzg/maxresdefault.jpg)](https://youtu.be/cKtHwEbcZzg)

## 📌 Project Overview
Large Language Models (LLMs) often struggle with complex logic and mathematics, especially in low-resource languages. Built on top of `vngrs-ai/Kumru-2B`, **Karga-2B-Thinking** bridges this gap by bringing **Chain-of-Thought (CoT)** capabilities to Edge-friendly models. 

By generating a `<think> ... </think>` block before answering, the model effectively plans its logic, drastically reducing hallucinations. Furthermore, this project pioneers a hybrid Agentic AI approach where mathematical hallucinations are corrected at the tensor level during inference.

## 🚀 Key Highlights
* **Edge AI Ready:** With only 2B parameters, the model easily runs on consumer-grade GPUs, laptops, and IoT devices via 4-bit/8-bit quantization.
* **Native Turkish CoT:** Unlike English models prompting in Turkish, Karga structurally thinks and reasons in native Turkish.
* **Agentic Inference:** Engineered a dynamic inference pipeline that acts as a "Safe Calculator", actively monitoring the generation loop and correcting arithmetic mistakes inside the model's memory (Input IDs tensor).

## 📊 Comparative Benchmark Results
*(Evaluated on a strictly unseen benchmark of 654 unique questions)*

| Category | Baseline (Original Kumru) | Finetuned (Karga-Thinking) | Difference |
| :--- | :--- | :--- | :--- |
| **Overall Average** | 21.71% | **24.31%** | 🟢 + 2.60% |
| **Coding (Python)** | 86.89% | **98.36%** | 🟢 + 11.47% |
| **Mathematics (GSM8K)** | 0.49% | **4.93%** | 🟢 + 4.44% (10x) |
| **Direct Knowledge (QA)** | 24.00% | 21.60% | 🔴 - 2.40% (Alignment Tax)* |

*\*Note: The drop in Direct QA is a well-documented academic phenomenon called the "Alignment Tax" or "Overthinking". Models forced into CoT frameworks tend to overcomplicate simple factual recall tasks.*

## 📂 Repository Structure
- `/src/inference/`: Standard text generation scripts and Hugging Face pipelines.
- `/src/evaluation/`: Scripts utilized for the comparative benchmark generation.
- `/notebooks/`: Jupyter notebooks for quick Colab deployments.
- *`/src/training/`*: (Temporarily private for peer-review).
- *`/data/`*: (Temporarily private for peer-review).

## 💻 Quick Start
Please refer to the `notebooks/Karga_Quickstart.ipynb` for a plug-and-play Google Colab experience, or visit the [Hugging Face Model Card](https://huggingface.co/ilkayO/Karga-2B-Thinking) for Python integration snippets.

## ⚖️ License
* **Source Code:** [GNU GPL v3.0](LICENSE) *(Ensuring open-source integrity for the codebase)*
* **Model Weights:** [Apache 2.0](https://huggingface.co/ilkayO/Karga-2B-Thinking) *(Commercially permissive for model deployment)*

## 🤝 Commercial Integration & Consulting
If your organization is looking to implement advanced GenAI pipelines, Agentic workflows, or requires custom LLM fine-tuning on internal data, feel free to reach out.

📧 **Contact:** ilkayonay2001@gmail.com | [LinkedIn](https://linkedin.com/in/ilkay-onay-391905254)
