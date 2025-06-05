# MissFortune - Prompt Tuning Service

This repository provides a lightweight **Prompt Tuning** pipeline. It fine-tunes prompts to generate domain-specific **IT interview questions** from structured `.jsonl` data — useful for HR systems, mock interview apps, or AI tutors.

---

## 🚀 Features

- 🔧 Prompt Tuning with PEFT (Parameter-Efficient Fine-Tuning)
- 🤖 Based on `google/flan-t5-small` (Seq2Seq encoder-decoder model)
- 📥 Load `.jsonl` training data using HuggingFace `datasets`
- ✍️ Train using structured prompts and completions
- 🧠 Custom prompts after training (free-form inference)
- ⚡ Lightweight and fast — suitable for local or cloud inference

---