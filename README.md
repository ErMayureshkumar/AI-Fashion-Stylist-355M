# AI-Fashion-Stylist-355M
From Scratch 355M Parameter AI Fashion Stylist: Two Stage Supervised Fine-Tuning with Synthetic Data Distillation and Forgot Aware Domain Adaptation.

## 1. Project Introduction

Large general-purpose language models are powerful, but they are often **cloud-dependent, costly to run, and not ideal for privacy-sensitive or low-resource settings**. This project explores a small, from-scratch **355M parameter GPT-style model** specialized as an **AI fashion stylist**, with the explicit goal of running locally on modest hardware.

The training pipeline is:

1. **Pretraining**: Train a 355M causal language model from scratch on general text.  
2. **Instruction Tuning**: Supervised fine-tuning for general chat and instruction following.  
3. **Fashion Tuning**: Domain-specific supervised fine-tuning using synthetic fashion data with forgetting-aware strategies to preserve general abilities.

The result is a **first-generation AI fashion stylist** that can provide text-based outfit suggestions, wardrobe planning tips, and style advice, while maintaining reasonable general conversational ability.

---

## 2. Motivation and Goals

This project is driven by two main lines of motivation:

- **Model behavior motivation**
  - Show that **instruction tuning** improves general chat and instruction-following ability.
  - Show that **fashion-specific tuning** improves fashion expertise and style advice quality.
  - Demonstrate that fashion tuning **does not significantly degrade general chat ability** in the first-generation model.

- **Deployment motivation**
  - Build an **offline-capable**, **cost-effective**, **mobile-first**, and **privacy-preserving** AI fashion stylist.
  - Enable local deployment on consumer hardware instead of relying on remote cloud APIs.
  - Use this model as a **research artifact** for studying domain adaptation and catastrophic forgetting in small LMs.

In summary, the goal of this first generation is to establish a strong, well-documented **baseline**: a small local-first stylist whose chat ability is improved by instruction tuning, whose fashion expertise is improved by domain tuning, and whose general abilities are largely preserved. Future versions can then focus on deeper reasoning, multimodality, and richer personalization.

---

## 3. Applications

The model is designed for the following **research and prototype** applications:

- **Personal styling assistant**  
  Provide outfit suggestions based on occasion, body type, climate/season, budget, and preferred style.

- **Wardrobe planning tool**  
  Help users build capsule wardrobes, plan outfits for trips or weekly office wear, and identify versatile key pieces.

- **Fashion advice chatbot**  
  Answer questions about color matching, fabric choices, dress codes, and basic grooming/accessory decisions.

- **Regional fashion guidance**  
  Offer suggestions tailored to **Indian occasions and contexts** (e.g., weddings, festivals, office wear, climate considerations).

- **Local-first mobile assistant**  
  Serve as a **mobile-friendly, offline-capable** stylist for users who prefer not to send personal prompts to the cloud.

- **Research prototype**  
  Act as a concrete example for studying **instruction tuning, domain-specific synthetic distillation,** and **forgetting-aware training** in small language models.

> **Intended Use:** The model and this repository are intended for **academic and research purposes only** and are **not** optimized or evaluated for commercial deployment.

---

## 4. Limitations

Despite careful design, the model has several important limitations:

- **Text-only model**  
  It cannot process or understand images (e.g., outfit photos, body photos, wardrobe pictures) and works only with text descriptions.

- **First-generation stylist**  
  Some suggestions may be **generic**, overly conservative, or not fully aligned with a user’s personal taste or local micro-trends.

- **Small model capacity (355M parameters)**  
  Compared to multi-billion parameter models, it has limited capacity for nuanced reasoning, very long context, and extremely fine-grained personalization.

- **Potential biases**  
  Recommendations may reflect **biases in the training data**, including cultural, regional, or body-type biases. It may over-fit to Indian/English-centric contexts.

- **No real-time trend awareness**  
  Offline and local-first design means the model does **not** automatically track live fashion trends, new brands, or changing social norms.

- **Not a professional stylist**  
  It is **not** a substitute for professional styling advice, and users should critically review outputs, especially in sensitive or high-stakes contexts.

- **Academic / research only**  
  This project is currently intended **only for academic and research use**, not for commercial deployment.

---

## 5. Roadmap and Future Work

Planned directions for future versions include:

- **Multimodal version**  
  Extend the model to accept **images of outfits, user body photos, or wardrobe snapshots** to provide more accurate, visually grounded styling advice.

- **More languages and regions**  
  Expand support beyond English and Indian contexts to cover more **languages**, **regions**, and **cultural dress codes**.

- **Richer personalization**  
  Incorporate user preference profiles, history, and style archetypes for more robust **personalized recommendations** over time.

- **Stronger reasoning and planning**  
  Enhance multi-step reasoning so the model can plan **multi-day outfits**, seasonal wardrobe transitions, and cohesive style evolution.

- **Deployment optimizations**  
  Explore quantization and further model compression for **faster mobile/local inference** with minimal quality loss.

- **Improved alignment and safety**  
  Investigate light-weight reasoning/RL alignment (e.g., GRPO/DPO) for better explanation quality and safer handling of sensitive fashion/body-image questions.

---

## 6. Model Usage Details

> The actual model is hosted on Hugging Face. The following describes how to use it once installed.

### 6.1 Basic loading (Python)

```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import os
import torch

MODEL_ID = "ErMayureshKumar/ai-fashion-stylist-355m"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_tokens = {
    "additional_special_tokens": [
        "<|user|>",
        "<|assistant|>"
    ]
}

tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = """
Suggest an outfit for a summer wedding in Mumbai
for a 26-year-old woman.
"""

def chat(prompt):

    text = f"<|user|>\n{prompt}\n\n<|assistant|>\n"
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False
    )
    return result

print(chat(prompt))
```
---

## 7. Hugging Face Model and Demo Links

> Replace the placeholders below with your actual URLs after you upload.

- **Model (Hugging Face Hub)**  
  `HF_MODEL_LINK`  
  Example: <https://huggingface.co/ErMayureshKumar/ai-fashion-stylist-355m>

- **Interactive Demo (Hugging Face Space)**  
  `HF_SPACE_LINK`  
  Example: <https://huggingface.co/spaces/ErMayureshKumar/AI-Fashion-Stylist-355M>

The Space demo provides a simple chat interface where you can type queries and see model responses in the browser.

---

## 8. Methodology (High-Level)

This section summarizes the training pipeline.

### 8.1 Model and Pretraining

- **Architecture**: GPT-style decoder-only transformer, ~355M parameters (e.g., 16 layers, 12 heads, 1024-dim hidden).  
- **Objective**: Causal language modeling on general text.  
- **Data**: General-purpose text corpus (no fashion-specific data used in pretraining).  
- **Goal**: Obtain a **domain-agnostic base LM** with solid language modeling capability.

### 8.2 Stage 1 – General Instruction Tuning

- **Data**: Mixture of instruction-following and chat datasets (QA, explanations, basic reasoning, rewriting, etc.).  
- **Format**: Unified instruction–input–response format, with loss computed only on assistant responses.  
- **Objective**: Improve **chat ability**, instruction following, and general helpfulness.  
- **Outcome**: A Stage-1 model that behaves as a small, general-purpose assistant.

### 8.3 Stage 2 – Fashion-Specific Tuning with Synthetic Distillation

- **Fashion topic taxonomy**:  
  - Occasions (weddings, office, parties, festivals, travel)  
  - Body types (plus-size, petite, curvy, tall, athletic)  
  - Wardrobe planning (capsule wardrobes, office basics, seasonal transitions)  
  - Regional/Indian fashion (sarees, lehengas, kurtas, Indo-western)  
  - Fashion psychology and confidence (dressing for confidence, comfort vs style)

- **Synthetic data generation**:  
  - Use a stronger teacher model to generate **fashion Q&A pairs** guided by the topic taxonomy and parameterized prompts (age, gender, city, season, budget, body type).  
  - Filter and balance the dataset to cover diverse scenarios.

- **Forgetting-aware training**:  
  - Mix a portion of **general SFT data** into each batch during fashion tuning (rehearsal).  
  - Optionally add a regularization term to limit harmful drift from the Stage-1 weights.  
  - Monitor general evaluation metrics to quantify catastrophic forgetting.

- **Objective**: Improve **fashion expertise and stylist-quality responses** while preserving most general chat ability.

### 8.4 Evaluation

- **General evaluation**  
  - Use a held-out general instruction set to measure performance of Base, Stage-1, and Stage-2 models and to quantify forgetting.

- **Fashion evaluation**  
  - Automatic: compare model outputs against reference fashion answers using similarity metrics.  
  - Human/model-judge: pairwise evaluations where raters/judges choose between Stage-1 and Stage-2 answers for fashion prompts.

- **Key findings (intended)**  
  - Instruction tuning improves general chat ability.  
  - Fashion tuning improves fashion expertise substantially.  
  - Fashion tuning does **not** significantly degrade general chat ability in this first-generation model.

---

## 9. Intended Use and License

- **Intended Use**:  
  This project and model are intended **for academic and research purposes only**. They are not designed, tested, or certified for commercial use.

- **Non-goals**:  
  The model is **not** meant to replace professional stylists, nor to be used for sensitive decision-making about health, body image, or identity.

- **License**:  
  Educational and research-only license

---

## 10. Hardware and Performance

This project targets **small, local-first deployment**, and the entire 355M training pipeline — including **pretraining, Stage 1, and Stage 2** — was executed on a **single consumer laptop**, without any data-center GPUs.

### 10.1 Actual hardware for the entire pipeline

All three phases (pretraining on Cosmopedia, general instruction tuning, and fashion domain tuning) were run on:

- **Machine**: Alienware 16 Aurora AC16250 (laptop-class gaming system).
- **GPU**: NVIDIA RTX 5060 Laptop GPU with **8 GB VRAM**.
- **CPU**: Intel Core i7-240H.
- **System RAM**: 16 GB.
- **Storage**: NVMe SSD.

On this hardware, we trained over **1.6B tokens total**:

- **Pretraining**: ~1.18B tokens from the Cosmopedia dataset subset.
- **Stage 1**: ~0.39B tokens from Alpaca, Dolly, OpenAssistant, and UltraChat.
- **Stage 2**: ~0.4B tokens combining 30% Stage 1 data, `nreimers/fashion-dataset`, and synthetic Q&A from GPT OSS 20B and LLaMA 3.2 3B.

To make this feasible on 8 GB VRAM, we relied on:

- Mixed-precision training (FP16) to reduce memory footprint.
- Gradient accumulation to simulate larger effective batch sizes.
- Careful tuning of context length and batch size to avoid out-of-memory errors.
- Longer wall-clock training times compared to workstation or data-center setups.

This demonstrates that **pretraining + SFT for a 355M model on 1.6B tokens** is possible on a single mid-range laptop GPU with enough patience and optimization.
This makes the model well-suited for **local-first fashion styling assistants** on gaming laptops and modest desktops, without any cloud GPU dependency.

