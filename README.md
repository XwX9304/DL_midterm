## 🧠 Deep Learning Midterm Project: Math Answer Verification

**Team:** Linglong Li, Zile Yang, Yizhe Liu  
**Course:** Deep Learning (NYU, Fall 2025)  
**Repository:** [XwX9304/DL_midterm](https://github.com/XwX9304/DL_midterm)  
**Project Files:** [Google Drive Folder](https://drive.google.com/drive/folders/1MdSK53v_WeVOBc3myVLTN3DVERaff1yg?usp=drive_link)  (checkpoint-900's performance is the best)

---

### 📘 Overview
This project was developed for the **Math Question Answer Verification Competition**, where the goal is to build a **binary classifier** that determines whether a given mathematical answer is correct or incorrect.

We fine-tuned **Llama-3 8B** using **LoRA-based parameter-efficient fine-tuning** on a cleaned subset of 33k samples.  
Our final model achieved **79.6% test accuracy**, improving over a **60% baseline**.

---

### 🚀 Key Contributions
- **Data Cleaning Pipeline:**  
  Removed markdown/code artifacts and normalized math symbols for cleaner inputs.  
- **LoRA Fine-Tuning:**  
  Applied low-rank adapters (r = 8) to Llama-3 8B with 4-bit quantization for single-GPU training.  
- **Prompt Engineering:**  
  Designed instruction-based prompts to enforce concise True/False outputs.  
- **Hyperparameter Optimization:**  
  Tuned LoRA rank, learning rate, and prompt structure for stable convergence.  

---

### 🧩 Dataset
- **Source:** Math reasoning QA dataset (> 1 M raw samples)  
- **Used Subset:** 30 k train / 3 k validation / 10 k test  
- **Label Balance:** 50.1 % True / 49.9 % False  
- **Average Input Length:** ~420 tokens (question + answer + instruction)

**Preprocessing Steps**
1. Remove `<llm-code>` tags and markdown blocks.  
2. Normalize symbols (e.g., “½” → `1/2`, “√” → `sqrt`).  
3. Compress whitespace and apply Unicode NFKC normalization.  

---

### ⚙️ Methodology

**Model:** Llama-3 8B (4-bit quantized)  
**Adapter:** LoRA (rank = 8, α = 64, dropout = 0.02)  
**Precision:** 4-bit (via bitsandbytes)  

**Prompt Template:**
Question: {question}
Answer: {answer}

Determine if the answer is correct.
Respond with True or False only.


**Training Configuration**
| Hyperparameter | Value |
|----------------|--------|
| Learning Rate | 7.0e-5 |
| Batch Size | 8 |
| Gradient Accumulation | 8 |
| Epochs | 2 |
| Max Seq Length | 2048 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| GPU | A100 (40 GB) |
| Training Time | ~3.5 hours |

---

### 📊 Results

| Experiment | Configuration | Accuracy |
|-------------|----------------|-----------|
| Baseline | r = 1, uncleaned data (60 steps) | 0.60 |
| Prompt + Solution Text | Full context included | 0.75 |
| LoRA Rank 8 | Cleaned data + optimized prompt | **0.796** |

**Final Test Accuracy:** **79.6 %**  
**Improvement:** +19.6 % over baseline

---

### 🔍 Observations
**What Worked**
- Clean data preprocessing (+1–2 %)  
- LoRA rank 8 (+3–4 %)  
- Lower learning rate (7e-5) and cosine scheduler  
- Expanded training samples (30 k → 50 k)  

**What Didn’t Work**
- Including long “solution” texts (caused overfitting)  
- High LoRA ranks (> 32) added instability without gain  

---

### ⚠️ Limitations
- Model sometimes misclassifies **fluent but incorrect** solutions.  
- Limited by single-GPU resources (used 50 k of 1 M samples).  
- LoRA initialization occasionally unstable with 4-bit quantization.  

---

### 📚 References
1. Meta AI. *Llama 3 Model Card.* (2024)  https://github.com/meta-llama/llama3  
2. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022  
