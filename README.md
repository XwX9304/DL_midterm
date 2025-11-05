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
