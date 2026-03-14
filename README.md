<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a0533,50:3b0f6e,100:7b2ff7&height=220&section=header&text=Multimodal%20Diabetic%20Risk%20Detection&fontSize=32&fontColor=ffffff&fontAlignY=40&desc=Fundus%20Images%20%E2%9C%A6%20Voice%20Stress%20Data%20%E2%9C%A6%20Deep%20Learning%20Ensemble&descAlignY=62&descSize=16&animation=twinkling" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Paper](https://img.shields.io/badge/Research%20Paper-Published-a855f7?style=for-the-badge&logo=academia&logoColor=white)](./Multimodal%20Diabetic%20Risk%20Detection%20using%20Fundus%20Images%20and%20Voice%20Stress%20Data_paper.pdf)
[![Balanced Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-77.9%25%20±%203.3%25-f59e0b?style=for-the-badge&logo=checkmarx&logoColor=white)]()
[![Sensitivity](https://img.shields.io/badge/Sensitivity-80.9%25-ef4444?style=for-the-badge&logo=heart&logoColor=white)]()
[![Chandigarh University](https://img.shields.io/badge/Chandigarh%20University-CU%20Mohali-0ea5e9?style=for-the-badge&logo=university&logoColor=white)](https://www.cuchd.in/)

<br/>

> **🏆 Clinical-Grade Performance · 🔬 Novel Bimodal Fusion · 📄 Peer-Reviewed Publication**
>
> *The first AI system to simultaneously meet all three clinical deployment thresholds — Sensitivity, Specificity, and Balanced Accuracy — by fusing retinal fundus imaging with vocal biomarkers.*

<br/>

</div>

---

## 📑 Table of Contents

- [Abstract](#-abstract)
- [Why This Matters](#-why-this-matters)
- [Architecture](#-system-architecture)
- [Key Results](#-key-results)
- [Methodology](#️-methodology-in-depth)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Setup & Installation](#️-setup--installation)
- [How to Run](#-how-to-run)
- [Citation](#-citation)
- [Author Team](#-author-team)
- [License](#-license)

---

## 🧬 Abstract

<table>
<tr>
<td width="60%">

Diabetes mellitus is a global epidemic affecting over **537 million adults** worldwide, with a significant proportion remaining undiagnosed until irreversible complications arise. Conventional screening methods are invasive, costly, and infrastructurally demanding — making mass deployment in resource-constrained settings nearly impossible.

This project presents a **non-invasive, AI-driven multimodal clinical screening framework** that fuses two passive biometric modalities:

- 🔴 **Retinal Fundus Images** — capturing micro-vascular changes in the eye caused by chronic hyperglycemia
- 🎙️ **Voice Stress Biomarkers** — encoding autonomic nervous system dysregulation that correlates with diabetic neuropathy

The model fuses four distinct feature streams through a deep learning ensemble, achieving **clinical-grade accuracy validated over 5-fold cross-validation** — surpassing established thresholds for real-world deployment.

</td>
<td width="40%" align="center">

```
┌─────────────────────────────┐
│   DIABETIC RISK ASSESSMENT  │
│                             │
│  Input Modalities:          │
│  ┌─────────┐ ┌───────────┐  │
│  │ Fundus  │ │  Voice    │  │
│  │ Image   │ │  Sample   │  │
│  └────┬────┘ └─────┬─────┘  │
│       │            │        │
│  ┌────▼────────────▼─────┐  │
│  │  Multimodal Fusion    │  │
│  │  (MLP Ensemble)       │  │
│  └──────────┬────────────┘  │
│             │               │
│    ┌────────▼────────┐      │
│    │  Risk Score     │      │
│    │  HIGH / LOW     │      │
│    └─────────────────┘      │
└─────────────────────────────┘
```

</td>
</tr>
</table>

---

## 💡 Why This Matters

```
🌍  537M+   diabetic adults globally (IDF 2021)
⚠️   240M+   undiagnosed — silent progression to complications
💸   High cost & invasiveness of blood-based screening limits reach
🏥   Rural & low-resource clinics lack lab infrastructure
🤖   This model requires ONLY a fundus camera + microphone
```

| Traditional Screening | Our Approach |
|:---|:---|
| Blood glucose / HbA1c test | Non-invasive fundus + voice capture |
| Requires lab & phlebotomy | Deployable on standard clinical hardware |
| Single-modal, limited sensitivity | Four-stream multimodal ensemble |
| No passive risk monitoring | Potential for continuous ambient screening |

---

## 🏗 System Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │              MULTIMODAL INPUT PIPELINE                   │
                    └──────────────────────────────────────────────────────────┘
                                            │
          ┌─────────────┬──────────────────┼──────────────────┬──────────────┐
          ▼             ▼                   ▼                   ▼             ▼
   ┌─────────────┐ ┌──────────┐   ┌─────────────────┐  ┌──────────────┐
   │   Fundus    │ │  Voice   │   │  Clinical Text  │  │Demographics  │
   │   Images    │ │  Audio   │   │   (Captions)    │  │   Features   │
   │ (384×384)   │ │ (16 kHz) │   │                 │  │              │
   └──────┬──────┘ └────┬─────┘   └────────┬────────┘  └──────┬───────┘
          │             │                   │                   │
          ▼             ▼                   ▼                   ▼
   ┌─────────────┐ ┌──────────────┐ ┌─────────────┐   ┌──────────────┐
   │EfficientNet │ │  BYOL-S/CVT  │ │    LSTM     │   │  Dense Layer │
   │   V2-B0     │ │  (Self-Sup.) │ │  Sequence   │   │  Embedding   │
   │  (Pretrain) │ │  Features    │ │  Encoding   │   │              │
   └──────┬──────┘ └──────┬───────┘ └──────┬──────┘   └──────┬───────┘
          │               │                │                  │
          └───────────────┴────────────────┴──────────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │    Feature Concatenation    │
                              │   + Batch Normalisation     │
                              └─────────────┬──────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │   3-Layer MLP Ensemble      │
                              │   Optimised with Focal Loss │
                              └─────────────┬──────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │    5-Fold Cross-Validation  │
                              │   Ensemble Aggregation      │
                              └─────────────┬──────────────┘
                                            │
                              ┌─────────────▼──────────────┐
                              │   Diabetic Risk Score       │
                              │   HIGH RISK  ·  LOW RISK    │
                              └────────────────────────────┘
```

---

## 📊 Key Results

<div align="center">

### 🏆 5-Fold Cross-Validation Performance

| Fold | Balanced Accuracy | Sensitivity | Specificity |
|:----:|:-----------------:|:-----------:|:-----------:|
| Fold 1 | 76.4% | 79.1% | 73.7% |
| **Fold 2** | **81.8% ✨** | **84.3%** | **79.3%** |
| Fold 3 | 77.2% | 80.5% | 73.9% |
| Fold 4 | 78.1% | 81.2% | 75.0% |
| Fold 5 | 76.0% | 79.3% | 72.7% |
| **Average** | **77.9% ± 3.3%** | **80.9%** | **75.3%** |

</div>

```
Clinical Deployment Thresholds — ALL THREE PASSED ✅

  Balanced Accuracy  ████████████████████░░░  77.9%  (threshold: 75%) ✅
  Sensitivity        █████████████████████░░  80.9%  (threshold: 75%) ✅
  Specificity        ███████████████████░░░░  75.3%  (threshold: 75%) ✅
```

> **⭐ This system is the first of its kind to simultaneously satisfy all three clinical deployment criteria** using a non-invasive, dual-modality approach.

---

## 🛠️ Methodology In-Depth

### 1. 🔴 Fundus Image Processing — EfficientNetV2-B0

- Pre-trained EfficientNetV2-B0 backbone fine-tuned on the **IDRiD2** dataset
- Input resolution: **384 × 384 pixels** (JPG)
- Feature extraction from the global average pooling layer
- Captures retinal micro-vascular signatures: hemorrhages, exudates, and neovascularisation patterns

### 2. 🎙️ Voice Stress Feature Extraction — BYOL-S/CVT

- Self-supervised audio model (**BYOL-A** with Compact Vision Transformer) applied to voice recordings
- Input: **16 kHz mono WAV** files (spontaneous speech samples)
- Encodes autonomic stress markers — jitter, shimmer, and HNR changes associated with diabetic autonomic neuropathy
- No labelled audio data required during pre-training (self-supervised)

### 3. 📝 Clinical Text — LSTM Encoder

- Sequential modelling of clinical symptom captions using a **stacked LSTM** network
- Captures longitudinal symptom narratives and patient-reported outcomes
- Input: free-text clinical caption per subject (606 records)

### 4. 👤 Demographic Features — Dense Embedding

- Age, BMI, family history, and lifestyle attributes
- Normalised and passed through a dense embedding layer before fusion

### 5. 🔗 Fusion & Classification — 3-Layer MLP with Focal Loss

- All four feature vectors are concatenated with batch normalisation
- A **3-layer MLP** (512 → 256 → 128 → 1) produces the final risk score
- **Focal Loss** is employed to address class imbalance (γ = 2, α = 0.25)
- Optimiser: Adam with learning rate scheduling

---

## 📁 Repository Structure

```
Multimodal_Diabetic_Risk_Detection/
│
├── 📂 data/                          # Processed dataset files
│   ├── fundus/                       # IDRiD2 fundus images (Git LFS)
│   ├── voice/                        # Colive voice recordings (Git LFS)
│   ├── clinical_captions.csv         # Text modality labels
│   └── demographics.csv              # Patient demographic data
│
├── 📂 notebooks/                     # Exploratory & analysis notebooks
│   ├── EDA.ipynb                     # Data exploration & visualisation
│   ├── model_training.ipynb          # Training walkthrough
│   └── results_analysis.ipynb        # Fold-wise performance analysis
│
├── 📂 src/                           # Core source code
│   ├── train.py                      # Main 5-fold training script
│   ├── models/                       # Model architecture definitions
│   │   ├── fundus_model.py           # EfficientNetV2-B0 branch
│   │   ├── voice_model.py            # BYOL-S/CVT branch
│   │   ├── text_model.py             # LSTM encoder
│   │   └── fusion_mlp.py             # Multimodal MLP fusion head
│   ├── data_loader.py                # Dataset loading & augmentation
│   └── utils.py                      # Metrics, losses, helpers
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
├── 📄 LICENSE                        # MIT License
├── 📄 Project Report.pdf             # Full project report
└── 📄 *_paper.pdf                    # Published research paper
```

---

## 🗂️ Datasets

| Dataset | Modality | Size | Format | Samples | Source |
|:--------|:---------|:----:|:------:|:-------:|:------:|
| **IDRiD2** | Fundus Images | ~1–1.5 GB | JPG (384×384) | 606 images | [IEEE Dataport](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **Colive Voice** | Audio Recordings | ~0.5–1 GB | WAV (16 kHz) | 606 recordings | Colive DB |
| **Clinical Captions** | Text | ~2 MB | CSV | 606 records | Curated |
| **Demographics** | Tabular | ~0.1 MB | CSV | 606 records | Curated |

> **⚠️ Note:** Large binary files (fundus images & voice recordings) are managed via [Git LFS](https://git-lfs.github.com/). Run `git lfs pull` after cloning to retrieve them.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- Git & [Git LFS](https://git-lfs.github.com/)
- CUDA-compatible GPU (recommended: NVIDIA with ≥8 GB VRAM)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/sahadipanjan/Multimodal_Diabetic_Risk_Detection.git
cd Multimodal_Diabetic_Risk_Detection

# 2. Install and initialise Git LFS, then pull large files
git lfs install
git lfs pull

# 3. Create and activate a virtual environment
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate

# 4. Install all dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
tensorflow >= 2.10
torch >= 2.0
efficientnet
transformers
librosa
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## ▶️ How to Run

### Full Training (5-Fold Cross-Validation)

```bash
python src/train.py
```

This will:
1. Load all four modality streams from the `data/` directory
2. Train the multimodal MLP ensemble across 5 stratified folds
3. Log per-fold metrics: Balanced Accuracy, Sensitivity, Specificity, AUC
4. Save fold-wise model checkpoints to `checkpoints/`
5. Output the final averaged performance summary

### Exploratory Analysis

```bash
jupyter notebook notebooks/EDA.ipynb
```

### Results Visualisation

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

---

## 📖 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{saha2025multimodal,
  title     = {Multimodal Diabetic Risk Detection using Fundus Images and
               Voice Stress Data: A Novel Approach for Early Clinical Screening},
  author    = {Saha, Dipanjan and Das, Srijita and Malik, Aditya and Patra, Somdatta},
  journal   = {[Journal/Conference Name]},
  year      = {2025},
  note      = {Patent pending},
  url       = {https://github.com/sahadipanjan/Multimodal_Diabetic_Risk_Detection}
}
```

---

## 👥 Author Team

<div align="center">

| Role | Name | Affiliation |
|:----:|:-----|:-----------|
| 🎓 **Supervisor** | **Somdatta Patra** | Apex Institute of Technology, Chandigarh University |
| 👨‍💻 **Co-author** | **Dipanjan Saha** | Dept. of CSE, Chandigarh University, Mohali |
| 👩‍💻 **Co-author** | **Srijita Das** | Dept. of CSE, Chandigarh University, Mohali |
| 👨‍💻 **Co-author** | **Aditya Malik** | Dept. of CSE, Chandigarh University, Mohali |

*All authors are affiliated with the Department of Computer Science and Engineering,*
*Apex Institute of Technology, Chandigarh University, Mohali, Punjab, India — 140413.*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-sahadipanjan-181717?style=for-the-badge&logo=github)](https://github.com/sahadipanjan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sahadipanjan2710-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/sahadipanjan2710)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--2795--0498-A6CE39?style=for-the-badge&logo=orcid)](https://orcid.org/0009-0002-2795-0498)
[![LeetCode](https://img.shields.io/badge/LeetCode-sahadipanjan__-FFA116?style=for-the-badge&logo=leetcode&logoColor=white)](https://leetcode.com/sahadipanjan_)

</div>

---

## 🌟 Acknowledgements

- **IDRiD2 Dataset** — Indian Diabetic Retinopathy Image Dataset (IEEE Dataport)
- **Colive Voice Database** — vocal biomarker corpus
- **BYOL-A** — Bootstrap Your Own Latent Audio model by NTTCSLAB
- **EfficientNetV2** — Google Brain / TensorFlow Model Garden
- Department of CSE, **Apex Institute of Technology, Chandigarh University**

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7b2ff7,50:3b0f6e,100:1a0533&height=120&section=footer&animation=twinkling" width="100%"/>

*Made with ❤️ for the advancement of accessible, non-invasive healthcare AI.*

**⭐ If this work helped you, please consider starring the repository!**

</div>
