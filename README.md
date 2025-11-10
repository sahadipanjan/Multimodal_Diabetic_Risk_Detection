# Multimodal\_Diabetic\_Risk\_Detection

Implementation of "Multimodal Diabetic Risk Detection using Fundus Images and Voice Stress Data"



\# Multimodal Diabetic Risk Detection



This repository contains the official implementation for the 2025 paper:

\*\*"Multimodal Diabetic Risk Detection using Fundus Images and Voice Stress Data: A Novel Approach for Early Clinical Screening"\*\*.





This project presents a non-invasive, AI-driven screening tool that combines retinal fundus images and voice stress biomarkers to assess diabetic risk. Our model achieves clinical-grade performance, surpassing the 75% benchmarks for sensitivity, specificity, and balanced accuracy.



---



\## 🚀 Key Results



Our multimodal system demonstrates robust performance, validated by 5-fold cross-validation:



\* \*\*Average Balanced Accuracy:\*\* 77.9% ± 3.3% 

\* \*\*Maximum Balanced Accuracy:\*\* 81.8% (Fold 2) 

\* \*\*Average Sensitivity:\*\* 80.9% 

\* \*\*Average Specificity:\*\* 75.3% 

This system is the first of its kind to combine these modalities and meet all three clinical deployment criteria simultaneously.



\## 🛠️ Methodology



The model employs a deep learning ensemble that fuses features from four distinct modalities:



1\.  \*\*Fundus Images:\*\* Processed using \*\*EfficientNetV2B0\*\*.

2\.  \*\*Voice Data:\*\* Features extracted using \*\*BYOL-S/CVT\*\*.

3\.  \*\*Clinical Text:\*\* Sequentially processed with an \*\*LSTM\*\* network.

4\.  \*\*Demographics:\*\* Integrated as additional features.



These features are integrated using a 3-layer Multilayer Perceptron (MLP) and optimized with Focal Loss.



\## 📂 Repository Structure



\## ⚙ï¸  Setup and Installation



1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/YOUR\_USERNAME/multimodal-diabetic-risk.git](https://github.com/YOUR\_USERNAME/multimodal-diabetic-risk.git)

&nbsp;   cd multimodal-diabetic-risk

&nbsp;   ```



2\.  \*\*Install Git LFS:\*\*

&nbsp;   Download and install \[Git LFS](https://git-lfs.github.com/). Then, pull the large data files:

&nbsp;   ```bash

&nbsp;   git lfs install

&nbsp;   git lfs pull

&nbsp;   ```



3\.  \*\*Create a virtual environment (recommended):\*\*

&nbsp;   ```bash

&nbsp;   python -m venv venv

&nbsp;   .\\venv\\Scripts\\activate  # On Windows

&nbsp;   ```



4\.  \*\*Install dependencies:\*\*

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



\## ðŸ”¥ How to Run



To train the full 5-fold cross-validation ensemble from scratch, run the `train.py` script from the root directory:



```bash

python src/train.py



👥 AUTHOR TEAM
Role	Name	Affiliation
Supervisor	Somdatta Patra	Apex Institute of Technology, CU
Co-author	Srijita Das	Apex Institute of Technology, CU
Co-author	Dipanjan Saha	Apex Institute of Technology, CU
Co-author	Aditya Malik	Apex Institute of Technology, CU

All authors are affiliated with the Dept. of Computer Science and Engineering, Apex Institute of Technology, Chandigarh University, Mohali, Punjab, India.

