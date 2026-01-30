# Multimodal Diabetic Risk Detection

Implementation of "Multimodal Diabetic Risk Detection using Fundus Images and Voice Stress Data"



This repository contains the official implementation for the academic paper:

# "Multimodal Diabetic Risk Detection using Fundus Images and Voice Stress Data: A Novel Approach for Early Clinical Screening"



This project presents a non-invasive, AI-driven screening tool that combines retinal fundus images and voice stress biomarkers to assess diabetic risk. Our model achieves clinical-grade performance, surpassing the 75% benchmarks for sensitivity, specificity, and balanced accuracy.



-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# 🚀 Key Results



Our multimodal system demonstrates robust performance, validated by 5-fold cross-validation:



Average Balanced Accuracy: 77.9% ± 3.3% 

Maximum Balanced Accuracy: 81.8% (Fold 2) 

Average Sensitivity: 80.9% 

Average Specificity: 75.3% 

This system is the first of its kind to combine these modalities and meet all three clinical deployment criteria simultaneously.



# 🛠️ Methodology



The model employs a deep learning ensemble that fuses features from four distinct modalities:



1.  Fundus Images: Processed using EfficientNetV2B0.

2.  Voice Data: Features extracted using BYOL-S/CVT.

3.  Clinical Text: Sequentially processed with an LSTM network.

4.  Demographics: Integrated as additional features.



These features are integrated using a 3-layer Multilayer Perceptron (MLP) and optimised with Focal Loss.



# 📂 Repository Structure



# ⚙  Setup and Installation

1.  Clone the repository:


&nbsp;   git clone https://github.com/sahadipanjan/Multimodal_Diabetic_Risk_Detection.git

&nbsp;   cd Multimodal_Diabetic_Risk_Detection



2.  Install Git LFS:

&nbsp;   Download and install Git LFS (https://git-lfs.github.com/). Then, pull the large data files:

&nbsp;   git lfs install

&nbsp;   git lfs pull



3.  Create a virtual environment (recommended):


&nbsp;   python -m venv venv

&nbsp;   .\\venv\\Scripts\\activate  # On Windows



4.  Install dependencies:


&nbsp;   pip install -r requirements.txt




#  How to Run



To train the full 5-fold cross-validation ensemble from scratch, run the `train.py` script from the root directory:


&nbsp; python src/train.py




# 📋 DATASETS INCLUDED

Dataset	Size	Format	Samples

&nbsp;IDRiD2 (Fundus)	~1-1.5 GB	JPG (384×384)	606 images

&nbsp;Colive Voice	~0.5-1 GB	WAV (16 kHz)	606 recordings

&nbsp;Clinical Captions	~2 MB	CSV	606 texts

&nbsp;Demographics	~0.1 MB	CSV	606 records




# 👥 AUTHOR TEAM

  Role	        Name	               Affiliation

&nbsp;Supervisor	Somdatta Patra	Apex Institute of Technology, CU

&nbsp;Co-author	  Dipanjan Saha	  Apex Institute of Technology, CU

&nbsp;Co-author	  Srijita Das	    Apex Institute of Technology, CU

&nbsp;Co-author	  Aditya Malik	  Apex Institute of Technology, CU


All authors are affiliated with the Department of Computer Science and Engineering, Apex Institute of Technology, Chandigarh University, Mohali, Punjab, India.

