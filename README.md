# 🧠 EEG Signal Classification for Neurodegenerative Disease Detection

This project focuses on classifying resting-state EEG signals to aid in the diagnosis of Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and to differentiate them from cognitively normal controls (CN). We employ advanced signal processing techniques and a deep learning model architecture to achieve high classification accuracy and provide interpretable insights.

## 📜 Overview

This project aims to differentiate between three categories based on resting-state EEG signals:

*   **Alzheimer's Disease (AD)**
*   **Frontotemporal Dementia (FTD)**
*   **Cognitively Normal (CN)**

Our approach utilizes a deep learning model that integrates an Autoencoder for feature extraction and a Bidirectional Long Short-Term Memory (LSTM) network for sequential data modeling.

## 🌟 Key Features

*   **Advanced Preprocessing:** Band-pass filtering, artifact removal techniques (ICA and ASR) for robust data quality.
*   **Deep Learning Model:** Combination of Autoencoder for feature extraction and Bidirectional LSTM for temporal sequence analysis.
*   **Interpretability:** SHAP (SHapley Additive exPlanations) to provide insights into feature importance.
*   **High Accuracy:** Achieves a test accuracy of 98%.

## 📊 Dataset Details

### 📚 Dataset Overview

The EEG dataset used in this project is publicly available on [OpenNeuro](https://openneuro.org/datasets/ds004504/versions/1.0.8), titled "A dataset of EEG recordings from Alzheimer's Disease, Frontotemporal Dementia, and Healthy Subjects".

This dataset includes resting-state EEG recordings from 88 participants, categorized into:
    *   Alzheimer's Disease (AD)
    *   Frontotemporal Dementia (FTD)
    *   Cognitively Normal (CN)

### 👥 Participants Information

*   **Distribution:**
    *   AD Group: 36 subjects
    *   FTD Group: 23 subjects
    *   CN Group: 29 subjects

*   **Average MMSE Scores (Mini-Mental State Examination):**
    *   AD: ~19.4
    *   FTD: ~23.7
    *   CN: ~29.5

*   **Average Age:**
    *   AD: ~72 years
    *   FTD: ~66 years
    *   CN: ~60 years

### ⚙️ Recording Details

*   **EEG Device:** Nihon Kohden EEG 2100
*   **Number of Electrodes:** 19 (Standard 10-20 System)
*   **Sampling Rate:** 500 Hz
*   **Average Recording Durations:**
    *   AD: ~13.5 minutes
    *   FTD: ~12 minutes
    *   CN: ~13.8 minutes
*  **Total Recording Durations:**
    * AD: ~485 minutes
    * FTD: ~276 minutes
    * CN: ~402 minutes

## 🔧 Preprocessing Steps

1.  **Band-pass Filtering:** 0.5–45 Hz to remove low-frequency drift and high-frequency noise.
2.  **Artifact Removal:**
    *   Artifact Subspace Reconstruction (ASR)
    *   Independent Component Analysis (ICA)
    *   Rejection of "eye" and "jaw" artifacts.
3. **Re-referencing:** EEG signals were re-referenced to the average of electrodes A1 and A2.

## 🏗️ Model Architecture and Training

### 🌟 Feature Extraction with Autoencoder

*   **Data Segmentation:** EEG signals were divided into overlapping 5-second epochs with a 50% overlap.
*   **Autoencoder:**
    *   **Encoder:** Dense layers with ReLU activations for compressed feature extraction.
    *   **Decoder:** Dense layers to reconstruct the original signal.
    *   The Autoencoder learns a compact representation of the EEG segments, preserving essential information while reducing dimensionality.

### 🔄 Sequence Learning with Bidirectional LSTM

*   **Sequence Creation:** Autoencoder-extracted features are combined with spectral entropy values into feature vectors for sequence analysis.
*   **Bidirectional LSTM:**
    *   Bidirectional LSTM layers to capture both forward and backward temporal dependencies.
    *   Dropout layers to prevent overfitting.
    *   Batch Normalization layers to enhance training stability.
    *   Output: Dense layers with ReLU activations followed by a Softmax layer for classification.
    *   The Bi-LSTM layers enable accurate modeling of sequential EEG data, capturing the time-based aspects of the data.

### ⚙️ Training Process

*   **Optimizer:** Adam
*   **Loss Function:** Categorical Cross-Entropy
*   **Class Weights:** Applied to address class imbalance.
*   **Callbacks:**
    *   Early stopping
    *   Reduce learning rate on plateau

### 🎯 Training Performance

*   **Test Accuracy:** 98%

## 📋 Feature Engineering

The following features were used for classification:

*   **Band Power:** Extracted from the frequency bands:
    *   Delta (1-4 Hz)
    *   Theta (4-8 Hz)
    *   Alpha (8-13 Hz)
    *   Beta (13-30 Hz)
    *   Gamma (30-60 Hz)
*   **Spectral Entropy:** Quantifies the randomness and complexity of EEG signals.

## 💾 Model Saving

*   **Model File:** `eegmodel.h5`
*   **Encoder File:** `encoder_model.h5`

These trained models are used for evaluation and can be integrated into real-time classification applications.

## 📁 Project Structure
EEG-Classification/
├── data/ # Preprocessed EEG data
├── EEGSimulation/ # the interface where we deployed our model
│ ├── .idea
│ ├── assets
│ ├── data
│ ├── .venv
│ ├── models
├── notebooks/
│ ├── Final_Model # here is the main deep learning model we trained
│ ├── models # here is the KNN and SVM benchmarking on the dataset to see the results
│ ├── XIA # here where we did the explainbility IA techniques
├── README.md # Documentation

* **Interface location:** The interface can be found on the `Project_Interface` folder

## 🚀 Installation

To set up the project:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/eeg-classification.git
    cd eeg-classification
    ```
2.  **Install Dependencies:** See Requirements.txt
