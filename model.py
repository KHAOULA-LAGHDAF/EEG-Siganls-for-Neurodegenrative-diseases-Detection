import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def model_page():
    st.title("Model Architecture and Training")

    st.write("""
    ## Overview
    This project employs a deep learning architecture combining an **autoencoder** and a **bidirectional Long Short-Term Memory (LSTM) network** 
    for the classification of resting-state EEG signals into three categories: 
    Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and Cognitively Normal (CN).
    """)

    st.write("""
    ## Feature Extraction with Autoencoder

    ### Data Segmentation
    EEG signals are segmented into overlapping epochs of 5 seconds with a 50% overlap.

    ### Autoencoder
    An autoencoder is employed to extract relevant features from the preprocessed EEG data. The autoencoder consists of:
      - An **encoder network** that learns a lower-dimensional representation of the input.
      - A **decoder network** that reconstructs the original input from the encoded representation.

    The encoder comprises dense layers with ReLU activations. This compressed representation serves as the foundation for subsequent feature analysis and sequence learning.
    """)

    st.write("""
    ## Sequence Learning with Bidirectional LSTM

    ### Sequence Creation
    The features extracted by the autoencoder, together with spectral entropy are combined into feature vectors.
    These are combined into sequences for analysis.

    ### Bidirectional LSTM Network
    The model is composed of multiple layers of:
        -   Bidirectional LSTM layers to capture both forward and backward dependencies in EEG sequences.
        -   Dropout layers to prevent overfitting.
        -   Batch Normalization layers to improve training stability.
    
    The final layers consist of dense layers with ReLU activation followed by a softmax output layer to classify the EEG sequences into one of the three categories.
    """)

    st.write("""
    ## Training Process

    ### Training Details
    - **Optimizer:** Adam
    - **Loss Function:** Categorical Cross-Entropy
    - **Class weights:** Class weights are used to handle class imbalance
    - **Callbacks:** Early Stopping and Reduce Learning Rate on Plateau

    ### Training Performance
    The model achieved a test accuracy of **98%** on the dataset.
    """)

    st.write("""
    ## Feature Engineering
    
    The features used by the bidirectional LSTM are computed using:
        - **Band Power**: Extracted from five frequency bands: delta, theta, alpha, beta, and gamma.
        - **Spectral Entropy**: Captures the randomness and complexity of EEG signals.

    """)

    st.write("""
    ## Model Saving
    The model was saved as `eegmodel.h5`, and `encoder_model.h5` which is used fro real-time application.
    """)