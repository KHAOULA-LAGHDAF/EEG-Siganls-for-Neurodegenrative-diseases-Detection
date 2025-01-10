import streamlit as st
import os

def about_page():
    st.title("About EEG Classification Project")
    # Chemin vers l'image (vérifiez qu'il est correct)
    image_path = "assets/banner.png"

    # Vérifiez si le fichier existe
    if os.path.exists(image_path):
        # Créez trois colonnes pour centrer l'image
        col1, col2, col3 = st.columns([1, 2, 1])  # Ratios ajustés pour un bon centrage
        with col1:
            st.empty()  # Colonne gauche vide
        with col2:
            # Ajustez la largeur avec le paramètre `width`
            st.image(image_path, width=200)  # Changez `300` pour la taille souhaitée
        with col3:
            st.empty()  # Colonne droite vide
    else:
        st.error(f"Image not found at {image_path}! Please check the path or file.")

    st.write("""
        ## Overview
        This project focuses on classifying EEG signals into three categories:
        - Alzheimer's Disease (AD)
        - Frontotemporal Dementia (FTD)
        - Cognitively Normal (CN)
    """)

    st.write("""
    ### Key Features
    - **Participants**:
      - AD Group: 36 subjects
      - FTD Group: 23 subjects
      - CN Group: 29 subjects
    - **Preprocessing**:
      - Band-pass filter: 0.5–45 Hz
      - Artifact removal with ICA and ASR
      - Eye and jaw artifacts rejected
    - **Data**:
      - EEG recordings: 19 channels, sampled at 500 Hz
      - Total duration: ~485 minutes (AD), ~276 minutes (FTD), ~402 minutes (CN)
    """)

    st.write("""
    ### Goals
    - Develop a classification model to differentiate between AD, FTD, and CN.
    - Use advanced techniques like LSTMs for temporal feature extraction.
    """)

# Lancez la fonction
about_page()
