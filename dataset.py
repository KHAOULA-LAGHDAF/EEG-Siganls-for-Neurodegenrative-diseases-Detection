import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def dataset_page():
    # Titre principal
    st.title("Dataset Overview")
    st.write("""
    ## Dataset Description
    This project uses an open dataset from **OpenNeuro** containing resting-state EEG recordings 
    from 88 subjects classified into three groups:
    - Alzheimer's Disease (AD)
    - Frontotemporal Dementia (FTD)
    - Cognitively Normal (CN)
    """)

    # Tableau récapitulatif des participants
    st.write("### Participants Information")
    data = {
        "Group": ["AD", "FTD", "CN"],
        "Number of Subjects": [36, 23, 29],
        "Average MMSE Score": [17.75, 22.17, 30],
        "Average Age (years)": [66.4, 63.6, 67.9]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    # Graphique circulaire : Répartition des participants
    st.write("### Participants Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))  # Ajustez la taille ici

    # Couleurs personnalisées
    colors = ['#EDDFE0', '#D9EAFD', '#A8CD89']

    # Graphique circulaire
    ax.pie(
        df["Number of Subjects"],
        labels=df["Group"],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors  # Appliquer les couleurs personnalisées
    )
    ax.set_title("Participant Distribution by Group", fontsize=10)
    st.pyplot(fig)

    # Graphique en barres : Scores MMSE
    st.write("### Average MMSE Score by Group")
    fig, ax = plt.subplots(figsize=(6, 4))  # Ajustez la taille ici
    ax.bar(df["Group"], df["Average MMSE Score"], color=["#EDDFE0", "#D9EAFD", "#A8CD89"])
    ax.set_ylabel("MMSE Score", fontsize=10)
    ax.set_title("Average MMSE Score by Group", fontsize=10)
    st.pyplot(fig)

    # Inclure les images pour des visualisations supplémentaires
    st.write("### MMSE Score Distribution by Group")
    st.image("assets/mmse_distribution.png",
             use_container_width=True)
    # Graphique en barres : Âge moyen
    st.write("### Average Age by Group")
    fig, ax = plt.subplots()
    ax.bar(df["Group"], df["Average Age (years)"], color=["#EDDFE0", "#D9EAFD", "#A8CD89"])
    ax.set_title("Average Age by Group", fontsize=10)
    ax.set_ylabel("Age (years)", fontsize=10)
    st.pyplot(fig)



    st.write("### Participant Age Distribution")
    st.image("assets/age_distribution.png", use_container_width=True)

    # Informations sur l'enregistrement
    st.write("""
    ### Recording Details
    - EEG device: Nihon Kohden EEG 2100
    - Number of scalp electrodes: 19
    - Sampling rate: 500 Hz
    - Duration of recordings:
      - AD: 13.5 minutes on average
      - FTD: 12 minutes on average
      - CN: 13.8 minutes on average
    """)

    # Prétraitement
    st.write("""
    ### Preprocessing Steps
    - Band-pass filter: 0.5–45 Hz
    - Artifact removal using:
      - Artifact Subspace Reconstruction (ASR)
      - Independent Component Analysis (ICA)
    - Re-referencing to A1-A2.
    """)

