import streamlit as st
import numpy as np
import mne
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from scipy.stats import entropy, mode
import scipy.signal as signal
import scipy.integrate as integrate
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.initializers import Orthogonal

def extract_band_power(segments, frequency_bands, sampling_rate):
    """
    Extrait la puissance des bandes de fréquence pour chaque segment EEG.
    """
    band_powers = []
    for segment in segments:
        freqs, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
        bp = []
        for band, (low_freq, high_freq) in frequency_bands.items():
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            power = integrate.simpson(psd[idx_band], freqs[idx_band])
            bp.append(power)
        band_powers.append(bp)
    return np.array(band_powers)

def calculate_spectral_entropy(band_powers):
    """
    Calcule l'entropie spectrale à partir des puissances des bandes de fréquences.
    """
    normalized_powers = band_powers / np.sum(band_powers, axis=1, keepdims=True)
    return entropy(normalized_powers, axis=1)

def process_new_eeg(eeg_file_path, model_path, encoder_path):
    """
    Traite un nouveau fichier EEG (.set) avec le même pipeline que dans le notebook.
    Retourne la prédiction finale et les scores de confiance.
    """
    classifier_model = keras.models.load_model(model_path)  # Modèle LSTM final
    encoder_model = keras.models.load_model(encoder_path)   # Autoencodeur/Encodeur

    # Paramètres
    segment_length_sec = 5
    sampling_rate_raw = 500
    segment_length = segment_length_sec * sampling_rate_raw
    overlap_ratio = 0.5
    overlap_step = int(segment_length * (1 - overlap_ratio))

    # Fréquence d'échantillonnage pour l'extraction de features
    sampling_rate_features = 128
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 25),
        'gamma': (25, 45)
    }
    sequence_length = 10

    # Lecture EEG
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)
    data = raw.get_data()  # (n_channels, n_samples)

    # Segmenter
    segments = []
    start = 0
    while start + segment_length <= data.shape[1]:
        segment = data[:, start:start + segment_length]
        segments.append(segment)
        start += overlap_step
    segments = np.array(segments)  # (N, n_channels, segment_length)

    if len(segments) == 0:
        return None, {}

    # Standardiser
    standardized_segments = []
    for seg in segments:
        scaler = StandardScaler()
        seg_std = scaler.fit_transform(seg)
        standardized_segments.append(seg_std)
    standardized_segments = np.array(standardized_segments)

    # Encoder
    encoder_output = encoder_model.predict(standardized_segments)

    # Extraire features
    band_powers = extract_band_power(encoder_output, frequency_bands, sampling_rate_features)
    spectral_entropy = calculate_spectral_entropy(band_powers)
    combined_features = np.hstack((band_powers, spectral_entropy.reshape(-1, 1)))

    # Créer séquences
    all_sequences = []
    for i in range(len(combined_features) - sequence_length + 1):
        seq = combined_features[i:i + sequence_length]
        all_sequences.append(seq)
    all_sequences = np.array(all_sequences)  # (N_sequences, sequence_length, nb_features)

    if len(all_sequences) == 0:
        return None, {}

    # Prédiction
    predictions = classifier_model.predict(all_sequences)

    # Mapping
    label_mapping = {
        0: 'Alzheimer',
        1: 'Control',
        2: 'Frontotemporal'
    }

    predicted_classes = np.argmax(predictions, axis=1)
    final_prediction_majority = mode(predicted_classes, keepdims=True).mode[0]
    final_class_label = label_mapping[final_prediction_majority]

    counts = Counter(predicted_classes)
    total = len(predicted_classes)
    confidence_scores = {}
    for class_idx, c in counts.items():
        confidence_scores[label_mapping[class_idx]] = (c / total) * 100.0

    return final_class_label, confidence_scores

def simulation_page():
    st.title("EEG Signal Simulation")
    st.write("Veuillez téléverser un fichier EEG au format `.set` pour lancer la classification.")

    MODEL_PATH = "models/eegmodelsim.h5"
    ENCODER_PATH = "models/encoder_modelsim.h5"

    # -- Initialiser la clé session_state s'il n'existe pas --
    if "uploaded_set_path" not in st.session_state:
        st.session_state["uploaded_set_path"] = None

    eeg_file = st.file_uploader("Upload EEG file", type=["set"])

    if eeg_file is not None:
        # 1) On crée un fichier temporaire.
        #    NE PAS le supprimer ensuite, car on le veut pour la "réal time" plus tard.
        temp_file_path = f"temp_{eeg_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(eeg_file.read())

        # 2) On stocke ce path dans la session pour le réutiliser
        st.session_state["uploaded_set_path"] = temp_file_path

        with st.spinner("Processing EEG file..."):
            final_prediction, confidence = process_new_eeg(
                eeg_file_path=temp_file_path,
                model_path=MODEL_PATH,
                encoder_path=ENCODER_PATH
            )

        # <-- On NE supprime pas le fichier ici
        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)

        if final_prediction is None:
            st.error("Impossible de segmenter ou de créer des séquences sur ce fichier EEG. Fichier trop court ?")
        else:
            st.success(f"Classe prédite (majorité) : {final_prediction}")

            st.write("**Scores de confiance (pourcentage de chaque classe sur toutes les séquences)** :")
            for cls_name, score in confidence.items():
                st.write(f"- {cls_name}: {score:.2f}%")

            ##################################
            #  Bouton "Predict in real time"
            ##################################
            if st.button("Predict in real time"):
                st.info("Affichage de la prédiction pour chaque séquence (avec graphe)...")
                # Appel de la fonction temps réel,
                # on lui passe le chemin sauvegardé
                realtime_prediction(st.session_state["uploaded_set_path"])

def realtime_prediction(eeg_file_path):
    """
    Lit à nouveau le EEG, et pour chaque séquence, affiche un graphe + la prédiction.
    """
    MODEL_PATH = "models/eegmodelsim.h5"
    ENCODER_PATH = "models/encoder_modelsim.h5"

    classifier_model = load_model(MODEL_PATH)
    encoder_model = load_model(ENCODER_PATH)

    segment_length_sec = 5
    sampling_rate_raw = 500
    segment_length = segment_length_sec * sampling_rate_raw
    overlap_ratio = 0.5
    overlap_step = int(segment_length * (1 - overlap_ratio))
    sampling_rate_features = 128
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 25),
        'gamma': (25, 45)
    }
    sequence_length = 10
    label_mapping = {
        0: 'Alzheimer',
        1: 'Control',
        2: 'Frontotemporal'
    }

    # Lecture EEG
    raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)
    data = raw.get_data()  # (n_channels, n_samples)

    # Segmenter
    segments = []
    start = 0
    while start + segment_length <= data.shape[1]:
        seg = data[:, start:start+segment_length]
        segments.append(seg)
        start += overlap_step
    segments = np.array(segments)

    if len(segments) == 0:
        st.warning("Fichier trop court, pas de segments à afficher.")
        return

    total_sequences = max(0, len(segments) - sequence_length + 1)
    if total_sequences == 0:
        st.warning("Pas assez de segments pour former une séquence de 10 segments.")
        return

    progress_bar = st.progress(0)
    status_txt = st.empty()

    seq_details = []

    for seq_index in range(total_sequences):
        status_txt.text(f"Traitement de la séquence {seq_index+1}/{total_sequences} ...")

        seq_segments = segments[seq_index : seq_index + sequence_length]

        # Graphe 1er segment
        fig, ax = plt.subplots(figsize=(6,3))
        first_segment = seq_segments[0]
        time_points = np.arange(first_segment.shape[1])
        for ch in range(first_segment.shape[0]):
            ax.plot(time_points, first_segment[ch, :])
        ax.set_title(f"Séquence #{seq_index+1} - 1er segment (tous canaux)")
        ax.set_xlabel("Time (points)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        # Standardiser + Encoder
        standardized_segments = []
        for seg in seq_segments:
            scaler = StandardScaler()
            seg_std = scaler.fit_transform(seg)
            standardized_segments.append(seg_std)
        standardized_segments = np.array(standardized_segments)

        encoded_segs = encoder_model.predict(standardized_segments)
        band_powers = extract_band_power(encoded_segs, frequency_bands, sampling_rate_features)
        spectral_entropy = calculate_spectral_entropy(band_powers)
        combined_features = np.hstack((band_powers, spectral_entropy.reshape(-1,1)))

        # Prédiction sur cette unique séquence
        input_seq = combined_features[np.newaxis, ...]  # (1,10,X)
        preds = classifier_model.predict(input_seq)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_str = label_mapping[pred_idx]

        st.write(f"**Prédiction de la séquence #{seq_index+1}** : {pred_str}")
        seq_details.append({"Sequence_index": seq_index+1, "Predicted_Class": pred_str})

        progress_bar.progress(int((seq_index+1)/total_sequences*100))
        st.markdown("<hr>", unsafe_allow_html=True)

    st.write("**Tableau récapitulatif**")
    st.table(seq_details)
