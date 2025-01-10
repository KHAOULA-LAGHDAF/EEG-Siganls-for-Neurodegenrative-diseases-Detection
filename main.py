import streamlit as st
from about import about_page
from simulation import simulation_page
from dataset import dataset_page  # Importez dataset_page depuis dataset.py
from model import model_page

# Ajouter des sections vides si nécessaire
def model_info_page():
    st.title("Model Info")
    st.write("This section will provide details about the models used.")

# Menu horizontal
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["About", "Simulation", "Dataset", "Model Info"]
)

# Naviguer entre les pages
if menu == "About":
    about_page()
elif menu == "Simulation":
    simulation_page()
elif menu == "Dataset":
    dataset_page()
elif menu == "Model Info":
    model_page()
