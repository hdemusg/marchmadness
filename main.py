import streamlit as st

st.title("March Madness Bracket Generator")
st.write("This app generates a bracket for previous NCAA Men's Basketball Tournaments.")
st.write("*After Selection Sunday, bracket prediction for 2025 will be available!*")

year = st.selectbox("Select a year", ["2024"])

