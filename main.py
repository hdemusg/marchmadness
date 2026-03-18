import streamlit as st
from brackets import generate_bracket

st.title("one shining model 🏀")
st.write("This app generates a March Madness bracket using ML!")
st.link_button("GitHub repo", "https://github.com/hdemusg/marchmadness")
st.link_button("About me", "https://linkedin.com/in/sumedh-garimella")

year = st.selectbox("Select a year to predict: ", ["2026"])
model = st.selectbox("Pick a model: ", ["Decision Tree", "Random Forest", "KNeighbors", "Lasso", "XGBoost", "MLP"])

if model == "KNeighbors":
    hyperparam = st.slider("n_neighbors", min_value=2, max_value=10, value=3, step=1)
elif model == "Decision Tree":
    hyperparam = st.slider("min_samples_split", min_value=2, max_value=10, value=2, step=1)
elif model == "Random Forest":
    hyperparam = st.slider("min_samples_split", min_value=2, max_value=10, value=2, step=1)
elif model == "Lasso":
    hyperparam = st.select_slider("alpha", options=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0], value=0.001)
elif model == "XGBoost":
    hyperparam = st.select_slider("learning_rate", options=[0.025, 0.05, 0.1, 0.2, 0.4], value=0.1)
elif model == "MLP":
    hyperparam = st.select_slider("tol", options=[0.001, 0.005, 0.01, 0.05, 0.1], value=0.01)

with st.form("generate_bracket"):
    matchup_weight = st.selectbox("Should later games have higher weight?", ["Yes", "No"])
    generate = st.form_submit_button("Create my bracket!")

if generate:
    with st.spinner("Generating your bracket..."):
        use_weight = matchup_weight == "Yes"
        result = generate_bracket(model, use_round_weight=use_weight, hyperparam=hyperparam)

    st.success("Bracket generated!")
    st.download_button(
        label="Download Bracket (.xlsx)",
        data=result,
        file_name=f"bracket_{year}_{model.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
