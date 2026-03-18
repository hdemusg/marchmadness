import streamlit as st
from brackets import generate_bracket

st.title("one shining model 🏀")
st.write("This app generates a March Madness bracket using ML!")
st.link_button("GitHub repo", "https://gith", *, help=None, type="secondary", icon=None, icon_position="left", disabled=False, use_container_width=None, width="content", shortcut=None)

with st.form("generate_bracket"):
    year = st.selectbox("Select a year to predict: ", ["2026"])
    model = st.selectbox("Pick a model: ", ["Decision Tree", "Random Forest", "KNeighbors", "Lasso", "XGBoost"])
    matchup_weight = st.selectbox("Should later games have higher weight?", ["Yes", "No"])
    generate = st.form_submit_button("Create my bracket!")

if generate:
    with st.spinner("Generating your bracket..."):
        use_weight = matchup_weight == "Yes"
        result = generate_bracket(model, use_round_weight=use_weight)

    st.success("Bracket generated!")
    st.download_button(
        label="Download Bracket (.xlsx)",
        data=result,
        file_name=f"bracket_{year}_{model.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
