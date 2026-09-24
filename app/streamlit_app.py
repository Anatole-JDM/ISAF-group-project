"""Client-facing app: the 'interactive application' deliverable.

    python -m src.train          # fit once, writes outputs/ (the model pages read it)
    streamlit run app/streamlit_app.py

The app never fits models. It reads cached artifacts from outputs/ so every
control is instant. Without them, the model pages say so and the rest still works.

Pages, all in the top bar, in the order of the argument (one file each in views/):
    Dataset                      - every stop on a map, with filters
    Problem definition           - objects, selective labels, why we stratify by search type
    White box models             - scorecard (logistic regression)
    Black box models             - gradient-boosted trees (XGBoost)
    Foundation models            - tabular foundation models: TabPFN
    Performance                  - accuracy, ranking agreement, value under a search budget, one stop
    Fairness testing             - the course taxonomy, one tab per model
    Findings & conclusions       - measured signal, recommendation
"""
import streamlit as st

st.set_page_config(page_title="Search Decision Support", page_icon="🚓", layout="wide")

page = st.navigation(
    [
        st.Page("views/dataset.py", title="Dataset", default=True),
        st.Page("views/problem.py", title="Problem definition"),
        st.Page("views/white_box.py", title="White box models"),
        st.Page("views/black_box.py", title="Black box models"),
        st.Page("views/foundation.py", title="Foundation models"),
        st.Page("views/performance.py", title="Performance"),
        st.Page("views/fairness.py", title="Fairness testing"),
        st.Page("views/findings.py", title="Findings & conclusions"),
    ],
    position="top",
)

st.title("Traffic-stop search decision support")
st.caption(
    "Nashville MPD, 2010–2018 · Stanford Open Policing Project · "
    "Pierson et al., *Nature Human Behaviour* 4 (2020)"
)
page.run()
