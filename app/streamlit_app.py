"""Client-facing app: the 'interactive application' deliverable.

    streamlit run app/streamlit_app.py

Pages, in the order the argument should be made to a client:
    Overview
        1. The problem      - selective labels, stated before any model
        2. Explore          - every stop on a map, with filters (where, when, who, search type)
    Analysis
        3. Pooling          - the headline table
        4. Models           - the three arms compared
        5. Fairness         - the course taxonomy, both Y codings
        6. Budget           - top-K economics and the equity/efficiency frontier
"""
import streamlit as st

st.set_page_config(page_title="Search Decision Support", page_icon="🚓", layout="wide")

pages = {
    "Overview": [
        st.Page("views/problem.py", title="1 · The problem", default=True),
        st.Page("views/explore.py", title="2 · Explore the stops"),
    ],
    "Analysis": [
        st.Page("views/pooling.py", title="3 · Pooling"),
        st.Page("views/models.py", title="4 · Models"),
        st.Page("views/fairness.py", title="5 · Fairness"),
        st.Page("views/budget.py", title="6 · Budget"),
    ],
}
page = st.navigation(pages, position="top")

st.title("Traffic-stop search decision support")
st.caption(
    "Nashville MPD, 2010–2018 · Stanford Open Policing Project · "
    "Pierson et al., *Nature Human Behaviour* 4 (2020)"
)
page.run()
