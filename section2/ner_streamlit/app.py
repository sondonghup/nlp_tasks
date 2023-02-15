import streamlit as st

col1, col2 = st.columns(2)

st.title("ENG NER SERVICE")
with col1:
    sentence = st.text_input("input")

with col2:
    st.header("col2")