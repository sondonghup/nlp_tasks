import streamlit as st
import requests



def get_ner_result(inputs):
    url = "http://127.0.0.1:8000/acer-lab/ner-eng"
    response = requests.post(
        url = url,
        json = {
            "user_input" : str(inputs)
        }
    )
    return response

st.title("ENG NER SERVICE")

col1, col2 = st.columns(2)

with col1:
    sentence = st.text_input("input")
    if st.button('enter'):
        ner_result = get_ner_result(sentence)


with col2:
    st.header("col2")
    st.markdown(ner_result.json()['result']['ner'])