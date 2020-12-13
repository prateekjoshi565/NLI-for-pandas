import pandas as pd
import streamlit as st
from pandas_nli import Pandas_NLI

st.title("Natural Language Interface for pandas")

st.markdown("### Titanic Dataset")

@st.cache
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

st.write(df.head()) 

def show_head(txt):
    nli = Pandas_NLI(df)

    st.markdown("### Output:")

    if len(txt) == 0:
        st.write(txt)
    else:
        nli.query(txt)
        #st.write(nli.query(txt), key = 2)

st.markdown("### Enter your query:")
user_input = st.text_input("")
show_head(user_input)

