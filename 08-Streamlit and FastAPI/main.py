import streamlit as st
import pandas as pd
import numpy as np 

st.write("Hello world !")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

st.button("Click me")

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

st.write("<h1>Heading </h1>",unsafe_allow_html=True)

st.markdown("# Markdown Heading")


# forms

num = st.number_input("Enter a number")


if st.button("Calculate"):
    st.write(f"The number is {num}")
    st.write(f"The square of the number is {num * num}")
