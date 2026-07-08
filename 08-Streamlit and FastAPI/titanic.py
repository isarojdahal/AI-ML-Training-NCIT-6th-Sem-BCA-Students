import streamlit as st 


st.write("# Titanic Prediction")

st.write("*If the given person were on titanic, would they survive or not. Use our AI model to find out prediction*")

passenger_age = st.number_input("Enter Passenger age",min_value=1, max_value=100) 

# passenger gender dropdown
gender = st.selectbox("Gender",options=["Male","Female"])

p_class = st.selectbox("Class",options=["1","2","3"] )

sib_spouse = st.number_input("Number of Siblings and Spouse",min_value=0,max_value=10)

parent_child = st.number_input("Total Number of Parent and Children ",min_value=0,max_value=10)

if st.button("Predict"):
    import requests 

    prediction = requests.get(f"http://localhost:8000/predict?age={passenger_age}&gender={gender}&p_class={p_class}&sib_sp={sib_spouse}&parent_child={parent_child}")

    st.success(prediction.json())
