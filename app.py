import streamlit as st
st.title("BMI Calculator")
#user Input
height=st.number_input("Entre your height(in meters):",min_value=0.1)
weight=st.number_input("Entre your weight(in kilogram):",min_value=0.1)
#calculate BMI
if st.button("Calculate BMI"):
    bmi=weight/(height**2)
    st.write(f"Your BMI is :{bmi:2f}")
#interpretation
    if bmi <18.5:
        st.warning("You are underWeight")
    elif 18.5<=bmi<25:
        st.success("you have a normal Weight")
    elif 25<=bmi<30 :
        st.info("You have overweight")
        st.error("You are obese")
