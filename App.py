import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
st.title("Disease  prediction App")
model=joblib.load('Diabetes_result.pkl')
heart_disease=joblib.load('Heart_disease_Prediction.pkl')
# Ask the user to choose a method
tab1,tab2=st.tabs(['Diabetes_prediction','Heart disease Prediction'])
with tab1:
    method = st.radio("Choose input method:", ["Load Dataset", "Fill Parameters"],key='display_choice')
    if method == "Load Dataset":
        fichier = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if fichier:
            data_set=pd.read_csv(fichier,usecols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
            st.success("Dataset loaded successfully!")
            st.dataframe(data_set)
            if st.button('Predict'):
                st.write('**********Prediction**********')
                label={0:'Not_Diabetic',1:'Diabetic'}
                st.write(label)
                input_data=data_set
                prediction=model.predict(input_data)
                data_set['Predictions']=prediction
                st.dataframe(data_set)
                # Optionally download the result
                csv = data_set.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")
        else:
            st.info("Please upload a CSV file.")

    elif method == "Fill Parameters":
        with st.form(key='User_info_form'):
            st.header('Fill the Parameters')
            st.divider()
            pregnancies= st.number_input('What is thre total number of pregnancies',min_value=1,max_value=18)
            Glucose=st.number_input('What is your Glucose Level')
            BloodPressure=st.number_input('What is your BloodPressure')
            SkinThickness=st.number_input('What is your SkinThickness')
            Insulin=st.number_input('What is your Insulin Level')
            BMI=st.number_input('What is your BMI')
            DiabetesPedigreeFunction=st.number_input('What is your DiabetesPedigreeFunction Level')
            Age=st.number_input('What is Your Age',max_value=80,min_value=1)
            submit_button=st.form_submit_button(label='Prediction')
            if submit_button:
                data=np.array([pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,-1)
                prediction=model.predict(data)
                if prediction==0:
                    st.success("You don't have diabetes")
                elif prediction==1:
                    st.write(" You have diabetes")
with tab2:

    # Input method selection
    meth = st.radio("Choose input method:", ["Load Dataset", "Fill Parameters"],key="input_method_radio")

    # ----------------- LOAD DATASET --------------------
    if meth == "Load Dataset":
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"] )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df=df.drop(columns=['Family Heart Disease','High Blood Pressure','Heart Disease Status'],axis=1)
            df=df.dropna()
            st.write("Raw uploaded data:")
            
            st.dataframe(df)

            # Encode all categorical columns with LabelEncoder
            cat_cols = df.select_dtypes(include='object').columns
            encoders = {}

            for col in cat_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                encoders[col] = encoder  # Store if needed later

            st.success("Categorical columns encoded successfully!")

            # Make predictions
            predictions = heart_disease.predict(df)
            df["Prediction"] = predictions

            # Show results
            st.subheader("Prediction Results")
            st.dataframe(df)

            # Optionally download the result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")

    # ----------------- FILL PARAMETERS --------------------
    elif meth == "Fill Parameters":
        
        st.subheader("Enter Your Health Parameters")
        with st.form("prediction_form"):
            Age = st.number_input('What is your Age', min_value=1, max_value=120)
            gender = st.selectbox("What's your gender", ['Male', 'Female'])
            BloodPressure = st.number_input('What is your Blood Pressure')
            Cholesterol_Level = st.number_input("What's your Cholesterol Level")
            Exercise_Habits = st.selectbox("What's your exercise habits level", ['High', 'Meduim', 'Low'])
            smoking = st.selectbox('Do you smoke?', ['Yes', 'No'])
            diabetes = st.selectbox('Are you suffering from diabetes?', ['Yes', 'No'])
            BMI = st.number_input("What's your BMI Level")
            Low_HDL_Cholesterol = st.selectbox("Do you have Low HDL Cholesterol level?", ['Yes', 'No'])
            High_LDL_Cholesterol = st.selectbox("Do you have High LDL Cholesterol level?", ['Yes', 'No'])
            Alcohol_Consumption = st.selectbox("What's your Alcohol Consumption Level?", ['High', 'Medium', 'Low'])
            stress_level = st.selectbox("What's your stress level?", ['High', 'Medium', 'Low'])
            sleep_hour = st.slider("What's the number of hours you sleep?", 1, 8, 1)
            sugar_Consumption = st.selectbox("What's your sugar Consumption Level?", ['High', 'Medium', 'Low'])
            Triglyceride_Level = st.number_input("What's your Triglyceride Level?")
            Fasting_Blood_Sugar = st.number_input("What's your Fasting Blood Sugar?")
            CRP_Level = st.number_input("What's your CRP Level?")
            Homocysteine_Level = st.number_input("What's your Homocysteine Level?")
            
            submit_button = st.form_submit_button(label='Prediction')

        if submit_button:
            # Prepare data as a single-row DataFrame
            data = pd.DataFrame({
                'Age': [Age],
                'Gender': [gender],
                'BloodPressure': [BloodPressure],
                'Cholesterol_Level': [Cholesterol_Level],
                'Exercise_Habits': [Exercise_Habits],
                'Smoking': [smoking],
                'Diabetes': [diabetes],
                'BMI': [BMI],
                'Low_HDL_Cholesterol': [Low_HDL_Cholesterol],
                'High_LDL_Cholesterol': [High_LDL_Cholesterol],
                'Alcohol_Consumption': [Alcohol_Consumption],
                'Stress_Level': [stress_level],
                'Sleep_Hour': [sleep_hour],
                'Sugar_Consumption': [sugar_Consumption],
                'Triglyceride_Level': [Triglyceride_Level],
                'Fasting_Blood_Sugar': [Fasting_Blood_Sugar],
                'CRP_Level': [CRP_Level],
                'Homocysteine_Level': [Homocysteine_Level]
            })

            # Encode categorical columns using LabelEncoder
            cat_columns = [
                'Gender', 'Exercise_Habits', 'Smoking', 'Diabetes', 'Low_HDL_Cholesterol',
                'High_LDL_Cholesterol', 'Alcohol_Consumption', 'Stress_Level', 'Sugar_Consumption'
            ]

            for col in cat_columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])

            # Convert DataFrame to numpy array
            input_data = data.values

            # Use your trained model to predict
            prediction = heart_disease.predict(input_data)
            if prediction[0] == 0:
                st.success("You don't have heart disease.")
            else:
                st.warning("You may have heart disease.")
