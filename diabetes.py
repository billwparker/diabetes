import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

st.title('Diabetes Prediction')

st.divider()

cols1, cols2, cols3, cols4 = st.columns(4)

with cols1:
  pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)

  bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=30.0, step=0.1)

with cols2:

  glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100, step=1)

  blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=100, step=1)
  
with cols3:
  
  skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=25, step=1)
  
  insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=100, step=1)
  
with cols4:
  
  age = st.number_input("Age", min_value=0, max_value=100, value=25, step=1)
  
  
with cols1:
  model_type = st.selectbox(
        "Which classification model would you like to use?",
        ("Random Forest", "Gradient Boost", "Neural Network (PyTorch)", "Neural Network (TensorFlow)"),
    )  
  
# Create a DataFrame for the input data  
ex = pd.DataFrame([[
  pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, age
]])
ex.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

if model_type == "Random Forest":
  model = pickle.load(open('rf.sav', 'rb'))
elif model_type == "Gradient Boost":
  model = pickle.load(open('gb.sav', 'rb'))
elif model_type == "Neural Network (PyTorch)":
  import torch
  from model_pytorch import DiabetesPTModel
  
  model = DiabetesPTModel()
  model.load_state_dict(torch.load('diabetes_model_pt.pth'))
  model.eval()
  
elif model_type == "Neural Network (TensorFlow)":
  import tensorflow as tf
  
  model = tf.keras.models.load_model('diabetes_model_tf.h5')  
  
# Load the scaler - only scaling is required for Neural Network models
if model_type == "Neural Network (PyTorch)" or model_type == "Neural Network (TensorFlow)":
  with open('scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

  # Standardize the input data
  input_data = scaler.transform(ex)


# Make prediction
st.write("")
if st.button('Predict Diabetes'):
  
    if model_type == "Random Forest" or model_type == "Gradient Boost":
      prediction = model.predict(ex)[0]
    else:
              
      if model_type == "Neural Network (PyTorch)":

        input_data = torch.tensor(input_data, dtype=torch.float32)
          
        with torch.no_grad():
            prediction = model(input_data)
            prediction = prediction.round().item()
      else:
        prediction = model.predict(input_data)
        prediction = (prediction > 0.5).astype(int).item()
        
    if prediction == 1:
        st.write('Diabetes Prediction: Positive')
    else:
        st.write('Diabetes Prediction: Negative')

