import streamlit as st
import pandas as pd
import numpy as np
import joblib



df = pd.read_csv("zomato_ready_data.csv")
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')
Encoder_listed_in_city = joblib.load('models/listed_in_city_Encoder.h5')
Encoder_listed_in_type= joblib.load('models/listed_in_type_Encoder.h5')
Encoder_location = joblib.load('models/location_Encoder.h5')
inp_data = []
result = ''


import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"ipg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('res17.jpg')   





st.markdown("# Predict If Your New Restaurant Will Succeed Or Not ")
st.markdown('______________________________________')
col1, col2 , col3 , col4 = st.columns(4)

with col1:
    f1 = st.text_input('Enter your approximate price for two customers ', '')
    f2 = st.multiselect('Choose your cuisines', list(df.columns)[32:])
    

with col2:
    f3 = st.multiselect('Select your restaurant listed_in(type)', df['listed_in(type)'].unique(), max_selections=1)
    f4 = st.multiselect('Choose your rest_type', list(df.columns)[8:32])
with col3:
    f5 = st.multiselect('Select your restaurant listed_in(city)', df['listed_in(city)'].unique(), max_selections=1)
    f6 = 1 if st.checkbox("Will your restaurant support online ordering?", False) else 0
    
with col4:
    f7 = st.multiselect('Select your restaurant location',df['location'].unique(),max_selections=1)
    f8 = 1 if st.checkbox("Will your restaurant support table booking?", False) else 0



if st.button(' Predict '):
        inp_data.append(f6)
        inp_data.append(f8)
        inp_data.append(int(Encoder_location.transform([f7[0]])[0]))
        inp_data.append(int(f1))
        inp_data.append(int(Encoder_listed_in_type.transform([f3[0]])[0]))
        inp_data.append(int(Encoder_listed_in_city.transform([f5[0]])[0]))
        inp_data.append(len(f2))
        inp_data.extend([1 if x in f4 else 0 for x in list(df.columns)[8:32]])
        inp_data.extend([1 if x in f2 else 0 for x in list(df.columns)[32:]])
        result = np.array(inp_data).reshape(1, 134)
        result = scaler.transform([inp_data])
        result = model.predict(result)[0]


if result == "Yes":
        result = '<p style= "font-size:30px;">Your Restaurant Will Succeed </p>'
        st.markdown(result,unsafe_allow_html=True)
elif result == "No":
        result = '<p style= "font-size:30px;">Sorry It Will Fail </p>'
        st.markdown(result,unsafe_allow_html=True)        
