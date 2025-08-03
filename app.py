import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Bike Price Predictor", page_icon="🏍️")

dic_img = {
    "Royal Enfield" : 'images/Royal Enfield.jpg',
    "Bajaj" : 'images/bajaj.jpg',
    "Hero" : 'images/Hero.jpg',
    "Honda" : 'images/Honda.png',
    "KTM" : 'images/KTM.jpg',
    "Other" : 'images/Other.jpg',
    "Suzuki" : 'images/Suzuki.jpg',
    "TVS" : 'images/TVS.png',
    "Yamaha" : 'images/yamaha.png'
}

st.title('Bike Price Predictor!')

# Load the model and encoder
with open('bike_predictor_rf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('search.pkl', 'rb') as sf:
    search_txt = pickle.load(sf)

brand = st.sidebar.selectbox("Brand Name", search_txt['brand'])
kms_driven = st.sidebar.number_input('kms driven', value=6100)
power = st.sidebar.number_input("Power Of Bike in CC", value=500)
age = st.sidebar.number_input('Age', value=2)
city = st.sidebar.selectbox('City', search_txt['city'])

if st.sidebar.button("Bike Price"):
    with st.container():
        img, txt = st.columns((1,2))

        with img:
            st.image(dic_img[brand], width=200)
        
        with txt:
            st.write(f"#### Bike Brand : {brand}")
            st.write(f"###### Driven Bike in kms : {kms_driven}")
            st.write(f"###### Power of bike in cc : {power}")
            st.write(f"###### Age of bike : {age}")
            st.write(f"###### city from you buy : {city}")
    
    # Prepare input for prediction
    categorical_input = np.array([[brand, city]])
    numerical_input = np.array([[kms_driven, power, age]])
    
    # Transform categorical features
    categorical_encoded = encoder.transform(categorical_input)
    
    # Combine features
    input_combined = np.hstack([categorical_encoded, numerical_input])
    
    # Make prediction
    y_pred = model.predict(input_combined)
    st.write(f"### I Think Bike Price Should Be ₹{round(y_pred[0], 2)}")
    
