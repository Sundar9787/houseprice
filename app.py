import numpy as np
import pickle
import streamlit as st
import base64
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
loaded_model = pickle.load(open('prediction_model.pkl', 'rb'))

with open('scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

def price_prediction(input_data):
    scaled_input = sc.transform(np.array(input_data).reshape(1, -1))
    prediction = loaded_model.predict(scaled_input)
    return prediction[0]


def format_currency(value):
    formatted_value = "{:,.2f}".format(value)
    return f"₹ {formatted_value}"

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: 100% ;
        font-family: 'Arial', sans-serif; 
         
    }}
    .title{{
          font-family: 'Georgia', serif;  
            }}
    .button {{
            font-family: 'Courier New', monospace; 
            }}
            
    .input-container label {{
                font-size: 16px;  
            }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# Add background image
add_bg_from_local("bg_img.jpeg")

def main():

    # Title
    st.title('House Price Prediction Web App')
    
    # Getting input from user
    Area = st.text_input("Area in sq.ft")
    Bedrooms =st.number_input("Total no of Bedrooms",min_value=0,step=1)
    Bathrooms =st.number_input("Total no of Bathrooms",min_value=0,step=1)
    
    options = {"Yes": 1, "No": 0}
    Guestrooms = st.selectbox("Guest room:", options.keys())
    numeric_guestrooms = options[Guestrooms]
    
    Basement = st.selectbox("Basement:", options.keys())
    numeric_basement = options[Basement]
    
    Hotwaterheating = st.selectbox("Hot Water Available?", options.keys())
    numeric_hotwaterheating = options[Hotwaterheating]
    
    Airconditioning = st.selectbox("AC available?", options.keys())
    numeric_airconditioning = options[Airconditioning]
    
    Parking = st.number_input("No of Parking lot",min_value=0,step=1)
    
    option_1 = {"Furnished": 0, "Semi-Furnished": 1, "Un-Furnished": 2}
    Furnishingstatus = st.selectbox("Furnishing status:", option_1.keys())
    numeric_furnishingstatus = option_1[Furnishingstatus]
    
    # Code for Prediction
    prediction = ''
    
   # Creating a button for prediction
    if st.button("Predict"):
        input_data = [Area, Bedrooms, Bathrooms, numeric_guestrooms, numeric_basement, 
                      numeric_hotwaterheating, numeric_airconditioning, Parking, numeric_furnishingstatus]
        prediction = price_prediction(input_data)
        formatted_prediction = format_currency(prediction)
        st.markdown(
            f'<div style="background-color: #99ff99; padding: 10px; border-radius: 8px;">'
            f'The predicted house price is: {formatted_prediction}'
            '</div>',
            unsafe_allow_html=True
        )
if __name__ == "__main__":
    main()