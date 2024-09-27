
#User Interface File - Streamlit
import streamlit as st
import pickle
import numpy as np
import time
from PIL import Image
import pandas as pd

#Load the saved model
#model = pickle.load(open(r'C:\Users\laksh\VS_Code\Machine_Learning\House_Prediction\multi_linear_regression_model.pkl', 'rb'))
House_data = pd.read_csv(r"D:\Data_Science&AI\ClassRoomMaterial\September\6th- Simple_Linear_regression\6th- slr\SLR - House price prediction\House_data.csv")

# Set the title of the Streamlit app
st.title("House Prediction Application")

image1 = Image.open('1bhk.jpg')
image2 = Image.open('2bhk.jpg')
image3 = Image.open('3bhk.jpg')
image4 = Image.open('5bhk.jpg')
image5 = Image.open('5bhk_floor_plan.jpg')
image6 = Image.open('4bhk.jpg')
image7=Image.open('HousePlan.jpg')
image8 = Image.open('3bhk_floor.jpg')
image9 = Image.open('2bhk_plan.jpg')


col1,col2,col3,col4,col5= st.columns(5)
# img1 = st.image("1bhk.jpg",caption="Beutiful 1bhk House", width=200,use_column_width ='never')
# img2 = st.image("2bhk.jpg", caption="Beutiful 2bhk House", width=200,use_column_width ='never')
# # # img3 = st.image("3bhk.jpg", caption="Beutiful 3bhk House", width=200,use_column_width ='never')
# # # img4 = st.image("5bhk.jpg", caption="Beutiful 5bhk House", width=200,use_column_width ='never')
# # img5 = st.image("Beach_house.jpg", caption="Beutiful Beach House", width=200,use_column_width ='never')

col1.image(image1, use_column_width=True, caption="1bhk House", clamp=255)
col2.image(image2, use_column_width=True, caption="2bhk House", clamp=255)
col2.image(image9, use_column_width=True, caption=" 2bhk House plan", clamp=255)
col3.image(image3, use_column_width=True, caption="3bhk House", clamp=255)
col3.image(image8, use_column_width=True, caption="3bhk House Plan", clamp=255)
col5.image(image3, use_column_width = True, caption="5bhk House", clamp=255)
col5.image(image5, use_column_width = True, caption=" 5bhk House plan", clamp=255)
col4.image(image6, use_column_width=True, caption="4bhk House", clamp=255)
col4.image(image7, use_column_width=True, caption="4bhk House Plan", clamp=255)


# Add a brief description
#des = st.write("This application predicts the Price of the House based on SquareFeet(Space) using a Multi Linear Regression model.")

st.write("<h6 style='text-align: left; color: black;'>This application predicts the price of the House based on SquareFeet(Space) using a Multi Linear Regression model.</h6>", unsafe_allow_html=True) # * is used to chnge font to bold
#st.markdown('<p style="color:blue;">Price</p>', unsafe_allow_html=True)
#st.markdown('<p style="color:blue;">SquareFeet</p>', unsafe_allow_html=True)

# Add input widget for user to enter years of experience
square_feet = st.number_input("**Enter the Squarefeet of the house:**", min_value=1000, max_value=5000, value=1001, step=25)
bedrooms = st.number_input("Enter number of bedrooms", min_value=1, max_value=10, value=1)
bathrooms = st.number_input("Enter number of bathrooms", min_value=1, max_value=5, value=1)
Floor = st.number_input("Floor", min_value=0, max_value=4, value=1)



# When the button is clicked, make predictions
if st.button("**Predict Price**"):
    # Make a prediction using the trained model
   input_data = np.array([[square_feet]])  # Convert the input to a 2D array for prediction 
   prediction = House_data.predict(input_data)
   st.success(f"Price  predicted  for  {square_feet}sqft  is :  ${prediction[0]:,.2f}",icon= "🏠")
    
st.write("<h5 style='text-align: left; color: black;'>The House Dataset </h5>", unsafe_allow_html=True) 
st.write(pd.DataFrame(House_data))

st.write("<h5 style='text-align: left; color: black;'>catterplot Graph (Sqft_living Vs Price)</h5>", unsafe_allow_html=True) 

chart_House_data =pd.DataFrame(House_data,columns=['sqft_living','price'])
st.scatter_chart(chart_House_data, x_label='House(sqft)', y_label='Price(millions)',width=400,height=500)

st.write("<h5 style='text-align: left; color: black;'>The model was trained using  Price and SquareFeet of the House Based on House Dataset.</h5>", unsafe_allow_html=True) 

#Go to terminal and execute streamlit run application.py

