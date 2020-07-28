import streamlit as st
import pandas as pd
import numpy as np
#import pickle 
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
#from sklearn.preprocessing import LabelEncoder
#from sklearn.ensemble import RandomForestClassifier
#import joblib as jl
import os
#import gzip
import warnings
warnings.filterwarnings('ignore')

def user_input_features():
		pr1 = st.sidebar.text_input('Payment Rate (USD)')
		pr2 = st.sidebar.slider('', 30,120,40)
		e11 = st.sidebar.text_input('Extra Income #1 (USD)')
		e12 = st.sidebar.slider('', 0,30,10)		
		e21 = st.sidebar.text_input('Extra Income #2 (USD)')
		e22 = st.sidebar.slider('', 0,30,5)		
		e31 = st.sidebar.text_input('Extra Income #3 (USD)')
		e32 = st.sidebar.slider('', 0,30,0)			
		if pr1:
			pr = pr1
		else:			
			pr = pr2		
		if e11:
			e1 = e11
		else:			
			e1 = e12		
		if e21:
			e2 = e21
		else:			
			e2 = e22		
		if e31:
			e3 = e31
		else:			
			e3 = e32
		data = {'Payment Rate (USD)': pr,
						'Extra Income #1 (USD)': e1,
						'Extra Income #2 (USD)': e2,
						'Extra Income #3 (USD)': e3}
		features = pd.DataFrame(data, index=[0])
		return features
				
st.write("""
## Home Credit Default Risk App

This app predicts the **Home Credit Default Risk**
""")

st.sidebar.header('User Data')
st.subheader('User Input Data')

uploaded_file = st.sidebar.file_uploader("", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(input_df)
else:
    input_df = user_input_features()
    st.write(input_df)

#rf = pd.read_pickle('rf2.sav')
#rf = jl.load("rf.sav")
#with open('rf2.sav', 'rb') as pickle_file:
    #rf = pickle.load(pickle_file)
	
from keras.models import load_model
model=load_model('hcrda.h5')

if input_df.size <= 25:	
		#input_df = pd.read_csv("rf0.csv")
		input_df = pd.read_csv("feat00.csv")
		
#log_reg_pred2 = rf.predict_proba(input_df)

p=model.predict(t)

st.subheader('Prediction Probability')
#st.write(log_reg_pred2)
st.write(p)
