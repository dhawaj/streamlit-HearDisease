import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import streamlit as st


# Loading the model
import pickle
filename = './data/farmingham-model.pkl'
model = pickle.load(open(filename,'rb'))

# loading vars
filename = './data/farmingham-dvars.pkl'
dVars = pickle.load(open(filename,'rb'))

#### Creating the interface

inp = []

st.title('10 Year Hear Disease Prediction')


col1,col2,col3 = st.columns(3)

### Male to int
male = col1.selectbox("Enter your gender",['Male','Female'])
if male == 'Male':
	male = 1
else:
	male = 0

inp.append(male)

age = col2.number_input("Enter your age",step = 1)
inp.append(age)


# ed to int
ed_list = ['School','High School','Under Graduate Degree','Post Graduate Degree']
education = col3.selectbox("Highest Academic Qualification",ed_list)

value = 1
ed_dict = {}
for i in ed_list:
	ed_dict[i] = value
	value = value + 1

education = ed_dict[education]

inp.append(education)


smoker = col1.selectbox("Are you currently a smoker?",['Yes','No'])

cigsPerDay = col2.number_input('How many cigarettes a day?',step = 1)
inp.append(cigsPerDay)

#BP meds to int
BPmeds = col3.selectbox('Are you currently on BP medications?',['Yes','No'])
if BPmeds == 'Yes':
	BPmeds = 1
else:
	BPmeds = 0

inp.append(BPmeds)


stroke = col1.selectbox('Have you ever experinced a stroke?',['Yes','No'])


#PrevalentHyp to int
prevalentHyp = col2.selectbox('Do you have hypertension',['Yes','No'])

if prevalentHyp == 'Yes':
	prevalentHyp = 1
else:
	prevalentHyp = 0

inp.append(prevalentHyp)


# Diabetes to int

diabetes = col3.selectbox('Do you have diabetes?',['Yes','No'])
if diabetes == 'Yes':
	diabetes = 1
else:
	diabetes = 0



totChol = col1.number_input('Enter your cholesterol level')
inp.append(totChol)

sysBP = col2.number_input('Enter your systolic blood pressure')
inp.append(sysBP)

diaBP = col3.number_input('Enter your diatolic blood pressure')
inp.append(diaBP)

BMI = col1.number_input('Enter your BMI')
inp.append(BMI)

heartRate= col2.number_input('Enter your resting heart rate')
inp.append(heartRate)

glucose = col3.number_input('Enter your glucose level')
inp.append(glucose)

inp = np.array(inp).reshape(1,-1)

if st.button('Predict'):
	pred = model.predict(inp)
	if pred == 0:
		st.write('You likely will not develop heart disease in the 10 years.')
	else:
		st.write('You likely will develop heart disease in the 10 years.')






