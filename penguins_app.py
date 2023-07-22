import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

csv = 'pengui.csv'
df = pd.read_csv(csv)

df.head()

df = df.dropna()

df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2})

X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state= 42)

svc = SVC(kernel= 'linear')
svc.fit(X_train, y_train)
svc_score = svc.score(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

rf_clf = RandomForestClassifier(n_jobs= -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache_data()
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
	pred = model.predict([island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex])
	return pred

st.title("Penguin Prediction")
blSlider = st.slider("bill_length_mm", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
bdSlider = st.slider("bill_depth_mm", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
flSlider = st.slider("flipper_length_mm", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
bmSlider = st.slider("body_mass_g", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))

sex = st.sidebar.selectbox("sex", (0, 1))
island = st.sidebar.selectbox("Island", (0, 1, 2))

classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
if st.sidebar.button("Predict"):
  if classifier == 'Support Vector Machine':
    species_type = prediction(svc_model, island, blSlider, bdSlider, flSlider, bmSlider, sex)
    score = svc_model.score(X_train, y_train)

  elif classifier =='Logistic Regression':
    species_type = prediction(log_reg, island, blSlider, bdSlider, flSlider, bmSlider, sex)
    score = log_reg.score(X_train, y_train)

  else:
    species_type = prediction(rf_clf, island, blSlider, bdSlider, flSlider, bmSlider, sex)
    score = rf_clf.score(X_train, y_train)

  st.write("Species predicted:", species_type)
  st.write("Accuracy score of this model is:", score)