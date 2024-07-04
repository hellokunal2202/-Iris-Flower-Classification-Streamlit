from os import write
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_data():
  iris = load_iris()
  df = pd.DataFrame(iris.data, columns=iris.feature_names)
  df['species'] = iris.target
  return df, iris.target_names


df, target_names = load_data()
st.title(':hibiscus: Iris Flower Classification')
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])
st.subheader("Input Feature")
sepal_length = st.number_input('Sepal Length', min_value=4.3, max_value=7.9)
sepal_width = st.number_input('Sepal Width', min_value=2.0, max_value=4.4)
petal_length = st.number_input('Petal Length', min_value=1.00, max_value=6.9)
petal_width = st.number_input('Petal Width', min_value=0.1, max_value=2.5)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

prediction = model.predict(input_data)
st.subheader(f'Prediction: {target_names[prediction[0]]}')

