import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
import random
import lime
from lime import lime_tabular
import warnings
import pickle
from pickle import *
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
#import seaborn as sn
from sklearn import preprocessing
import copy
import requests
import json
import seaborn as sns
from numerize.numerize import numerize

st.title("Comparaisons")
st.sidebar.image("pret_a_depenser.png")


f = open("x","rb")
x = load(f)
f.close()

f = open("y","rb")
y = load(f)
f.close()

f = open("z","rb")
z = load(f)
f.close()

f = open("r","rb")
r = load(f)
f.close()

f = open("t","rb")
t = load(f)
f.close()

f = open("u","rb")
u = load(f)
f.close()

f = open("v","rb")
v = load(f)
f.close()

f = open("s","rb")
s = load(f)
f.close()



f = open("w","rb")
numero_client = load(f)
f.close()

xtrain = x 
ytest_2 = y
xtest_4 = z
modele = r
xtest_final = t
max_seuil = u
train_set_proba = v
xtest_2 = s

data = xtest_final
data_2 = xtest_final




result = requests.get("http://127.0.0.1:5000/get_pret/")
result_2 = requests.get("http://127.0.0.1:5000/get_pret_data/")
dict_new = result_2.json()



data_2['predict'] = pd.Series(dict_new.values())



option = st.selectbox(
    'Variables à comparer',
    ('AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    ))

st.write('You selected:', option)

valeurs_option = data[option].value_counts()


df_2 = data_2[data_2['predict']== 0]
df_3 = data_2[data_2['predict']== 1]
df_4 = data[data['SK_ID_CURR']== numero_client]
fig, ax = plt.subplots()
    
ax = sns.kdeplot(data=df_2, x='AMT_INCOME_TOTAL', hue = 'TARGET')
ax  = sns.kdeplot(data=df_3, x='AMT_INCOME_TOTAL', hue = 'TARGET')    
ax = plt.axvline(x=float(df_4['AMT_INCOME_TOTAL']), ymin = 0, ymax = 1, linewidth=2, color='r')

st.pyplot(fig)

#
df_2 = data_2[data_2['predict']== 0]
df_3 = data_2[data_2['predict']== 1]
df_4 = data[data['SK_ID_CURR']== numero_client]
fig2, ax2 = plt.subplots()
     
ax2 = sns.kdeplot(data=df_2, x='AMT_CREDIT', hue = 'TARGET') 
ax2  = sns.kdeplot(data=df_3, x='AMT_CREDIT', hue = 'TARGET')
    
ax2 = plt.axvline(x=float(df_4['AMT_CREDIT']), ymin = 0, ymax = 1, linewidth=2, color='r')
    
st.pyplot(fig2)
    
option_2 = st.sidebar.selectbox(
    'Variables à comparer',
    (list(data_2['NAME_EDUCATION_TYPE'].unique())
     ))

option_3 = st.sidebar.selectbox(
    'Variables à comparer',
    (list(data_2['NAME_FAMILY_STATUS'].unique())
     ))

option_4 = st.sidebar.selectbox(
    'Variables à comparer',
    (list(data_2['NAME_HOUSING_TYPE'].unique())
     ))


df_education = data_2[data_2['NAME_EDUCATION_TYPE'] == option_2 ]
df_family = data_2[data_2['NAME_FAMILY_STATUS'] == option_3 ]
df_house = data_2[data_2['NAME_HOUSING_TYPE'] == option_4 ]


df_education_final = df_education['predict'].value_counts()
labels = df_education['predict'].unique()
fig4, ax4 = plt.subplots()
ax4.pie(df_education_final,  autopct='%.2f%%')
ax4.legend(labels)   
ax4.set_title("Predict Name education")   
st.pyplot(fig4)


df_family_final = df_family['predict'].value_counts()
fig6, ax = plt.subplots()
ax.pie(df_family_final,  autopct='%.2f%%') 
ax.legend(labels)   
ax.set_title("Predict Family status")   
st.pyplot(fig6)


df_house_final = df_house['predict'].value_counts()
fig7, ax = plt.subplots()
ax.pie(df_house_final,  autopct='%.2f%%') 
ax.legend(labels)   
ax.set_title("Predict House Type")   
st.pyplot(fig7)
     

    