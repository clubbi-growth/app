# %%  Imports
# Imports
import time 
import plotly.express as px 
  
import plotly.graph_objects as go
from datetime import timedelta 
import streamlit as st
import altair as alt
import pandas as pd
from datetime import date

import matplotlib.pyplot as plt 
 
import plotly.express as py 
from PIL import Image
import redshift_connector
import mysql.connector as connection
import datetime
import numpy as np   
  
#%matplotlib inline
from statsmodels.tsa.seasonal import STL 

import matplotlib.pyplot as plt
import ruptures as rpt
 
import mpld3
from mpld3 import plugins
import streamlit.components.v1 as components
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from feature_engine.creation import CyclicalFeatures 
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures  
from feature_engine.timeseries.forecasting import (LagFeatures,WindowFeatures,)
from sklearn.linear_model import Lasso  
from math import sqrt 

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# %% Data Atual
# Data Atual 
print("Dataa Atual")
data_atual = datetime.date.today()
hora_atual = datetime.datetime.now()
 
data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")
 





#fuso_horario = datetime.timezone(datetime.timedelta(hours=-3))
#data_hora_brasil = datetime.datetime.now(fuso_horario)
 
# %% Imagem 
# Imagem

from PIL import Image
import requests
from io import BytesIO

url = 'https://github.com/clubbi-growth/app/blob/main/Clubbi.png?raw=true'

response = requests.get(url)
icon = Image.open(BytesIO(response.content))

  

data_imagem = data_atual.strftime('%d/%m/%Y')
hora_imagem = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_imagem}")
print(f"Hora: {hora_imagem}") 
 
 




# %% Streamlit 
# Streamlit
alt.themes.enable("dark")

st.set_page_config(
    page_title="Data Growth App",
    page_icon=icon,
    layout="wide",
#    initial_sidebar_state="expanded")
    initial_sidebar_state="collapsed")


# %%  My Sql 
# My Sql



print('My Sql') 

#@st.cache_resource(ttl = 1)  
#@st.cache_resource()  
@st.cache_resource( ttl = 600)  
def load_my_sql():
    mydb =  connection.connect(
        host="aurora-mysql-db.cluster-ro-cjcocankcwqi.us-east-1.rds.amazonaws.com",
        user="ops-excellence-ro",
        password="L5!jj@Jm#9J+9K"
    )     
    return mydb


data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")

# %%  Redshift
# Redshit


print('Redshit') 


#@st.cache_resource( ttl = 600)  
@st.cache_resource( ttl = 600)  
def load_redshift():
    conn = redshift_connector.connect(
        host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
        database='dev',
        user='pbi_user',
        password='4cL6z0E7wiBpAjNRlqKkFiLW'
    )

    cursor: redshift_connector.Cursor = conn.cursor()
    return cursor


data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}") 

# %% Query Ofertao
# Query Ofertao

print("Query Ofertão")
 
query_ofertao  = "select \
o.date_start as Data, \
p.unit_ean as Ean, \
cat.category, \
p.description as Description, \
s.type, \
case when infoprice_code in ('Ofertao-Warehouse-5-9-cx') then '5-9 Cxs' else '1-4 Cxs' end as tipo_ofertao,\
rs.region_id, \
CASE WHEN rs.region_id in (1,7,19,27,28,29,30,31,36,37,62,63)  THEN 'RJC' \
WHEN rs.region_id   in (22,24,25) THEN 'RJI' \
WHEN rs.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE rs.region_id  END as 'Região',\
o.price \
FROM clubbi.offer o \
left JOIN clubbi.product p on p.id = o.product_id \
left join clubbi.supplier s on s.id = o.supplier_id \
left join clubbi.region_supplier rs on rs.supplier_id = s.id \
left join clubbi.category cat on cat.id = p.category_id \
where \
o.is_ofertao = 1 \
and DATE(o.date_start) >= '2022-01-01' \
and DATE(o.date_start) < '2025-01-01' \
;"\

     
@st.cache_resource( ttl = 43200)  
def load_ofertao():
    mydb = load_my_sql()
    data = pd.read_sql(query_ofertao,mydb) 
    return data

  
df_ofertao_inicial = load_ofertao() 
df_ofertao_inicial['Data'] = pd.to_datetime(df_ofertao_inicial['Data'])
df_ofertao_inicial['Data'] = df_ofertao_inicial['Data'].dt.date
 
data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")
 



# %% Query Produtos 
# Query Produtos
print("Query Produtos - df_produtos")

query_produtos  = "select convert(prod.ean,char) as ean ,prod.description,prod.category_id, prod.unit_ean, prod.only_sell_package, cat.category as Categoria, cat.section  from clubbi.product prod left join clubbi.category cat on cat.id = prod.category_id ;"


@st.cache_resource( ttl = 43200) 
def load_produtos():
    mydb = load_my_sql()
    data = pd.read_sql(query_produtos,mydb) 
    return data

 

data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}") 


 
