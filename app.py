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


# from sqlalchemy import create_engine
 
# connection_string = f"mysql+mysqlconnector://username:password@host:port/database_name"
# engine = create_engine(connection_string)

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
 



def ofertao(df_ofertao_inicial, lista_categorias, regiao, size):

    #df_ofertao = ofertao(df_ofertao_inicial, ['Derivados de Leite'] ,  ['RJC'],['1-4 Cxs'])
    #df_ofertao
    df_ofertao = df_ofertao_inicial.copy() 
    if lista_categorias[0] != 'lista_categorias': df_ofertao = df_ofertao[df_ofertao['category'].isin(lista_categorias)]
     
    if regiao[0] != 'regiao': df_ofertao = df_ofertao[df_ofertao['Região'].isin(regiao)]
     
    if size[0] != 'size': df_ofertao = df_ofertao[df_ofertao['tipo_ofertao'].isin(size)]  
    df_ofertao['Ofertão'] = 1 
    
    df_ofertao_dia = df_ofertao.copy()  
    df_ofertao_prod = df_ofertao.copy()  



    df_ofertao = df_ofertao[['Data','category','price']].groupby(['Data','category']).min('price')
    df_ofertao = df_ofertao.reset_index(drop = False)
    df_ofertao = df_ofertao.set_index('Data') 

    df_ofertao = pd.get_dummies(df_ofertao['category']).astype(float)
    df_ofertao = df_ofertao.groupby('Data').max() 
    df_ofertao.columns=[ "Ofertão " + str(df_ofertao.columns[k-1])   for k in range(1, df_ofertao.shape[1] + 1)]
 


    if len(df_ofertao) == 0 : return df_ofertao
    df_ofertao =  pd.DataFrame(df_ofertao.asfreq('D').index).set_index('Data').merge(df_ofertao, left_index = True, right_index=True,how = "left") 
    df_ofertao = df_ofertao.replace(np.nan, 0)  
  

    return df_ofertao 


 
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

# %% Query Concorrência
# Query Concorrencia
print("Query  Concorrencia") 

query_conc =  '''select * from public.concorrencia_python '''


@st.cache_resource( ttl = 43200) 
def load_concorrencia():
    cursor = load_redshift()
    cursor.execute(query_conc)
    query_concorrencia: pd.DataFrame = cursor.fetch_dataframe() 
    query_concorrencia['data'] = pd.to_datetime(query_concorrencia['data'])
    query_concorrencia.sort_values('data',ascending=True )
    query_concorrencia['ean'] = query_concorrencia['ean'].astype(str) 
    
    return query_concorrencia

query_concorrencia = load_concorrencia()
 

# %% Query Users Previsao
# Query Users Previsão
print("Query  Users Previsão") 

 
query_users_prev =  '''select * from public.ops_customer '''


@st.cache_resource( ttl = 43200) 
def load_users_previsao():
    cursor = load_redshift()
    cursor.execute(query_users_prev)
    df_users: pd.DataFrame = cursor.fetch_dataframe()  

    df_users['Tipo_Cliente'] = np.where(
    
                                        (df_users['size']== 'counter') |  
                                        (df_users['size']== 'one_checkout') |  
                                        (df_users['size']== 'two_checkouts') | 
                                        (df_users['size']== 'three_to_four_checkouts'),
                                        '1-4 Cxs' , '5-9 Cxs' )



    df_users['1-4 Cxs'] = np.where((df_users['Tipo_Cliente'] == '1-4 Cxs') , 1 , 0 )
    df_users['5-9 Cxs'] = np.where((df_users['Tipo_Cliente'] == '1-4 Cxs') , 0 , 1 )
    df_users = df_users.rename(columns = {'region name':'Region Name'})  
    df_users['Não_Mercado'] =  np.where((df_users['tipo da loja'] == 'Mercado' ) ,   0, 1  )


    return df_users

 
df_users_previsao =  load_users_previsao()



# %% Query Orders 
# Query Orders 
 
print("Query Orders - df_orders")

query_order_d_1  = "select \
DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour,\
Date(ord.order_datetime) as Data,\
HOUR(ord.order_datetime) as Hora,\
CONVERT(ord.id, char) as order_id,\
CONVERT(ord_ite.id, char) as order_item_id,\
ord.customer_id, \
ord.region_id, \
CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN cli.region_id in (22,24,25) THEN 'RJI' \
WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
ord_ite.store_id, \
CONVERT(ord_ite.product_id, CHAR) as ean,\
CONVERT(ord_ite.unit_product_id, CHAR) as unit_ean,\
prod.description as Produto,\
ord_ite.category as Categoria,\
ord_ite.is_multi_package,\
ord_ite.product_package_qtd,\
ord_ite.price_managers,\
Convert(ord_ite.offer_id, char) as offer_id,  \
case when ord_ite.original_price > ord_ite.unit_price then 1 else 0 end as flag_desconto,\
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as Original_Price, \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end as Price, \
case when ord_ite.original_price > ord_ite.unit_price then \
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end \
else 0 end as desconto_unitario, \
case when ord_ite.original_price > ord_ite.unit_price then \
(case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end ) * \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end \
else 0 end as desconto_total, \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end as Quantity, \
ord_ite.quantity *  \
(CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END) / 1000.0 as 'Peso',\
ord_ite.total_price  as 'Gmv' \
from  clubbi_backend.order ord \
left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
left join clubbi.product prod ON ord_ite.product_id = prod.ean \
left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
left join clubbi.category_volume cat ON prod.category_id = cat.category_id  \
where    \
1 = 1 \
and DATE(ord.order_datetime) >= '2024-01-01' \
and DATE(ord.order_datetime) <  CURDATE()  \
;"\

 
query_order_d_0  = "select \
DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour,\
Date(ord.order_datetime) as Data,\
HOUR(ord.order_datetime) as Hora,\
CONVERT(ord.id, char) as order_id,\
CONVERT(ord_ite.id, char) as order_item_id,\
ord.customer_id, \
ord.region_id, \
CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
WHEN cli.region_id in (22,24,25) THEN 'RJI' \
WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
ord_ite.store_id, \
CONVERT(ord_ite.product_id, CHAR) as ean,\
CONVERT(ord_ite.unit_product_id, CHAR) as unit_ean,\
prod.description as Produto,\
ord_ite.category as Categoria,\
ord_ite.is_multi_package,\
ord_ite.product_package_qtd,\
ord_ite.price_managers,\
Convert(ord_ite.offer_id, char) as offer_id,  \
case when ord_ite.original_price > ord_ite.unit_price then 1 else 0 end as flag_desconto,\
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as Original_Price, \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end as Price, \
case when ord_ite.original_price > ord_ite.unit_price then \
case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end \
else 0 end as desconto_unitario, \
case when ord_ite.original_price > ord_ite.unit_price then \
(case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end ) * \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end \
else 0 end as desconto_total, \
case \
when ord_ite.product_package_qtd  is null then ord_ite.quantity \
when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
else ord_ite.product_package_qtd  * ord_ite.quantity end as Quantity, \
ord_ite.quantity *  \
(CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END) / 1000.0 as 'Peso',\
ord_ite.total_price  as 'Gmv' \
from  clubbi_backend.order ord \
left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
left join clubbi.product prod ON ord_ite.product_id = prod.ean \
left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
left join clubbi.category_volume cat ON prod.category_id = cat.category_id  \
where    \
1 = 1 \
and DATE(ord.order_datetime) >= CURDATE() \
and DATE(ord.order_datetime) <= '2025-04-01'  \
;"\



 

@st.cache_resource( ttl = 43200) 
def load_orders_d_1():
    mydb = load_my_sql()
    data = pd.read_sql(query_order_d_1,mydb) 
    return data


 

@st.cache_resource( ttl = 600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
def orders_final():
    mydb = load_my_sql()
    
    
    df_produtos = load_produtos() 

    df_produtos['ean'] = df_produtos['ean'].astype(np.int64).astype(str)
    df_produtos = df_produtos.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']] 

    df_orders_d_1 = load_orders_d_1()
    df_orders_d_0 = pd.read_sql(query_order_d_0,mydb) 
    df_orders = pd.concat([df_orders_d_1, df_orders_d_0 ] )
    df_orders['DateHour'] = pd.to_datetime(df_orders['DateHour'])   


    df_orders = df_orders.drop(columns = ['Categoria'])
    df_orders = df_orders.merge(df_produtos  ,how ='left', left_on='unit_ean', right_on='unit_ean_prod', suffixes=(False, False))
    
    df_orders['Produtos'] = df_orders['unit_ean_prod'].astype(str) + ' - ' +  df_orders['Unit_Description'].astype(str) 

    return df_orders
 
 
df_orders = orders_final() 

@st.cache_resource( ttl = 600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
def hora_orders_final():
    data_atual = datetime.date.today()
    hora_atual = datetime.datetime.now()  - timedelta(hours=3)   

    data_formatada = data_atual.strftime('%d/%m/%Y')
    hora_formatada = hora_atual.strftime('%H:%M:%S')
    print(f"Data: {data_formatada}")
    print(f"Hora: {hora_formatada}")  


    data_hora_completa = f"{data_formatada} {hora_formatada}"
    return data_hora_completa


hora_atualizacao = hora_orders_final() 
 
date_hour_orders = df_orders[df_orders['Data'] == df_orders['Data'].max()]['DateHour'].max() - timedelta(hours=1)    
max_hora_orders = df_orders[df_orders['Data'] == df_orders['Data'].max()]['Hora'].max() -1 
if  max_hora_orders<0: max_hora_orders= 23

# %% Df Order Previsão


# query_order_previsao  = "select \
# DATE_FORMAT(ord.order_datetime,'%Y-%m-%d %H:00:00') as DateHour,\
# Date(ord.order_datetime) as Data,\
# HOUR(ord.order_datetime) as Hora,\
# CONVERT(ord.id, char) as order_id,\
# CONVERT(ord_ite.id, char) as order_item_id,\
# ord.customer_id, \
# ord.region_id, \
# CASE WHEN cli.region_id in (1,7,19,27,28,29,30,31,36,37)  THEN 'RJC' \
# WHEN cli.region_id in (22,24,25) THEN 'RJI' \
# WHEN cli.region_id in (16,26,49,50,51,52,53) THEN 'BAC' ELSE ord.region_id END as 'Região',\
# ord_ite.store_id, \
# CONVERT(ord_ite.product_id, CHAR) as ean,\
# CONVERT(ord_ite.unit_product_id, CHAR) as unit_ean,\
# prod.description as Produto,\
# ord_ite.category as Categoria,\
# ord_ite.is_multi_package,\
# ord_ite.product_package_qtd,\
# ord_ite.price_managers,\
# Convert(ord_ite.offer_id, char) as offer_id,  \
# case when ord_ite.original_price > ord_ite.unit_price then 1 else 0 end as flag_desconto,\
# case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd  end as Original_Price, \
# case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end as Price, \
# case when ord_ite.original_price > ord_ite.unit_price then \
# case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
# case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end \
# else 0 end as desconto_unitario, \
# case when ord_ite.original_price > ord_ite.unit_price then \
# (case when  ord_ite.is_multi_package = 0 then  COALESCE(ord_ite.original_price,ord_ite.unit_price) else COALESCE(ord_ite.original_price,ord_ite.unit_price)/ ord_ite.product_package_qtd end - \
# case when  ord_ite.is_multi_package = 0 then  ord_ite.unit_price else  ord_ite.unit_price/ ord_ite.product_package_qtd  end ) * \
# case \
# when ord_ite.product_package_qtd  is null then ord_ite.quantity \
# when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
# else ord_ite.product_package_qtd  * ord_ite.quantity end \
# else 0 end as desconto_total, \
# case \
# when ord_ite.product_package_qtd  is null then ord_ite.quantity \
# when ord_ite.product_package_qtd  <=0 then ord_ite.quantity \
# else ord_ite.product_package_qtd  * ord_ite.quantity end as Quantity, \
# ord_ite.quantity *  \
# (CASE WHEN prod.gross_weight_in_gram IS NOT NULL THEN prod.gross_weight_in_gram  WHEN prod.net_volume_in_liters IS NOT NULL AND cat.gross_weight_per_content_volume_liter IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_volume_in_liters * cat.gross_weight_per_content_volume_liter  \
# WHEN prod.net_weight_in_gram IS NOT NULL AND cat.gross_weight_per_net_weight_gram IS NOT NULL THEN COALESCE(prod.number_of_items, 1) * prod.net_weight_in_gram * cat.gross_weight_per_net_weight_gram  \
# WHEN cat.category_id IS NOT NULL THEN COALESCE(cat.average_without_outliers, cat.average) * COALESCE(prod.number_of_items, 1)  \
# WHEN prod.net_weight_in_gram IS NOT NULL THEN prod.net_weight_in_gram  \
# WHEN prod.net_volume_in_liters IS NOT NULL THEN prod.net_volume_in_liters * 1000 ELSE 1000  END) / 1000.0 as 'Peso',\
# ord_ite.total_price  as 'Gmv' \
# from  clubbi_backend.order ord \
# left join clubbi_backend.order_item ord_ite on ord_ite.order_id = ord.id and (ord_ite.is_cancelled = 0 or ord_ite.is_cancelled IS NULL) \
# left join clubbi.product prod ON ord_ite.product_id = prod.ean \
# left join  clubbi.merchants  cli on cli.client_site_code = ord.customer_id \
# left join clubbi.category_volume cat ON prod.category_id = cat.category_id  \
# where    \
# 1 = 1 \
# and DATE(ord.order_datetime) >= '2022-01-01' \
# and DATE(ord.order_datetime) <  CURDATE()  \
# ;"\


# query_produtos_previsao  = "select convert(prod.ean,char) as ean ,prod.description,prod.category_id, prod.unit_ean, prod.only_sell_package, cat.category as Categoria, cat.section  from clubbi.product prod left join clubbi.category cat on cat.id = prod.category_id ;"
 
 
# @st.cache_resource( ttl = 45000) 
# def load_produtos_previsao():
#     mydb = load_my_sql()  
#     query_produtos  = pd.read_sql(query_produtos_previsao,mydb)  
#     query_produtos['ean'] = query_produtos['ean'].astype(np.int64).astype(str)
#     return query_produtos

# df_produtos_previsao = load_produtos_previsao()
 
# df_produtos_previsao = df_produtos_previsao.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']]  

# @st.cache_resource( ttl = 45000) 
# def load_orders_previsao():
#     mydb = load_my_sql() 
#     query_orders = pd.read_sql(query_order_previsao,mydb)  
        
#     df_produtos = load_produtos_previsao()
 
#     df_inicial = query_orders.copy()

#     df_inicial['Quantity'] = df_inicial['Quantity'].replace(np.nan,0)
#     df_inicial = df_inicial[df_inicial['Quantity'] > 0]

#     df_inicial['ean'] = df_inicial['ean'].astype(np.int64).astype(str) 
#     df_inicial['unit_ean'] = df_inicial['unit_ean'].astype(np.int64).astype(str) 
    

#     df_inicial = df_inicial.drop(columns = ['Categoria'])

    
#     df_produtos = df_produtos.copy()
#     df_produtos['ean'] = df_produtos['ean'].astype(np.int64).astype(str)
#     df_produtos = df_produtos.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']]  



#     df_inicial = df_inicial.merge(df_produtos  ,how ='left', left_on='unit_ean', right_on='unit_ean_prod', suffixes=(False, False))
#     df_inicial['Categoria'] =   np.where((df_inicial['Categoria'] == 'Óleos, Azeites e Vinagres') ,  'Óleos, Azeites E Vinagres'  , df_inicial['Categoria'] )
#     df_inicial['price_managers'] = df_inicial['price_managers'].replace(np.nan, 0 )
#     df_inicial['offer_id'] = df_inicial['offer_id'].replace(np.nan, 0 ).astype(np.int64).astype(float)
 
#     return df_inicial

# df_order_previsao = load_orders_previsao()

# %% Trafego Previsão 

 
# query_trafego_prev =  '''select * from public.trafego_site_hours '''
  
# @st.cache_resource( ttl = 45000) 
# def load_trafego_previsao():
#     cursor = load_redshift()
#     cursor.execute(query_trafego_prev)
#     query_trafego_previsao: pd.DataFrame = cursor.fetch_dataframe()  

      
#     df_trafego = query_trafego_previsao.copy()
#     df_trafego.columns=[ str(df_trafego.columns[k-1]).title()  for k in range(1, df_trafego.shape[1] + 1)]
#     df_trafego = df_trafego[['Datas', 'Datetimes', 'Hora', 'User_Id', 'Chave_Cliente_Dia', 'Chave_Final','Acessos' , 'Trafego', 'Search_Products', 'Add_To_Cart','Checkout']]
    
    

#     df_trafego = df_trafego.rename(columns = {'User_Id':'User','Datas':'Data', 'Search_Products':'Trafego_Search_Products', 'Add_To_Cart':'Trafego_Add_To_Cart', 'Checkout':'Trafego_Checkout' })
    
#     df_trafego = df_trafego.drop(columns = ['Datetimes','Hora','Chave_Final','Chave_Cliente_Dia'])
#     df_trafego['Data'] = pd.to_datetime(df_trafego['Data'])
#     df_trafego = df_trafego.groupby(['Data','User']).sum().reset_index(drop = False) 
    

#     df_trafego['key'] = df_trafego['Data'].astype(str) + df_trafego['User'].astype(str)
    
#     df_trafego['Trafego'] = np.where((df_trafego['Trafego'] > 0) ,  1  , 0 )
#     df_trafego['Trafego_Search_Products'] = np.where((df_trafego['Trafego_Search_Products'] > 0) ,  1  , 0 )
#     df_trafego['Trafego_Add_To_Cart'] = np.where((df_trafego['Trafego_Add_To_Cart'] > 0) ,  1  , 0 )
#     df_trafego['Trafego_Checkout'] = np.where((df_trafego['Trafego_Checkout'] > 0) ,  1  , 0 )
#     df_trafego = df_trafego.drop(columns = ['Acessos'])
     
#     return df_trafego
 
 
# df_trafego_previsao = load_trafego_previsao()


 
 
# %% Top Skus 



@st.cache_data()
def top_skus_long():

    df_top_skus = df_orders[df_orders['Região']=='RJC'].copy()  
    df_top_skus = df_orders[df_orders['Região']=='RJC'].copy()  
    df_top_skus['Data'] =  pd.to_datetime(df_top_skus['Data']) 
    df_top_skus = df_top_skus[df_top_skus['Data'] >=   pd.to_datetime('2024-01-01') ]
    
    df_top_skus = df_top_skus.iloc[int(df_top_skus.shape[0] *0.5):,:] 
    df_top_skus = df_top_skus[['Categoria','unit_ean','Unit_Description','Gmv']].groupby(['Categoria', 'unit_ean','Unit_Description']).sum().sort_values(by =['Categoria','Gmv'],ascending= False)
     
    df_top_skus = df_top_skus.reset_index(drop=False)
    df_gmv_categoria = df_top_skus[['Categoria','Gmv']].groupby('Categoria').sum().reset_index(drop = False) 
    df_gmv_categoria = df_gmv_categoria.sort_values('Gmv',ascending = False) 
    #df_gmv_categoria.head(10)['Categoria'].to_list()

    
    df_gmv_categoria = df_gmv_categoria.rename(columns = {'Gmv':'Gmv_Categoria'})
    
    

    df_top_skus  = df_top_skus.merge(df_gmv_categoria , how ='left', left_on='Categoria', right_on='Categoria', suffixes=(False, False))
    df_top_skus['Share_Produto'] = df_top_skus['Gmv']/df_top_skus['Gmv_Categoria']
    df_top_skus['Share_Acumulado'] = df_top_skus.groupby(['Categoria'])['Share_Produto'].cumsum()
    df_top_skus['Ranking'] = 1 
    df_top_skus['Ranking'] = df_top_skus.groupby(['Categoria'])['Ranking'].cumsum()
    df_top_skus = df_top_skus[df_top_skus['Ranking']<=5]
    df_top_skus = df_top_skus[df_top_skus['Share_Acumulado']<=0.9].sort_values('Gmv',ascending=False).reset_index(drop=True).reset_index(drop=False)
    
    top_skus = df_top_skus['unit_ean'].unique().tolist()
     
    top_skus_description = df_top_skus['Unit_Description'].unique().tolist()

    df_top_categorias = df_top_skus[['Categoria','Ranking']].groupby('Categoria').count()
    df_top_categorias = df_top_categorias.rename( columns = {'Ranking':'Top Skus'})
    df_top_categorias =  df_top_categorias.reset_index(drop = False)
    
    df_top_categorias['key_categoria'] = 1 
    df_top_categorias = df_top_categorias.set_index('key_categoria')


    df_top_categorias  = pd.pivot_table(df_top_categorias , values=['Top Skus'], index=['key_categoria'], columns=['Categoria'],aggfunc={ 'Top Skus': [ "median" ]})
    df_top_categorias.columns = df_top_categorias .columns.droplevel(0)
    df_top_categorias.columns = df_top_categorias .columns.droplevel(0)
    


    df_top_categorias.columns=[ "Top Produtos Total Categoria_" +   str(df_top_categorias.columns[k-1]) for k in range(1, df_top_categorias.shape[1] + 1)]


    df_top_categorias  = df_top_categorias.reset_index(drop = False).set_index('key_categoria')
    df_top_categorias  = pd.DataFrame(df_top_categorias.to_records()).set_index('key_categoria')
    df_top_categorias = df_top_categorias.reset_index(drop = False)

     

    return top_skus
 
 
top_skus_long_lista = top_skus_long()
 

# %% Top Skus D0 



@st.cache_resource( ttl = 600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
def top_skus_d0():
 
    data_atual = pd.to_datetime(datetime.date.today().strftime('%Y-%m-%d'))
    df_top_skus = df_orders[df_orders['Região']=='RJC'].copy()   
    df_top_skus['Data'] =  pd.to_datetime(df_top_skus['Data'])
    df_top_skus = df_top_skus[df_top_skus['Data'] ==   data_atual] 
    df_top_skus = df_top_skus[['Categoria','unit_ean','Unit_Description','Gmv']].groupby(['Categoria', 'unit_ean','Unit_Description']).sum().sort_values(by =['Categoria','Gmv'],ascending= False)
     
    df_top_skus = df_top_skus.reset_index(drop=False)
    df_gmv_categoria = df_top_skus[['Categoria','Gmv']].groupby('Categoria').sum().reset_index(drop = False) 
    df_gmv_categoria = df_gmv_categoria.sort_values('Gmv',ascending = False) 
    #df_gmv_categoria.head(10)['Categoria'].to_list()

    
    df_gmv_categoria = df_gmv_categoria.rename(columns = {'Gmv':'Gmv_Categoria'})
    
    

    df_top_skus  = df_top_skus.merge(df_gmv_categoria , how ='left', left_on='Categoria', right_on='Categoria', suffixes=(False, False))
    df_top_skus['Share_Produto'] = df_top_skus['Gmv']/df_top_skus['Gmv_Categoria']
    df_top_skus['Share_Acumulado'] = df_top_skus.groupby(['Categoria'])['Share_Produto'].cumsum()
    df_top_skus['Ranking'] = 1 
    df_top_skus['Ranking'] = df_top_skus.groupby(['Categoria'])['Ranking'].cumsum()
#    df_top_skus = df_top_skus[df_top_skus['Ranking']<=5]
    df_top_skus = df_top_skus[df_top_skus['Share_Acumulado']<=0.90].sort_values('Gmv',ascending=False).reset_index(drop=True).reset_index(drop=False)
    
    top_skus = df_top_skus['unit_ean'].unique().tolist()

    return top_skus
#df_skus = df_orders[df_orders['Categoria'] == categoria ][df_orders['unit_ean_prod'].isin(top_skus)]


top_skus_atual = top_skus_d0()  
  
def combine_and_keep_unique_order(list1, list2): 

  # Use list comprehension to filter new items and maintain order
  return list1 + [item for item in list2 if item not in list1]
 

# Create a new list with combined unique elements
top_skus = combine_and_keep_unique_order(top_skus_atual, top_skus_long_lista) 
  
 
#df_teste = df_orders[df_orders['Região']=='RJC'].copy()  
#df_teste['Data'] =  pd.to_datetime(df_teste['Data'])
#df_teste = df_teste[df_teste['Data'] ==   data_atual] 
#df_teste = df_teste[['Data','Categoria','unit_ean_prod','Gmv']][df_teste['unit_ean_prod'].isin(top_skus_d0)][df_teste['Categoria']=='Leite'].groupby(['Data','Categoria','unit_ean_prod']).sum().sort_values('Gmv',ascending=False)

#df_teste#.reset_index(drop=False)[['Data','Gmv']].groupby('Data').sum()




# %% Query User 
# Query User

print("Query User - df_users")
query_users =  '''select * from public.ops_customer '''



query_users  = "select *, \
case when region_id in (49,50,51,52,53) then 'BAC' \
when region_id in (1,7,19,27,28,29,30,31,36,37) then 'RJC' \
when region_id in (22,24,25) then 'RJI' else '-' \
end as 'region name', \
case when size in ('counter','one_checkout', 'two_checkouts','three_to_four_checkouts') then '1-4 Cxs' else '5-9 Cxs' end as size_final \
from  clubbi.merchants  \
where    \
1 = 1 \
;"\

 
@st.cache_resource( ttl = 43200) 
def load_users():
    mydb = load_my_sql()
    data = pd.read_sql(query_users,mydb) 
    return data



# @st.cache_data()
# def load_users():
#     cursor = load_redshift()
#     cursor.execute(query_users)
#     data: pd.DataFrame = cursor.fetch_dataframe() 
#     return data
  
df_users = load_users()  

data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")

 
#df_users['sizes'] = df_users['size_final']
#df_users['size_final'] = np.where((df_users['size_final']  == '5-6 caixas') , 0 ,  df_users['sizes'] )
 


# %% Trafego Geral
# Trafego Geral

print("Trafego Geral - df_trafego")
query_trafego =  '''select date_trunc('day', event_date::date )::date as datas, date_trunc('hour',(TIMESTAMP 'epoch' + event_timestamp / 1000000 * INTERVAL '1 second')::timestamptz  - interval '3 hour') as datetimes, extract(hour from datetimes) as hora, event_user_id as user_id, count(case when event_name = 'login' then event_user_id else null end) as acessos, count(distinct case when event_name = 'login' then event_user_id else null end) as trafego, count(distinct case when event_name = 'searchProducts' then event_user_id else null end) as search_products, count(distinct case when event_name = 'addToCart' then event_user_id else null end) as add_to_cart, count(distinct case when event_name = 'checkout' then event_user_id else null end) as checkout from ga4.events_data WHERE event_date >= (DATEADD(day,-100,CURRENT_DATE)) group by 1,2,3,4'''
query_trafegod0 =  '''select date_trunc('day', event_date::date )::date as datas, date_trunc('hour',(TIMESTAMP 'epoch' + event_timestamp / 1000000 * INTERVAL '1 second')::timestamptz  - interval '3 hour') as datetimes, extract(hour from datetimes) as hora, event_user_id as user_id, count(case when event_name = 'login' then event_user_id else null end) as acessos, count(distinct case when event_name = 'login' then event_user_id else null end) as trafego, count(distinct case when event_name = 'searchProducts' then event_user_id else null end) as search_products, count(distinct case when event_name = 'addToCart' then event_user_id else null end) as add_to_cart, count(distinct case when event_name = 'checkout' then event_user_id else null end) as checkout from ga4.events_data_intraday group by 1,2,3,4'''


@st.cache_resource( ttl = 43200) 
def load_trafego_d_1():
    cursor = load_redshift()
    cursor.execute(query_trafego)
    data: pd.DataFrame = cursor.fetch_dataframe() 
    return data

 
@st.cache_resource( ttl = 600) 
def load_trafego():
    cursor = load_redshift()
    df_d_1 = load_trafego_d_1()
   # cursor.execute(query_trafegod0)
   # df_d0: pd.DataFrame = cursor.fetch_dataframe()  
    data = df_d_1.copy() 
   
   #data = pd.concat([data, df_d0 ] ) 
    data = data.sort_values('datetimes')   
    data['datas'] = pd.to_datetime(data['datas'], format='%Y-%m-%d') 
    data['DateHour'] = data['datas'] + pd.to_timedelta(data['hora'], unit='H')
    data['DateHour']  = data['DateHour'].dt.tz_localize(None)
    
    # data['Trafego'] = np.where((data['Trafego'] > 0) ,  1  , 0 )
    # data['Trafego_Search_Products'] = np.where((data['Trafego_Search_Products'] > 0) ,  1  , 0 )
    # data['Trafego_Add_To_Cart'] = np.where((data['Trafego_Add_To_Cart'] > 0) ,  1  , 0 )
    # data['Trafego_Checkout'] = np.where((data['Trafego_Checkout'] > 0) ,  1  , 0 )
 
    return data

  
df_trafego = load_trafego()  
 

data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")
   


# %% Trafego Produtos D0
# Trafego Produtos 
print("Tradego Produtos")


query_trafego_produtos_d_1 =  '''select  date_trunc('day', event_date::date )::date as datas, date_trunc('hour',(TIMESTAMP 'epoch' + event_timestamp / 1000000 * INTERVAL '1 second')::timestamptz  - interval '3 hour') as datetimes, extract(hour from datetimes) as hora, event_user_id as user_id,  event_ean, count(distinct case when event_name = 'searchProducts' then event_user_id else null end) as search_products, count(distinct case when event_name = 'addToCart' then event_user_id else null end) as add_to_cart, count(distinct case when event_name = 'checkout' then event_user_id else null end) as checkout FROM ga4.events_data  where event_date >= (DATEADD(day,-100,CURRENT_DATE)) and event_ean in ('7891910000197','7891107101621','7896894900013','7896036090244','7896109801005','7896079500151','7896183202187','7898080640611','7898080640611','7896079500151','7891149104109','7898215151708','7896256600223','7896223709423','7896183202187','7893500020110','7898215151708','7896401100097','7896089011982','7896089012019','7896332007380','7898080640413','7891149011001','7896332007359','7898915949193','7896024761651','78936683','7896401100332','7896024760357','7898215151784','7622210565563','7891991009164','7898080640222','7891149440801','7894321722016','7891149102808','7891021006125','7891000100103','7891152802054','7896046900236','7894900027020','7894900027013','78912939','7896081800010','7896024722324','7898215157403') group by 1,2,3,4,5  '''
query_trafego_produtos_d0 =  '''select  date_trunc('day', event_date::date )::date as datas, date_trunc('hour',(TIMESTAMP 'epoch' + event_timestamp / 1000000 * INTERVAL '1 second')::timestamptz  - interval '3 hour') as datetimes, extract(hour from datetimes) as hora, event_user_id as user_id,  event_ean, count(distinct case when event_name = 'searchProducts' then event_user_id else null end) as search_products, count(distinct case when event_name = 'addToCart' then event_user_id else null end) as add_to_cart, count(distinct case when event_name = 'checkout' then event_user_id else null end) as checkout FROM ga4.events_data_intraday where event_ean in ('7891910000197','7891107101621','7896894900013','7896036090244','7896109801005','7896079500151','7896183202187','7898080640611','7898080640611','7896079500151','7891149104109','7898215151708','7896256600223','7896223709423','7896183202187','7893500020110','7898215151708','7896401100097','7896089011982','7896089012019','7896332007380','7898080640413','7891149011001','7896332007359','7898915949193','7896024761651','78936683','7896401100332','7896024760357','7898215151784','7622210565563','7891991009164','7898080640222','7891149440801','7894321722016','7891149102808','7891021006125','7891000100103','7891152802054','7896046900236','7894900027020','7894900027013','78912939','7896081800010','7896024722324','7898215157403') group by 1,2,3,4,5  '''


@st.cache_resource( ttl = 43200) 
def load_trafego_produtos_d_1():
    cursor = load_redshift()
    cursor.execute(query_trafego_produtos_d_1)
    data: pd.DataFrame = cursor.fetch_dataframe() 
    return data

 
@st.cache_resource( ttl = 600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
def load_trafego_produtos(): 
    cursor = load_redshift()
    df_d_1 = load_trafego_produtos_d_1()
    #cursor.execute(query_trafego_produtos_d0)
    #df_d0: pd.DataFrame = cursor.fetch_dataframe()  
    
    data = df_d_1.copy()
    #data = pd.concat([data, df_d0 ] ) 

    data = data.sort_values('datetimes')   
    data['datas'] = pd.to_datetime(data['datas'], format='%Y-%m-%d') 
    data['DateHour'] = data['datas'] + pd.to_timedelta(data['hora'], unit='H')
    data['DateHour']  = data['DateHour'].dt.tz_localize(None)

    data['Minuto'] = data.datetimes.dt.minute

    
    df_produtos = load_produtos() 

    df_produtos['ean'] = df_produtos['ean'].astype(np.int64).astype(str)
    df_produtos = df_produtos.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']] 

    
    data = data.merge(df_produtos  ,how ='left', left_on='event_ean', right_on='unit_ean_prod', suffixes=(False, False))

    
    data['Produtos'] = data['unit_ean_prod'].astype(str) + ' - ' +  data['Unit_Description'].astype(str) 


    return data

 

df_trafego_produtos = load_trafego_produtos()
 

@st.cache_resource( ttl = 600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
def hora_trafego_final():
    data_atual = datetime.date.today()
    hora_atual = datetime.datetime.now() 

    data_formatada = data_atual.strftime('%d/%m/%Y')
    hora_formatada = hora_atual.strftime('%H:%M:%S')
    print(f"Data: {data_formatada}")
    print(f"Hora: {hora_formatada}")  


    data_hora_completa = f"{data_formatada} {hora_formatada}"
    return data_hora_completa


hora_atualizacao_trafego = hora_trafego_final() 
  



 
data_atual = datetime.date.today()
hora_atual = datetime.datetime.now() 

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")

max_hora_trafego = df_trafego_produtos[df_trafego_produtos['datas'] == df_trafego_produtos['datas'].max()]['hora'].max() -1 

 

 
 
# %% Df Datetime 
# Df Datetime
print("Df Datetime")



@st.cache_resource( ttl = 600)  # ttl = 60 segundos   
def load_datetime(): 
    df_datetime = pd.concat([df_trafego[['DateHour']], df_orders[['DateHour']]]).sort_values('DateHour')
    df_datetime = df_datetime.groupby('DateHour').count() 
    df_datetime = df_datetime.reset_index(drop = False)
    df_datetime = df_datetime.set_index('DateHour')
    df_datetime =  pd.DataFrame(df_datetime.asfreq('H').index).set_index('DateHour').merge(df_datetime, left_index = True, right_index=True,how = "left") 
    df_datetime = df_datetime.reset_index(drop = False)
    return df_datetime

df_datetime = load_datetime()
 

# %% Funções Df View

def cria_df_view(df_datetime,df_users, df_trafego,df_orders,min_date,max_date,weekday_list,hora_list, region_id_list,regiao_list,size_list,categoria_list,ean_lista):
     
    df_datetime = df_datetime[df_datetime['DateHour']>=  min_date ]
    df_datetime = df_datetime[df_datetime['DateHour']<   max_date + pd.offsets.Day(1) ]

    df_users = df_users[['client_site_code','region_id','region name','size_final']]
    df_orders = df_orders.drop(columns = ['region_id','Região'])
    df_orders =  df_orders.merge( df_users ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False))
    df_trafego =  df_trafego.merge( df_users ,how ='left', left_on='user_id', right_on='client_site_code', suffixes=(False, False))
  
 
    if region_id_list[0] != 'region_id': df_orders = df_orders[df_orders['region_id'].isin(region_id_list)] 
    if regiao_list[0] != 'Região': df_orders = df_orders[df_orders['region name'].isin(regiao_list)] 
    if size_list[0] != 'size': df_orders = df_orders[df_orders['size_final'].isin(size_list)] 
    if categoria_list[0] != 'categoria': df_orders = df_orders[df_orders['Categoria'].isin(categoria_list)] 
    if ean_lista[0] != 'ean': df_orders = df_orders[df_orders['ean'].isin(ean_lista)] 


    if region_id_list[0] != 'region_id': df_trafego = df_trafego[df_trafego['region_id'].isin(region_id_list)] 
    if regiao_list[0] != 'Região': df_trafego = df_trafego[df_trafego['region name'].isin(regiao_list)] 
    if size_list[0] != 'size': df_trafego = df_trafego[df_trafego['size_final'].isin(size_list)]  

   
    df_order_date_hour = df_orders[['DateHour','Gmv','Peso','order_id']].groupby('DateHour').agg({'Gmv':'sum', 'Peso':'sum' , 'order_id': pd.Series.nunique   })    
    df_trafego_date_hour = df_trafego.drop(columns = ['datas','datetimes','hora','user_id']).groupby('DateHour').sum()
    #df_vendas =  df_datetime.merge( df_trafego_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))

    df_vendas =  df_datetime.merge( df_trafego_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_vendas =  df_vendas.merge( df_order_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_vendas = df_vendas.rename(columns = {'order_id':'Orders'})
    df_vendas['Hora'] = df_vendas['DateHour'].dt.hour
    df_vendas['Data'] = df_vendas['DateHour'].dt.date
    
    

    df_vendas = df_vendas.replace(np.nan , 0) 
    df_vendas = df_vendas[['DateHour','Hora','Data','trafego','search_products','add_to_cart','checkout','Gmv','Peso','Orders']]  
   # df_vendas = df_vendas[['DateHour','Hora','Data','trafego','search_products','add_to_cart','checkout','Gmv','Peso','Orders']]  
    df_vendas = df_vendas.set_index('DateHour') 

    cols_df = df_vendas.iloc[:,1:].columns.to_list() 
    df_lags = df_vendas[cols_df]
    
    
    for i in range(1, len(df_lags.columns) ):  df_lags[cols_df[i] + ' Acum'] = df_lags.groupby(['Data'])[cols_df[i]].cumsum()

    df_lags['% Conversão'] = df_lags['Orders']/ df_lags['trafego'] 
    df_lags['% Conversão Acum'] = df_lags['Orders Acum']/ df_lags['trafego Acum'] 
    df_gmv_dia = df_lags[['Data','Gmv','Peso']].groupby(['Data']).sum() 
    df_gmv_dia = df_gmv_dia.rename(columns={'Gmv':'Gmv Dia', 'Peso':'Peso Dia'})
    df_gmv_dia = df_gmv_dia.reset_index(drop = False)
    df_lags = df_lags.reset_index(drop = False)
    df_lags =  df_lags.merge( df_gmv_dia,how ='left', left_on='Data', right_on='Data', suffixes=(False, False))
    df_lags =  df_lags.set_index('DateHour')
    df_lags['% Share Gmv'] = df_lags['Gmv Acum']/ df_lags['Gmv Dia']
    df_lags['% Share Peso'] = df_lags['Peso Acum']/ df_lags['Peso Dia']
    
    for i in range(1, len(df_lags.columns) ): 

        coluna = df_lags.columns[i]
        
        lag_list = [7,14,21,28]
    
        
        for lag in lag_list:

            lag_final = 24* lag
            lag_name = coluna +  ' Lag ' + str(lag) 
            df_lags[lag_name] = df_lags[coluna].shift(periods= lag_final, freq="H") 
            df_lags = df_lags.replace(np.nan, 0)


        lag_media = coluna  + ' Lag Mean 7/14'      
        df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] )/2    
  
        lag_media = coluna  + ' Lag Mean 7/14/21'      
        df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] +  df_lags[coluna + ' Lag 21'] )/3

    

        lag_media = coluna  + ' Lag Mean 7/14/21/28'      
        df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] + +  df_lags[coluna + ' Lag 21'] +  df_lags[coluna + ' Lag 28'] )/4
    

        lag_var = '% Var ' +  coluna  + ' Lag Mean 7/14/21/28'     
        lag_var = '% Var ' +  coluna         
        df_lags[lag_var] = (df_lags[coluna]/ df_lags[lag_media] -1)
        df_lags[lag_var] = df_lags[lag_var].apply(lambda x: f"{x * 100:.2f}%")
     
        
    df_lags['Forecast Gmv'] =  df_lags['Gmv Acum'] /  df_lags['% Share Gmv Lag Mean 7/14/21/28'] 
    df_lags['Forecast Peso'] =  df_lags['Peso Acum'] /  df_lags['% Share Peso Lag Mean 7/14/21/28'] 
        

    df_lags['% Conversão'] = df_lags['% Conversão'].apply(lambda x: f"{x * 100:.2f}%")
    df_lags['% Conversão Acum'] = df_lags['% Conversão Acum'].apply(lambda x: f"{x * 100:.2f}%")
    df_vendas =  df_vendas[['Hora']].merge(  df_lags.drop(columns = 'Data'), how='left', left_index=True, right_index=True)   
 

    df_vendas['Ano'] = df_vendas.index.year 
    df_vendas['Mês'] = df_vendas.index.month
    df_vendas['Date'] = df_vendas.index.date
    df_vendas['Semana'] = df_vendas.index.isocalendar().week
    df_vendas['Weekday'] = df_vendas.index.weekday
    
    

    if weekday_list[0] != 'Weekday': df_vendas = df_vendas[df_vendas['Weekday'].isin(weekday_list)] 
    if hora_list[0] != 'Hora': df_vendas = df_vendas[df_vendas['Hora'].isin(hora_list)] 
    
    return df_vendas 



# %% Função Categoria
 
def cria_df_view_categoria(df_datetime,df_users, df_trafego,df_orders,min_date,max_date,weekday_list,hora_list, region_id_list,regiao_list,size_list,categoria_list,ean_lista):
     
    df_datetime = df_datetime[df_datetime['DateHour']>=  min_date ]
    df_datetime = df_datetime[df_datetime['DateHour']<   max_date + pd.offsets.Day(1) ]

    df_users = df_users[['client_site_code','region_id','region name','size_final']]
    df_orders = df_orders.drop(columns = ['region_id','Região']) 
    df_orders['Clientes'] = df_orders['Data'].astype(str) + df_orders['customer_id'].astype(str)
    df_orders =  df_orders.merge( df_users ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False))
    df_trafego =  df_trafego.merge( df_users ,how ='left', left_on='user_id', right_on='client_site_code', suffixes=(False, False))
 

 
    if region_id_list[0] != 'region_id': df_orders = df_orders[df_orders['region_id'].isin(region_id_list)] 
    if regiao_list[0] != 'Região': df_orders = df_orders[df_orders['region name'].isin(regiao_list)] 
    if size_list[0] != 'size': df_orders = df_orders[df_orders['size_final'].isin(size_list)] 


    if region_id_list[0] != 'region_id': df_trafego = df_trafego[df_trafego['region_id'].isin(region_id_list)] 
    if regiao_list[0] != 'Região': df_trafego = df_trafego[df_trafego['region name'].isin(regiao_list)] 
    if size_list[0] != 'size': df_trafego = df_trafego[df_trafego['size_final'].isin(size_list)]  
 
    df_clientes = df_orders.drop(columns = ['Hora']).copy()  
    df_clientes =  df_datetime.merge( df_clientes ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_clientes['Hora'] = df_clientes['DateHour'].dt.hour
    df_clientes['Data'] = df_clientes['DateHour'].dt.date
    df_clientes['Data'] = pd.to_datetime(df_clientes['Data']) 
    df_clientes['Clientes'] = df_clientes['Data'].astype(str) + df_clientes['customer_id'].astype(str)
    df_clientes = df_clientes[['Data','DateHour','Clientes','Hora']].groupby(['Data','DateHour','Clientes','Hora']).max().reset_index(drop = False)

     

    df_d0 = df_clientes.copy() 
  #  hour_lista = df_clientes[df_clientes['Data'] >=   pd.to_datetime(date.today())].Hora.unique().tolist() 
    hour_lista = df_clientes.Hora.unique().tolist() 
   # df_d0 = df_d0[df_d0['Data'] >=   pd.to_datetime(date.today())]


    # D-0 
    
    
      
    for i in range(0,len(hora_list)):
 
        df_hora = df_d0.copy() 
        df_hora = df_hora[['Data','DateHour','Clientes','Hora']][df_hora['Hora']<=hora_list[i]].groupby('Data').agg({ 'DateHour':'max','Hora': 'max', 'Clientes': pd.Series.nunique})
        df_hora = df_hora.reset_index(drop = False)   

        if i == 0: 
            df_positivacao_geral = df_hora.copy()
        else:
            df_positivacao_geral = pd.concat([df_positivacao_geral, df_hora])
     

    df_positivacao_geral = df_positivacao_geral.groupby('DateHour').max().reset_index(drop = False)
    df_positivacao_geral = df_positivacao_geral.rename( columns = {'Clientes':'Positivação Geral'})
    df_positivacao_geral = df_positivacao_geral[['DateHour','Positivação Geral']] 
 
    if categoria_list[0] != 'categoria': df_orders = df_orders[df_orders['Categoria'].isin(categoria_list)] 
    if ean_lista[0] != 'ean': df_orders = df_orders[df_orders['unit_ean_prod'] == ean_lista[0]] 

    if categoria_list[0] != 'categoria': df_trafego = df_trafego[df_trafego['Categoria'].isin(categoria_list)] 
  #  if ean_lista[0] != 'ean': df_trafego = df_trafego[df_trafego['unit_ean_prod'] == ean_lista[0]] 

    df_trafego = df_trafego[['DateHour', 'search_products','add_to_cart' ]]  

 
    df_clientes = df_orders.drop(columns = ['Hora']).copy()  
    df_clientes =  df_datetime.merge( df_clientes ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_clientes['Hora'] = df_clientes['DateHour'].dt.hour
    df_clientes['Data'] = df_clientes['DateHour'].dt.date
    df_clientes['Data'] = pd.to_datetime(df_clientes['Data']) 
    df_clientes['Clientes'] = df_clientes['Data'].astype(str) + df_clientes['customer_id'].astype(str)
    df_clientes = df_clientes[['Data','DateHour','Clientes','Hora']].groupby(['Data','DateHour','Clientes','Hora']).max().reset_index(drop = False)



    hour_lista = df_clientes[df_clientes['Data'] >=   pd.to_datetime(date.today())].Hora.unique().tolist() 
    
    df_hora = df_clientes.copy() 
    
    for i in range(0,len(hora_list)):


        df_hora = df_clientes.copy() 
        df_hora = df_hora[['Data','DateHour','Clientes','Hora']][df_hora['Hora']<=hora_list[i]].groupby('Data').agg({ 'DateHour':'max','Hora': 'max', 'Clientes': pd.Series.nunique})
        df_hora = df_hora.reset_index(drop = False)   

        if i == 0: 
            df_positivacao_categoria = df_hora.copy()
        else:
            df_positivacao_categoria = pd.concat([df_positivacao_categoria, df_hora])
     

    df_positivacao_categoria = df_positivacao_categoria.groupby('DateHour').max().reset_index(drop = False)
    df_positivacao_categoria = df_positivacao_categoria.rename( columns = {'Clientes':'Positivação Categoria'})
    df_positivacao_categoria = df_positivacao_categoria[['DateHour','Positivação Categoria']]  
 
 
    df_order_date_hour = df_orders[['DateHour','Gmv','Peso','order_id','Clientes']].groupby('DateHour').agg({'Gmv':'sum', 'Peso':'sum' , 'Clientes': pd.Series.nunique , 'order_id': pd.Series.nunique})    
    df_trafego_date_hour = df_trafego.groupby('DateHour').sum()
    df_vendas =  df_datetime.merge( df_trafego_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_vendas =  df_vendas.merge( df_order_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    

    
    df_vendas =  df_vendas.merge( df_positivacao_categoria ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_vendas =  df_vendas.merge( df_positivacao_geral ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
   


  
    df_vendas = df_vendas.rename(columns = {'order_id':'Orders'})
    df_vendas['Hora'] = df_vendas['DateHour'].dt.hour
    df_vendas['Data'] = df_vendas['DateHour'].dt.date

    df_vendas = df_vendas.replace(np.nan , 0) 


    df_vendas = df_vendas[['DateHour','Hora','Positivação Categoria','Positivação Geral','Data','search_products','add_to_cart','Gmv','Peso','Orders']]  
   # df_vendas = df_vendas[['DateHour','Hora','Data','trafego','search_products','add_to_cart','checkout','Gmv','Peso','Orders']]  
    df_vendas = df_vendas.set_index('DateHour') 



    cols_df = df_vendas.iloc[:,3:].columns.to_list() 
    df_lags = df_vendas[cols_df]
    
    
    #df_vendas =  df_vendas.merge( df_clientes ,how ='left', left_on='Data', right_on='Data', suffixes=(False, False))

 
    
    for i in range(1, len(df_lags.columns) ):  df_lags[cols_df[i] + ' Acum'] = df_lags.groupby(['Data'])[cols_df[i]].cumsum()

    df_lags['% Conversão Search'] = df_lags['Orders']/ df_lags['search_products'] 
    df_lags['% Conversão Search Acum'] = df_lags['Orders Acum']/ df_lags['search_products Acum'] 

    df_gmv_dia = df_lags[['Data','Gmv','Peso']].groupby(['Data']).sum() 
    df_gmv_dia = df_gmv_dia.rename(columns={'Gmv':'Gmv Dia', 'Peso':'Peso Dia'})
    df_gmv_dia = df_gmv_dia.reset_index(drop = False)
    df_lags = df_lags.reset_index(drop = False)
    df_lags =  df_lags.merge( df_gmv_dia,how ='left', left_on='Data', right_on='Data', suffixes=(False, False))
    df_lags =  df_lags.set_index('DateHour')
    df_lags['% Share Gmv'] = df_lags['Gmv Acum']/ df_lags['Gmv Dia']
    df_lags['% Share Peso'] = df_lags['Peso Acum']/ df_lags['Peso Dia']
    
    for i in range(1, len(df_lags.columns) ): 

        coluna = df_lags.columns[i]
        
        lag_list = [7,14,21,28]
    
        
        for lag in lag_list:

            lag_final = 24* lag
            lag_name = coluna +  ' Lag ' + str(lag) 
            df_lags[lag_name] = df_lags[coluna].shift(periods= lag_final, freq="H") 
            df_lags = df_lags.replace(np.nan, 0)


        lag_media = coluna  + ' Lag Mean 7/14'      
        df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] )/2    
  
        lag_media = coluna  + ' Lag Mean 7/14/21'      
        df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] +  df_lags[coluna + ' Lag 21'] )/3

    

        lag_media = coluna  + ' Lag Mean 7/14/21/28'      
        df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] + +  df_lags[coluna + ' Lag 21'] +  df_lags[coluna + ' Lag 28'] )/4
    

        lag_var = '% Var ' +  coluna  + ' Lag Mean 7/14/21/28'     
        lag_var = '% Var ' +  coluna         
        df_lags[lag_var] = (df_lags[coluna]/ df_lags[lag_media] -1)
        df_lags[lag_var] = df_lags[lag_var].apply(lambda x: f"{x * 100:.2f}%")
     
        



    df_lags['Forecast Gmv'] =  df_lags['Gmv Acum'] /  df_lags['% Share Gmv Lag Mean 7/14/21/28'] 
    df_lags['Forecast Peso'] =  df_lags['Peso Acum'] /  df_lags['% Share Peso Lag Mean 7/14/21/28'] 
        

    df_lags['% Conversão Search'] = df_lags['% Conversão Search'].apply(lambda x: f"{x * 100:.2f}%")
    df_lags['% Conversão Search Acum'] = df_lags['% Conversão Search Acum'].apply(lambda x: f"{x * 100:.2f}%")
    df_vendas =  df_vendas[['Hora','Positivação Categoria','Positivação Geral']].merge(  df_lags.drop(columns = 'Data'), how='left', left_index=True, right_index=True)   
 
  
    df_vendas['% Share Positivação Categoria'] = df_vendas['Positivação Categoria'] /  df_vendas['Positivação Geral']

    df_vendas['Ano'] = df_vendas.index.year 
    df_vendas['Mês'] = df_vendas.index.month
    df_vendas['Date'] = df_vendas.index.date
    df_vendas['Semana'] = df_vendas.index.isocalendar().week
    df_vendas['Weekday'] = df_vendas.index.weekday
    
    

    if weekday_list[0] != 'Weekday': df_vendas = df_vendas[df_vendas['Weekday'].isin(weekday_list)] 
    if hora_list[0] != 'Hora': df_vendas = df_vendas[df_vendas['Hora'].isin(hora_list)] 
    
    return df_vendas 
  
def cria_df_view_categoria2(df_datetime,df_users, df_trafego,df_orders,min_date,max_date,weekday_list,hora_list, region_id_list,regiao_list,size_list,categoria_list,ean_lista):
     
    df_datetime = df_datetime[df_datetime['DateHour']>=  min_date ]
    df_datetime = df_datetime[df_datetime['DateHour']<   max_date + pd.offsets.Day(1) ]

    df_users = df_users[['client_site_code','region_id','region name','size_final']]
    df_orders = df_orders.drop(columns = ['region_id','Região']) 
    df_orders['Clientes'] = df_orders['Data'].astype(str) + df_orders['customer_id'].astype(str)
    df_orders =  df_orders.merge( df_users ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False))
    df_trafego =  df_trafego.merge( df_users ,how ='left', left_on='user_id', right_on='client_site_code', suffixes=(False, False))
 

 
    if region_id_list[0] != 'region_id': df_orders = df_orders[df_orders['region_id'].isin(region_id_list)] 
    if regiao_list[0] != 'Região': df_orders = df_orders[df_orders['region name'].isin(regiao_list)] 
    if size_list[0] != 'size': df_orders = df_orders[df_orders['size_final'].isin(size_list)] 


    if region_id_list[0] != 'region_id': df_trafego = df_trafego[df_trafego['region_id'].isin(region_id_list)] 
    if regiao_list[0] != 'Região': df_trafego = df_trafego[df_trafego['region name'].isin(regiao_list)] 
    if size_list[0] != 'size': df_trafego = df_trafego[df_trafego['size_final'].isin(size_list)]  
 
    df_clientes = df_orders.drop(columns = ['Hora']).copy()  
    df_clientes =  df_datetime.merge( df_clientes ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_clientes['Hora'] = df_clientes['DateHour'].dt.hour
    df_clientes['Data'] = df_clientes['DateHour'].dt.date
    df_clientes['Data'] = pd.to_datetime(df_clientes['Data']) 
    df_clientes['Clientes'] = df_clientes['Data'].astype(str) + df_clientes['customer_id'].astype(str)
    df_clientes = df_clientes[['Data','DateHour','Clientes','Hora']].groupby(['Data','DateHour','Clientes','Hora']).max().reset_index(drop = False)

     

    df_d0 = df_clientes.copy() 
  #  hour_lista = df_clientes[df_clientes['Data'] >=   pd.to_datetime(date.today())].Hora.unique().tolist() 
    hour_lista = df_clientes.Hora.unique().tolist() 
   # df_d0 = df_d0[df_d0['Data'] >=   pd.to_datetime(date.today())]


    # D-0 
    
    
      
    for i in range(0,len(hora_list)):
 
        df_hora = df_d0.copy() 
        df_hora = df_hora[['Data','DateHour','Clientes','Hora']][df_hora['Hora']<=hora_list[i]].groupby('Data').agg({ 'DateHour':'max','Hora': 'max', 'Clientes': pd.Series.nunique})
        df_hora = df_hora.reset_index(drop = False)   

        if i == 0: 
            df_positivacao_geral = df_hora.copy()
        else:
            df_positivacao_geral = pd.concat([df_positivacao_geral, df_hora])
     

    df_positivacao_geral = df_positivacao_geral.groupby('DateHour').max().reset_index(drop = False)
    df_positivacao_geral = df_positivacao_geral.rename( columns = {'Clientes':'Positivação Geral'})
    df_positivacao_geral = df_positivacao_geral[['DateHour','Positivação Geral']] 
 
    if categoria_list[0] != 'categoria': df_orders = df_orders[df_orders['Categoria'].isin(categoria_list)] 
    if ean_lista[0] != 'ean': df_orders = df_orders[df_orders['Produtos'] == ean_lista[0]] 

    if categoria_list[0] != 'categoria': df_trafego = df_trafego[df_trafego['Categoria'].isin(categoria_list)] 
    if ean_lista[0] != 'ean': df_trafego = df_trafego[df_trafego['Produtos'] == ean_lista[0]] 

    df_trafego = df_trafego[['DateHour', 'search_products','add_to_cart' ]]  

 
    df_clientes = df_orders.drop(columns = ['Hora']).copy()  
    df_clientes =  df_datetime.merge( df_clientes ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_clientes['Hora'] = df_clientes['DateHour'].dt.hour
    df_clientes['Data'] = df_clientes['DateHour'].dt.date
    df_clientes['Data'] = pd.to_datetime(df_clientes['Data']) 
    df_clientes['Clientes'] = df_clientes['Data'].astype(str) + df_clientes['customer_id'].astype(str)
    df_clientes = df_clientes[['Data','DateHour','Clientes','Hora']].groupby(['Data','DateHour','Clientes','Hora']).max().reset_index(drop = False)



    hour_lista = df_clientes[df_clientes['Data'] >=   pd.to_datetime(date.today())].Hora.unique().tolist() 
    
    df_hora = df_clientes.copy() 
    
    for i in range(0,len(hora_list)):


        df_hora = df_clientes.copy() 
        df_hora = df_hora[['Data','DateHour','Clientes','Hora']][df_hora['Hora']<=hora_list[i]].groupby('Data').agg({ 'DateHour':'max','Hora': 'max', 'Clientes': pd.Series.nunique})
        df_hora = df_hora.reset_index(drop = False)   

        if i == 0: 
            df_positivacao_categoria = df_hora.copy()
        else:
            df_positivacao_categoria = pd.concat([df_positivacao_categoria, df_hora])
     

    df_positivacao_categoria = df_positivacao_categoria.groupby('DateHour').max().reset_index(drop = False)
    df_positivacao_categoria = df_positivacao_categoria.rename( columns = {'Clientes':'Positivação Categoria'})
    df_positivacao_categoria = df_positivacao_categoria[['DateHour','Positivação Categoria']]  
 
 
    df_order_date_hour = df_orders[['DateHour','Gmv','Peso','order_id','Clientes']].groupby('DateHour').agg({'Gmv':'sum', 'Peso':'sum' , 'Clientes': pd.Series.nunique , 'order_id': pd.Series.nunique})    
    df_trafego_date_hour = df_trafego.groupby('DateHour').sum()
    df_vendas =  df_datetime.merge( df_trafego_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_vendas =  df_vendas.merge( df_order_date_hour ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    

    
    df_vendas =  df_vendas.merge( df_positivacao_categoria ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
    df_vendas =  df_vendas.merge( df_positivacao_geral ,how ='left', left_on='DateHour', right_on='DateHour', suffixes=(False, False))
   


  
    df_vendas = df_vendas.rename(columns = {'order_id':'Orders'})
    df_vendas['Hora'] = df_vendas['DateHour'].dt.hour
    df_vendas['Data'] = df_vendas['DateHour'].dt.date

    df_vendas = df_vendas.replace(np.nan , 0) 


    df_vendas = df_vendas[['DateHour','Hora','Positivação Categoria','Positivação Geral','Data','search_products','add_to_cart','Gmv','Peso','Orders']]  
   # df_vendas = df_vendas[['DateHour','Hora','Data','trafego','search_products','add_to_cart','checkout','Gmv','Peso','Orders']]  
    df_vendas = df_vendas.set_index('DateHour') 



    cols_df = df_vendas.iloc[:,3:].columns.to_list() 
    df_lags = df_vendas[cols_df]
    
    
    #df_vendas =  df_vendas.merge( df_clientes ,how ='left', left_on='Data', right_on='Data', suffixes=(False, False))

 
    
    for i in range(1, len(df_lags.columns) ):  df_lags[cols_df[i] + ' Acum'] = df_lags.groupby(['Data'])[cols_df[i]].cumsum()

    df_lags['% Conversão Search'] = df_lags['Orders']/ df_lags['search_products'] 
    df_lags['% Conversão Search Acum'] = df_lags['Orders Acum']/ df_lags['search_products Acum'] 

    df_gmv_dia = df_lags[['Data','Gmv','Peso']].groupby(['Data']).sum() 
    df_gmv_dia = df_gmv_dia.rename(columns={'Gmv':'Gmv Dia', 'Peso':'Peso Dia'})
    df_gmv_dia = df_gmv_dia.reset_index(drop = False)
    df_lags = df_lags.reset_index(drop = False)
    df_lags =  df_lags.merge( df_gmv_dia,how ='left', left_on='Data', right_on='Data', suffixes=(False, False))
    df_lags =  df_lags.set_index('DateHour')
    df_lags['% Share Gmv'] = df_lags['Gmv Acum']/ df_lags['Gmv Dia']
    df_lags['% Share Peso'] = df_lags['Peso Acum']/ df_lags['Peso Dia']
    
    # for i in range(1, len(df_lags.columns) ): 

    #     coluna = df_lags.columns[i]
        
    #     lag_list = [7,14,21,28]
    
        
    #     for lag in lag_list:

    #         lag_final = 24* lag
    #         lag_name = coluna +  ' Lag ' + str(lag) 
    #         df_lags[lag_name] = df_lags[coluna].shift(periods= lag_final, freq="H") 
    #         df_lags = df_lags.replace(np.nan, 0)


    #     lag_media = coluna  + ' Lag Mean 7/14'      
    #     df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] )/2    
  
    #     lag_media = coluna  + ' Lag Mean 7/14/21'      
    #     df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] +  df_lags[coluna + ' Lag 21'] )/3

    

    #     lag_media = coluna  + ' Lag Mean 7/14/21/28'      
    #     df_lags[lag_media] = (df_lags[coluna + ' Lag 7']  +  df_lags[coluna + ' Lag 14'] + +  df_lags[coluna + ' Lag 21'] +  df_lags[coluna + ' Lag 28'] )/4
    

    #     lag_var = '% Var ' +  coluna  + ' Lag Mean 7/14/21/28'     
    #     lag_var = '% Var ' +  coluna         
    #     df_lags[lag_var] = (df_lags[coluna]/ df_lags[lag_media] -1)
    #     df_lags[lag_var] = df_lags[lag_var].apply(lambda x: f"{x * 100:.2f}%")
     
         

    # df_lags['Forecast Gmv'] =  df_lags['Gmv Acum'] /  df_lags['% Share Gmv Lag Mean 7/14/21/28'] 
    # df_lags['Forecast Peso'] =  df_lags['Peso Acum'] /  df_lags['% Share Peso Lag Mean 7/14/21/28'] 
        

    df_lags['% Conversão Search'] = df_lags['% Conversão Search'].apply(lambda x: f"{x * 100:.2f}%")
    df_lags['% Conversão Search Acum'] = df_lags['% Conversão Search Acum'].apply(lambda x: f"{x * 100:.2f}%")
    df_vendas =  df_vendas[['Hora','Positivação Categoria','Positivação Geral']].merge(  df_lags.drop(columns = 'Data'), how='left', left_index=True, right_index=True)   
 
  
    df_vendas['% Share Positivação Categoria'] = df_vendas['Positivação Categoria'] /  df_vendas['Positivação Geral']

    df_vendas['Ano'] = df_vendas.index.year 
    df_vendas['Mês'] = df_vendas.index.month
    df_vendas['Date'] = df_vendas.index.date
    df_vendas['Semana'] = df_vendas.index.isocalendar().week
    df_vendas['Weekday'] = df_vendas.index.weekday
    
    

    if weekday_list[0] != 'Weekday': df_vendas = df_vendas[df_vendas['Weekday'].isin(weekday_list)] 
    if hora_list[0] != 'Hora': df_vendas = df_vendas[df_vendas['Hora'].isin(hora_list)] 
    
    return df_vendas 
  


# top_skus_atual = df_orders[df_orders['Categoria'] == 'Leite' ][df_orders['unit_ean_prod'].isin(top_skus)]['unit_ean_prod'].unique().tolist()
 
# #df_prod = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],[top_skus_atual[k]]) 
# df_prod = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],['7898403782387']) 
 

# df_prod[['Gmv','Gmv Acum']].tail(20) 
# df_RJ = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Chocolates'],['ean'])  
# df_RJ = df_RJ.reset_index(drop = False)
# hora_teste =  [22]
# df_RJ['Hora'] = df_RJ['DateHour'].dt.hour
# df_RJ[df_RJ['Hora'].isin(hora_teste)]
 
 

# %% Load View Geral
print('Load View Geral')

#@st.cache_resource( ttl = 1600)
@st.cache_resource( ttl = 43200)  # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
def load_view():

    data = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),['Weekday'],['Hora'] , ['region_id'], ['Região'], ['size'],['categoria'],['ean'])

    return data


 

df_view = load_view()  
year_list = list(df_view.Ano.unique())[::-1]
month_list = list(df_view.Mês.unique())[::-1]
date_list = list(df_view.Date.unique())[::-1]
week_list = list(df_view.Semana.unique())[::-1]
weekday_list = list(df_view.Weekday.unique())[::-1]
hora_list = list(df_view.Hora.unique())[::-1]
region_list = list(df_users['region_id'].unique())[::-1]
regiao_list = list(df_users['region name'].unique())[::-1]
size_list = list(df_users['size_final'].unique())[::-1]


hora_list_inicial = hora_list
weekday_list_inicial = weekday_list
size_list_inicial = size_list  
region_list_inicial = region_list

min_date = df_view['Date'].min()
max_date = df_view['Date'].max() 




# %% Load Region

print('Load Region')

#@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
@st.cache_resource( ttl = 43200) 
def df_regions():
# Load DataFrame 1
    df_rjc = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],['ean']) 
    df_rji = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJI'],size_list,['categoria'],['ean'])
    df_rj1 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[1],  regiao_list,size_list,['categoria'],['ean'])
    df_rj7 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[7],  regiao_list,size_list,['categoria'],['ean'])
    df_rj19 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[19],  regiao_list,size_list,['categoria'],['ean'])
    df_rj36 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[36],  regiao_list,size_list,['categoria'],['ean'])
    df_rj37 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[37],  regiao_list,size_list,['categoria'],['ean'])
    df_rj31 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[31],  regiao_list,size_list,['categoria'],['ean'])
    df_rj27 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[27],  regiao_list,size_list,['categoria'],['ean'])
    df_rj29 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[29],  regiao_list,size_list,['categoria'],['ean'])
    df_rj30 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[30],  regiao_list,size_list,['categoria'],['ean'])
    df_rj24 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[24],  regiao_list,size_list,['categoria'],['ean'])
    df_rj25 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[25],  regiao_list,size_list,['categoria'],['ean'])
    df_rj22 = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp('2024-03-01'),pd.Timestamp(date.today()),weekday_list,hora_list,[22],  regiao_list,size_list,['categoria'],['ean'])
    
    
    df_rjc['Region Final'] = 'RJC'
    df_rji['Region Final'] = 'RJI'
    df_rj1['Region Final'] = 'RJ1'
    df_rj7['Region Final'] = 'RJ7'
    df_rj19['Region Final'] = 'RJ19'
    df_rj36['Region Final'] = 'RJ36'
    df_rj37['Region Final'] = 'RJ37'
    df_rj31['Region Final'] = 'RJ31'
    df_rj27['Region Final'] = 'RJ27'
    df_rj29['Region Final'] = 'RJ29'
    df_rj30['Region Final'] = 'RJ30'
    df_rj24['Region Final'] = 'RJ24'
    df_rj25['Region Final'] = 'RJ25'
    df_rj22['Region Final'] = 'RJ22' 



    data = {"df_rjc": df_rjc, 
            "df_rji": df_rji,
            "df_rj1": df_rj1,
            "df_rj7": df_rj7, 
            "df_rj19": df_rj19,
            "df_rj36": df_rj36,
            "df_rj37": df_rj37,
            "df_rj31": df_rj31,
            "df_rj27": df_rj27,
            "df_rj29": df_rj29,
            "df_rj30": df_rj30,  
            "df_rj24": df_rj24,
            "df_rj25": df_rj25,
            "df_rj22": df_rj22 
            }
    return data
    
cached_data = df_regions()
df_rjc = cached_data["df_rjc"]
df_rji = cached_data["df_rji"]
df_rj1 = cached_data["df_rj1"]
df_rj7 = cached_data["df_rj7"]
df_rj19 = cached_data["df_rj19"]
df_rj36 = cached_data["df_rj36"]
df_rj37 = cached_data["df_rj37"]
df_rj31 = cached_data["df_rj31"]
df_rj27 = cached_data["df_rj27"]
df_rj29 = cached_data["df_rj29"]
df_rj30 = cached_data["df_rj30"]
df_rj24 = cached_data["df_rj24"]
df_rj25 = cached_data["df_rj25"]
df_rj22 = cached_data["df_rj22"]
   
 

 
# %% Load Categoria

print('Load Categoria')

#@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
@st.cache_resource( ttl = 43200) 
def df_categorias():
# Load DataFrame 1
    df_RJ = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['categoria'],['ean'])  
    df_RJ_1_4 = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['categoria'],['ean'])  
    df_RJ_5_9 = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['5-9 Cxs'],['categoria'],['ean'])  
    df_BAC = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['BAC'],['1-4 Cxs'],['categoria'],['ean'])  
    df_Leite = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Leite'],['ean']) 
    df_Acucar = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Açúcar e Adoçante'],['ean']) 
    df_Biscoitos = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Biscoitos'],['ean']) 
    df_Arroz_Feijao = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Arroz e Feijão'],['ean']) 
    df_Derivados_Leite = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Derivados de Leite'],['ean']) 
    df_Oleos = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Óleos, Azeites e Vinagres'],['ean']) 
    df_Cafe = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Cafés, Chás e Achocolatados'],['ean']) 
    df_Massas = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Massas Secas'],['ean']) 
    df_Cervejas = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Cervejas'],['ean']) 
    df_Sucos = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Sucos E Refrescos'],['ean']) 
    df_Refrigerantes = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Refrigerantes'],['ean']) 
    df_Limpeza = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Limpeza de Roupa'],['ean']) 
    df_Chocolates = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Chocolates'],['ean']) 
    df_Graos = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2023-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['Grãos e Farináceos'],['ean']) 



 
    df_RJ['Categoria'] = 'RJC'
    df_RJ_1_4['Categoria'] = 'RJ 1-4 Cxs'
    df_RJ_5_9['Categoria'] = 'RJ 5-9 Cxs'
    df_BAC['Categoria'] = 'BAC'
    df_Leite['Categoria'] = 'Leite' 
    df_Acucar['Categoria'] = 'Açúcar e Adoçante' 
    df_Biscoitos['Categoria'] = 'Biscoitos' 
    df_Arroz_Feijao['Categoria'] = 'Arroz e Feijão' 
    df_Derivados_Leite['Categoria'] = 'Derivados de Leite' 
    df_Oleos['Categoria'] = 'Óleos, Azeites e Vinagres' 
    df_Cafe['Categoria'] = 'Cafés, Chás e Achocolatados' 
    df_Massas['Categoria'] = 'Massas Secas' 
    df_Cervejas['Categoria'] = 'Cervejas' 
    df_Sucos['Categoria'] = 'Sucos E Refrescos' 
    df_Refrigerantes['Categoria'] = 'Refrigerantes'  
    df_Limpeza['Categoria'] = 'Limpeza de Roupa' 
    df_Chocolates['Categoria'] = 'Chocolates' 
    df_Graos['Categoria'] = 'Grãos e Farináceos'

    data = {
        
            "df_RJ": df_RJ, 
            "df_RJ_1_4": df_RJ_1_4, 
            "df_RJ_5_9": df_RJ_5_9, 
            "df_BAC": df_BAC, 
            "df_Leite": df_Leite,
            "df_Acucar": df_Acucar,
            "df_Biscoitos": df_Biscoitos,
            "df_Arroz_Feijao": df_Arroz_Feijao, 
            "df_Derivados_Leite": df_Derivados_Leite,
            "df_Oleos": df_Oleos,
            "df_Cafe": df_Cafe,
            "df_Massas": df_Massas,
            "df_Cervejas": df_Cervejas,
            "df_Sucos": df_Sucos,
            "df_Refrigerantes": df_Refrigerantes,  
            "df_Limpeza": df_Limpeza,
            "df_Chocolates": df_Chocolates,
            "df_Graos": df_Graos 
            }
    return data
    
cached_data = df_categorias() 
 
@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
def df_categorias_fim(): 

    for key, value in cached_data.items(): 
        try:
            df_fim = pd.concat([df_fim, cached_data[key]])
        except:
            df_fim = cached_data[key].copy()
    return df_fim

df_categorias_final = df_categorias_fim() 
 

# %% Load Categoria 2

data_atual = datetime.date.today()
hora_atual = datetime.datetime.now()  - timedelta(hours=3)   

data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")   

df_top_categorias = df_orders.copy()
df_top_categorias = df_top_categorias[['Categoria','Gmv']].groupby('Categoria').sum()
df_top_categorias = df_top_categorias.sort_values('Gmv',ascending = False)
df_top_categorias = df_top_categorias.head(20)
df_top_categorias = df_top_categorias.reset_index()
categoria_list = df_top_categorias['Categoria'].unique().tolist() 
 

dicts = {}
name_list = []
list_dicts = []

regional_list =  ['RJC','RJI','BAC']
size_list = ['size','1-4 Cxs','5-9 Cxs']    
 


@st.cache_resource( ttl = 64800) 
def categorias2(): 

    for k in regional_list:
        region = k

        for z in size_list:
            size = z
                
            for i in categoria_list:  # Categorias RJC 1-4 Cxs 
                categoria = i

                var = categoria + ' ' + region + ' ' + size  
                print('') 
                print(var)
                print('')

                try:
                    df = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  [region ],[size],[categoria],['ean']) 
                    df['Categoria'] = categoria
                    df['Region'] = region 
                    df['Size'] = size
                    dicts = {"df_"+ f'{var}':df} 
                    list_dicts.append(dicts)
                    name_list.append("df_"+ f'{var}')
                except:
                    pass


    df_fim = pd.DataFrame()
    
    for i in range(0,len(name_list)):

        if len(df_fim) ==0: 
            df_fim = list_dicts[i][name_list[i]]
        else:
            df_fim = pd.concat([df_fim, list_dicts[i][name_list[i]]])
        
        
    return df_fim

df_categorias2 = categorias2() 
  
 

# %% Loops Categoria 


button_produto = []
buttons_dic = {} 
cont = 0 
for key, value in cached_data.items():
    cont = cont+1
      
    if cont >=5:
        button = "Produtos " + cached_data[key]['Categoria'].unique()[0]
        button_produto.append(button)
        buttons_dic[button] = False 

  
button_produto2 = []
buttons_dic2 = {} 
cont = 0 
for key, value in cached_data.items():
    cont = cont+1
    if cont >=5:
        button = "Collapse Produtos " + key[3:]
        button_produto2.append(button)
        buttons_dic2[button] = False 

  
  
#data_max = df_view['Date'].max()     


# %% Streamlit Frontend

# %% Side Bar 

with st.sidebar:
    
    #st.image(icon, width=80) 
    st.title('Data Growth App')
        
    date_range = st.slider("Data:", min_value=max_date , max_value=min_date,  value=[min_date, max_date] , format="YY-MM-DD") 
     

    # regiao_list= st.multiselect("Região:", regiao_list )
    # region_id_list= st.multiselect("Region Id:", region_list )
    weekday_list= st.multiselect("Dia da Semana:", weekday_list, default= data_atual.weekday() )
    hora_list= st.multiselect("Hora:", hora_list, default  = max_hora_orders )
    # size_list= st.multiselect("Tipo:", size_list )



    # if len(regiao_list)== 0: regiao_list=['Região']
    # if len(region_id_list)== 0: region_id_list=['region_id']
    #if len(weekday_list)== 0: weekday_list=['Weekday']
   # if len(hora_list)== 0: hora_list=['Hora']
    # if len(size_list)== 0: size_list=['size']

        
        

    # print('region_id_list')
    # print(regiao_list)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False 

    def click_button():
        st.session_state.clicked = True 

    # st.button('Filter', on_click=click_button)

    if st.session_state.clicked: 
        st.session_state.clicked = False  
        df_view = cria_df_view(df_datetime,df_users, df_trafego, df_orders,pd.Timestamp(date_range[0]),pd.Timestamp(date_range[1]),weekday_list,hora_list,region_id_list,  regiao_list,size_list,['categoria'],['ean'])


# %% Variaveis iniciais 

data_min = date_range[0] 
data_max = date_range[1] + pd.offsets.Day(1)
    
 
tab1 , tab0   , tab2, tab3, tab4,tab5 = st.tabs(["Performance D0","Categorias", "Forecast Gmv D0", "Forecast Peso D0", "Forecast Gmv D+1",'Trafego'])
df_count = 0 

button_count = 0   

# %% Categorias 

with tab0:
 

    for key, value in cached_data.items():

        df_count = df_count + 1 
 
        if df_count == 1:
            
            st.markdown('###### Atualizado em: ' + str(hora_atualizacao) + ' / Hora Filtrada: ' + str(max_hora_orders))  
            

            st.header("Geral") 
 

            st.markdown('#### ' + cached_data[key]['Categoria'].unique()[0])  

            df_plot =  cached_data[key].copy()   
            df_plot = df_plot.reset_index(drop = False)
            df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
            df_plot = df_plot.set_index('DateHour') 

 

            df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
            df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 

            if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
            if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]

 
            dados_x =  df_plot.index
            dados_y1 =  df_plot['Positivação Categoria']
            dados_y2 =  df_plot['Gmv Acum'] 

            df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
            dados_x_forecast =  df_plot_forecast.index
            dados_y3 = df_plot_forecast['Forecast Gmv'] 
             
            col = st.columns((2,  2, 2), gap='medium')

            with col[0]:
                
                fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação" , x="Data") , height=300, width= 450, markers = True,    line_shape='spline')

                fig 

            with col[1]:
                
                fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum", x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                fig      
            
            with col[2]:
                
                fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                fig        


            with st.expander('Detalhes', expanded= False):
                st.markdown('#### ' + cached_data["df_RJ_1_4"]['Categoria'].unique()[0])  

            
                df_plot =  cached_data['df_RJ_1_4'].copy()   
                df_plot = df_plot.reset_index(drop = False)
                df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
                df_plot = df_plot.set_index('DateHour') 

                df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
                df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 

                if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
                if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
 
                dados_x =  df_plot.index
                dados_y1 =  df_plot['Positivação Categoria']
                dados_y2 =  df_plot['Gmv Acum'] 

                df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
                dados_x_forecast =  df_plot_forecast.index
                dados_y3 = df_plot_forecast['Forecast Gmv'] 
                
                
                col = st.columns((2,  2, 2), gap='medium')

                with col[0]:
                    
                    fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação", x="Data", ) , height=300, width= 450, markers = True,    line_shape='spline')

                    fig 

                with col[1]:
                    
                    fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum", x="Hora") , height=300, width= 450, markers = True,    line_shape='spline')

                    fig      
                
                with col[2]:
                    
                    fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(x="Data", y="Forecast Gmv") , height=300, width= 450, markers = True,    line_shape='spline')

                    fig      

                st.markdown('#### ' + cached_data["df_RJ_5_9"]['Categoria'].unique()[0])  

                df_plot =  cached_data['df_RJ_5_9'].copy()   
                df_plot = df_plot.reset_index(drop = False)
                df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
                df_plot = df_plot.set_index('DateHour') 

                if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
                if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
                df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
                df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 


                #df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
                dados_x =  df_plot.index
                dados_y1 =  df_plot['Positivação Categoria']
                dados_y2 =  df_plot['Gmv Acum'] 

                df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
                dados_x_forecast =  df_plot_forecast.index
                dados_y3 = df_plot_forecast['Forecast Gmv']  
                
                col = st.columns((2,  2, 2), gap='medium')



                with col[0]:
                    
                    fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação", x="Data") , height=300, width= 450, markers = True,    line_shape='spline')

                    fig 

                with col[1]:
                    
                    fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Hora" ) , height=300, width= 450, markers = True,    line_shape='spline')

                    fig      
                
                with col[2]:
                    
                    fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                    fig      

  
        elif df_count == 4:
            st.markdown('#### ' + cached_data[key]['Categoria'].unique()[0])  

            df_plot =  cached_data[key].copy()   
            df_plot = df_plot.reset_index(drop = False)
            df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
            df_plot = df_plot.set_index('DateHour') 

            if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
            if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
            df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
            df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 


           # df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
            dados_x =  df_plot.index
            dados_y1 =  df_plot['Positivação Categoria']
            dados_y2 =  df_plot['Gmv Acum'] 

            df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
            dados_x_forecast =  df_plot_forecast.index
            dados_y3 = df_plot_forecast['Forecast Gmv'] 
            
            
            col = st.columns((2,  2, 2), gap='medium')

            with col[0]:
                
                fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação Categoria" , x="Data") , height=300, width= 450, markers = True,    line_shape='spline')

                fig 

            with col[1]:
                
                fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Hora") , height=300, width= 450, markers = True,    line_shape='spline')

                fig      
            
            with col[2]:
                
                fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                fig        


         
        elif df_count >= 5 : 
            
            if df_count==5: st.header("Categoria")   

            st.markdown('#### ' + cached_data[key]['Categoria'].unique()[0])  
            col = st.columns((2,  2, 2), gap='medium')
            with col[0]:
                
                            
                df_plot =  cached_data[key].copy()   
                df_plot = df_plot.reset_index(drop = False)
                df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
                df_plot = df_plot.set_index('DateHour') 

                if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]

                if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]


                df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
                df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 
               # df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
                dados_x =  df_plot.index
 
                dados_y =  df_plot['Positivação Categoria']
                dados_y2 =  df_plot['% Share Positivação Categoria'] 
                dados_y3 =  df_plot['Gmv Acum'] 
                fig=py.line(x=dados_x, y=dados_y,   title = 'Positivação Categoria' ,  labels=dict(y="Positivação", x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                fig 

            with col[1]:
                
                fig=py.line(x=dados_x, y=dados_y3,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                fig     
            
            with col[2]:
                
                fig=py.line(x=dados_x, y=dados_y2,  title = '% Positivação Categoria' ,  labels=dict(y="% Positivação" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                fig        
    
  
            # with st.expander('Detalhes', expanded= False):

            #     var = 10 


                # st.markdown('#### Métricas Tráfego Categoria' )   

 
                # col = st.columns((2,  2), gap='medium')
                # with col[0]:
                    
                                
                #     df_plot =  cached_data[key].copy()
                #     df_plot = df_plot.reset_index(drop = False)
                #     df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
                #     df_plot = df_plot.set_index('DateHour') 
        

                #     if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
                #     if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]

                #     df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
                #     df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 
                #     #df_plot = df_plot[df_plot['Hora'] == max_hora_trafego]
            
                    
                #     dados_x =  df_plot.index 
                #     dados_y =  df_plot['search_products Acum']
                #     dados_y2 =  df_plot['% Conversão Search Acum'] 
                #     fig=py.line(x=dados_x, y=dados_y,   title = 'Search Products' ,  labels=dict(y="Search Product", x="Data" ) , height=300, width= 500, markers = True,    line_shape='spline')

                #     fig 

                # with col[1]:
                    
                #     fig=py.line(x=dados_x, y=dados_y2,  title = '% Conversão Search Acum' ,  labels=dict(y="% Conversão Search Acum" , x="Hora") , height=300, width= 500, markers = True,    line_shape='spline')

                #     fig     

                

            categoria_atual = cached_data[key]['Categoria'].unique()[0]     

            for key in buttons_dic:

                  

                if key == 'Produtos ' + categoria_atual: 
                
                    
                    buttons_dic[key] = st.checkbox('Detalhe ' + key)
                    
                    
                    if buttons_dic[key]:  

                            
                        st.markdown('#### Métricas Top Skus' )  
                        
                        st.markdown('#### ' )  


                        top_skus_atual = df_orders[df_orders['Categoria'] == categoria_atual ][df_orders['unit_ean_prod'].isin(top_skus)]['unit_ean_prod'].unique().tolist()
                            
                        for k in range(0,len(top_skus_atual)):

                            produto = df_orders[df_orders['Categoria'] == categoria_atual ][df_orders['unit_ean_prod'] == top_skus_atual[k]]['Produtos'].unique()[0]
                            ean_prod = df_orders[df_orders['Categoria'] == categoria_atual ][df_orders['unit_ean_prod'] == top_skus_atual[k]]['unit_ean_prod'].unique()[0]
                            st.markdown('##### ' + produto )                            
                                
                                            
                            df_prod = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],[top_skus_atual[k]]) 
                                

                            df_plot =  df_prod.copy()   
                            df_plot = df_plot.reset_index(drop = False)
                            df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
                            df_plot = df_plot.set_index('DateHour')  

                            col = st.columns((2,  2, 2), gap='medium')

                            with col[0]:
                                
                                            
                                df_plot =  df_prod.copy()   
                                df_plot = df_plot.reset_index(drop = False)
                                df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
                                df_plot = df_plot.set_index('DateHour') 
                    

                                if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
                                if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]


                                df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
                                df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 
                                #  df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
                                
                                dados_x =  df_plot.index
                                dados_y =  df_plot['Positivação Categoria']
                                dados_y2 =  df_plot['% Share Positivação Categoria'] 
                                dados_y3 =  df_plot['Gmv Acum'] 
                                fig=py.line(x=dados_x, y=dados_y,   title = 'Positivação Categoria' ,  labels=dict(y="Positivação", x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                                fig 

                            with col[1]:
                                
                                fig=py.line(x=dados_x, y=dados_y3,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                                fig     
                            
                            with col[2]:
                                
                                fig=py.line(x=dados_x, y=dados_y2,  title = '% Positivação Categoria' ,  labels=dict(y="% Positivação" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                                fig    


    st.markdown("###") 

    st.markdown("### Histórico") 
    st.markdown("###") 


    print(df_categorias_final.columns.to_list())
    df_categoria_historico = df_categorias_final[['Categoria','Gmv Acum','Positivação Categoria','Positivação Geral']]
    df_categoria_historico = df_categoria_historico.sort_values( by = ['DateHour','Gmv Acum'] , ascending=[False,False])
    df_categoria_historico


# %% Trend, Ruputuras e Pricing 


with tab1: 
                              
    df_categorias2 = df_categorias2.rename(columns={'% Share Positivação Categoria':'% Positivação Categoria'})   
    df_categorias2['Hora'] = df_categorias2.index.hour 
    df_categorias2['Data'] = df_categorias2.index.date  
    df_categorias2["week"] = df_categorias2.index.isocalendar().week
    df_categorias2["day_of_month"] = df_categorias2.index.day
    df_categorias2["month"] = df_categorias2.index.month

    categoria_list = df_categorias2.sort_values('Categoria', ascending = True).Categoria.unique().tolist()
    
    

    def df_plot_trend(df_plot, categoria_list,var, tipo):
        
        trend_list = []     
        dict_trends = {}
        

        for i in categoria_list:
        
            df_trend = df_plot.copy()
            df_trend = df_trend[df_trend[tipo] == i]
                
            df_trend =  pd.DataFrame(df_trend.asfreq('d').index).set_index('Data').merge(df_trend, left_index = True, right_index=True,how = "left")
            
            df_trend = df_trend[[var]]
            
            

            res = STL(  df_trend,  robust=True,  ).fit()


            df_trend["residual"] = res.resid
            df_trend["Trend " + i] = res.trend 
            df_trend["seasonal"] = res.seasonal 
            df_trend = df_trend.reset_index('Data')

            slopeList = df_trend["Trend " + i].tolist()
            slope1 = (slopeList[-1] - slopeList[0]) / (len(slopeList) - 1) 

            # X = np.arange(len(slopeList)).reshape(-1, 1)  # Independent variable (time)
            # y = slopeList
            # model = LinearRegression()
            # model.fit(X, y)
            # slope2 = model.coef_[0] 
            dict_trends = {tipo: i, 'Slope 1': slope1}
            trend_list.append(dict_trends) 
            
            df_trend = df_trend.set_index('Data')
            
            df_plot = df_plot.merge(  df_trend[['Trend ' + i]], how='left', left_index=True, right_index=True)  
            
            df_trend_slope = pd.DataFrame(trend_list)[[tipo,'Slope 1']].sort_values('Slope 1', ascending = True)
            
            df_trend_slope  = df_trend_slope.set_index(tipo)[['Slope 1']]


        return df_plot, df_trend_slope, var


    def plot_trends(df_plot,categoria_filter, var, tipo ):

        if len(categoria_filter)== 0: categoria_filter=[tipo]

        df_plot_trend = df_plot.copy()

        if categoria_filter[0] != tipo: df_plot_trend = df_plot_trend[df_plot_trend[tipo].isin(categoria_filter)]

        lista_trend = df_plot.filter(regex='Trend').columns.to_list()  
        lista_trend_final = []

        for i in lista_trend: 
            
            for k in categoria_filter:            
                if i[6:].find( k ,0)==0: lista_trend_final.append(i)


        if len(lista_trend_final) !=0 : lista_trend = lista_trend_final


        df_plot_trend = df_plot_trend[lista_trend].reset_index()

        df_plot_trend = df_plot_trend.groupby(df_plot_trend['Data']).max()

        df_plot_trend = df_plot_trend.reset_index()


    
        fig = px.line(df_plot_trend, x="Data", y=df_plot_trend.columns , hover_data={"Data": "|%B %d, %Y"}, width=1200) #, title='Trend Categorias')
        fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
        st.plotly_chart(fig, theme="streamlit")

 

    Trend , Rupturas, Pricing    = st.tabs([ "Trend","Rupturas", "Pricing"  ])
    
# %% Trend 
    
    with Trend: 
        
        with st.expander('Filtros', expanded= False):

            with st.form(key = "my_form"):
                
                col = st.columns((4, 6), gap='medium')
                
                with col[0]:
                    col = st.columns((2, 2), gap='medium')

                    with col[0]:

                        var_trends = ['% Positivação Categoria', 'Positivação Categoria', 'Gmv Acum' ]
                        var = st.radio("Métrica", var_trends ) 

                    with col[1]:
                        categoria_filter = st.multiselect("Categoria", categoria_list,)

                submit_button = st.form_submit_button(label = "Submit")
         

        regional_list =  ['RJC', 'RJI','BAC']  
        size_list = ['size','1-4 Cxs','5-9 Cxs']   
        df_trend_slope_final = pd.DataFrame()

        for i in regional_list:
            
              
            for s in size_list:


                df_plot = df_categorias2.copy()
                df_plot = df_plot.reset_index() 
                df_plot = df_plot.set_index('Data') 
                df_plot = df_plot[df_plot['Hora']==23]  

                df_plot = df_plot[df_plot['Region']== i ]  
                df_plot = df_plot[df_plot['Size']== s]   
        

                if s == 'size': size_name = 'Geral' 
                else: size_name = s 

                nome_var =  i + ' - ' + size_name + ' - Trend ' + var  
 

                df_plot, df_trend_slope ,var = df_plot_trend(df_plot,categoria_list, var, 'Categoria')

 
                if   i + ' - ' + size_name != 'BAC - 5-9 Cxs':


                    
                    st.markdown("##  " +  nome_var)
                    st.markdown("##  "  )

                    col = st.columns((2,  8), gap='medium') 

                    with col[0]:
                                    
                        df_trend_slope = df_trend_slope.rename(columns = {'Slope 1': 'Slope'})
                        df_trend_slope

                        df_trend_slope = df_trend_slope.rename(columns = {'Slope': 'Slope ' + i + ' - ' + size_name})

                        if len(df_trend_slope_final) == 0: 
                            df_trend_slope_final = df_trend_slope.copy()                                
                        else:
                            df_trend_slope_final = df_trend_slope_final.merge( df_trend_slope, how='left', left_index=True, right_index=True)  
                    
                    with col[1]:
            
                        plot_trends(df_plot,categoria_filter, var, 'Categoria')


        st.markdown("##  Resumo Slopes")
        st.markdown("##  "  )
        
        df_trend_slope_final


# %% Ruputuras

    with Rupturas: 
        
        with st.expander('Filtros', expanded= False):

            with st.form(key = "my_forms"):
                
                col = st.columns((4, 6), gap='medium')
                
                with col[0]:
                    col = st.columns((2, 2), gap='medium')

                    with col[0]:
                        categoria_filters = st.radio("Categoria", categoria_list )

                        var_trends = ['% Positivação Categoria', 'Positivação Categoria', 'Gmv Acum' ]
#                        var_change_points = st.radio("Métrica", var_trends ) 
                        var_change_points = '% Positivação Categoria'
                    with col[1]:
                       var_change_points = '% Positivação Categoria'
 
                submit_button = st.form_submit_button(label = "Submit")
         
 
        regional_list =  ['RJC', 'RJI','BAC']    
        size_list = ['size','1-4 Cxs','5-9 Cxs']   

        def rupturas_targets(df_categorias2,regional_list ,size_list,categoria_filters, tipo ): 
 
            df_trend_slope_final = pd.DataFrame() 
            df_categoria_target = pd.DataFrame()  
            df_plot_final = pd.DataFrame()  

 
            for c in categoria_filters:
    
                for i in regional_list: 
                    
                    for s in size_list: 
                        flag_zera_ofertao = 0 
                        df_plot = df_categorias2.copy()
                        df_plot = df_plot.reset_index() 
                        df_plot = df_plot.set_index('Data') 
                        df_plot = df_plot[df_plot['Hora']==23]  
                    

                        df_plot = df_plot[df_plot['Categoria'] == c]  

                        df_plot = df_plot[df_plot['Size']== s]       

                        df_plot = df_plot[df_plot['Region']== i ]  

                         
                        if s == 'size': size_name = 'Geral' 
                        else: size_name = s 


                        if str(i) + ' - ' +  str(size_name) != 'BAC - 5-9 Cxs':
                            
                             
                            nome_var =  c + ' ' +  str(i) + ' - ' +  str(size_name) + ' - Trend ' + str(var_change_points) + ' + Rupturas' 
 

                            #st.text( str(i) + ' - ' +  str(size_name) + ' ' + tipo)

                            
                            if tipo != "Geral"  :

                                
                                df_ofertao_trend = ofertao(df_ofertao_inicial,[c], [i], [s])
                                
                                df_ofertao_trend = df_ofertao_trend.rename(columns = {'Ofertão ' + c :'Ofertão' }) 

                                df_plot = df_plot.merge( df_ofertao_trend, how='left', left_index=True, right_index=True) 
 
                                
                                if len(df_ofertao_trend) <= 1:
                                    df_plot['Ofertão'] = 0 

 



                                df_plot['Ofertão'] = df_plot['Ofertão'].replace(np.nan,0)
                        

                                if tipo == "Ofertão"  :
                                                

                                                
                                    if len(df_plot[df_plot['Ofertão']>0])>0: 
 
                                        flag_zera_ofertao = 0
                                        df_plot = df_plot[df_plot['Ofertão']>0]

                                    else:

                                        flag_zera_ofertao = 1 
                                        df_plot = df_plot[df_plot['Ofertão']<=0]                                            

                                elif  tipo == "Bau":
                                    
                                    df_plot = df_plot[df_plot['Ofertão']<=0]


                            # if s == '5-9 Cxs':

                            #     st.text( str(i) + ' - ' +  str(size_name) + ' ' + tipo + ' ' +  c)
                            #     df_plot

 


                            df_plot = df_plot[['Categoria','Positivação Categoria', 'Positivação Geral', '% Positivação Categoria']]
                            

                            df_plot =  pd.DataFrame(df_plot.asfreq('d').index).set_index('Data').merge(df_plot, left_index = True, right_index=True,how = "left")
                        
                            df_plot = df_plot[ [var_change_points ]] 
                            df_plot[var_change_points ] = df_plot[[var_change_points]].fillna(method='ffill')

                            df_plot['Categoria'] = c  
                            
               
                            df_plot, df_trend_slope ,var = df_plot_trend(df_plot,[c], var_change_points, 'Categoria')
                            


                            df_plot = df_plot['Trend ' + c]
            

                            data = df_plot.copy()  
                            

                            dt_today = datetime.datetime.today()
                            dt_today = dt_today.strftime("%Y-%m-%d %H:%M:%S")
        

                            algo_c = rpt.KernelCPD(kernel= "linear", min_size = 3).fit(data.values )
                            change_points = algo_c.predict(n_bkps = 5)
                            
                            df_change_points = pd.DataFrame(change_points)
                    
                            df_change_points['Change Point'] = 1
                            df_change_points['Change Point'] = df_change_points.groupby(df_change_points['Change Point'])['Change Point'].cumsum()
                            df_change_points = df_change_points.set_index(0)
                            
                            data = data.reset_index(drop = False).merge( df_change_points, how='left', left_index=True, right_index=True)
                            data['Change Point Number']  = np.where(data['Change Point'] > 0  , data['Change Point']  , 0 )
                            data['Change Point']  = np.where(data['Change Point']  >0  , 1 , 0 )
                    

                            data = data.set_index('Data')
                            change_point_list = data[data['Change Point']>0].index.tolist()
                            change_point_list.append(dt_today)
                            
                                        
 
                            df_rupturas = change_point_list.copy() 
                            df_rupturas = pd.DataFrame(df_rupturas)
                            df_rupturas = df_rupturas.reset_index(drop = False)
                            df_rupturas['index'] = df_rupturas['index'] + 1
                            
                            
                            for cp in range(0 , len(change_point_list)):
                                if cp == 0:
                                    var_change_point = change_point_list[cp]
                        
                                    data['Change Point Number'] = np.where(data.index <= var_change_point , cp +1, data['Change Point Number'] )

                                else:
                                    var_change_point = change_point_list[cp -1] 
                                    var_change_point2 = change_point_list[cp ]
                                    

                                    data['Change Point Number'] = np.where((data.index >  change_point_list[cp -1]) &  (data.index <=  change_point_list[cp ]) , cp +1, data['Change Point Number'] )

                            trend_groups = data.reset_index(drop = False)
                            trend_groups['Data2'] = trend_groups['Data']
                            trend_groups = trend_groups.groupby('Change Point Number').agg({'Data':'min', 'Data2':'max', 'Trend ' + c:'mean' })
                            trend_groups = trend_groups.rename( columns = {'Trend ' + c:'Trend'})
                            
                            trend_groups['Categoria'] = c
                            trend_groups['Size'] = s
                            trend_groups['Region'] = i  


                            
                            trend_groups = trend_groups.sort_values('Trend', ascending = False)

                            trend_groups  = trend_groups.reset_index(drop = False).reset_index(drop = False) 
                            
                            max_cp = trend_groups.reset_index(drop = False)['Change Point Number'].max()

                            trend_groups['Trend Atual'] =  np.where(trend_groups['Change Point Number']==max_cp, trend_groups['Trend'] ,0)
                            
                            trend_groups['Trend Top 1'] =  np.where(trend_groups['index']==0, trend_groups['Trend'] ,0)
                            trend_groups['Trend Top 2'] =  np.where(trend_groups['index']==1, trend_groups['Trend'] ,0)

  
                            trend_groups = trend_groups.set_index('Change Point Number')
                             
                            trend_groups = trend_groups[[ 'index','Data','Data2','Categoria','Region','Size','Trend','Trend Atual','Trend Top 1','Trend Top 2']]
                             


                            df_plot = df_plot.reset_index()  
                            df_plot['Categoria'] = c
                            df_plot['Size'] = s
                            df_plot['Region'] = i  
                            df_plot['Tipo'] = tipo  
                            df_plot = df_plot.rename(columns = {'Trend ' + c : 'Trend'})
                            df_plot = df_plot[['Data','Tipo','Categoria','Region','Size','Trend']]
                              

                            if flag_zera_ofertao == 0: 

                                if len(df_plot_final) == 0:
                                    df_plot_final = df_plot.copy()
                                else:
                                    df_plot_final = pd.concat([df_plot_final, df_plot])


                                if len(df_categoria_target) == 0:
                                    df_categoria_target = trend_groups.copy()
                                else:
                                    df_categoria_target = pd.concat([df_categoria_target, trend_groups])
                                
                         
             
            df_categoria_target['Tipo'] = tipo      
 
             
            dict_trends_targets = {
                "df_categoria_target": df_categoria_target,                        
                "df_plot_final": df_plot_final,
            }  

            return dict_trends_targets 
         
 
        dict_trends_targets_geral = {}
        dict_trends_targets_bau = {}
        dict_trends_targets_ofertao = {} 
 

        @st.cache_resource( ttl = 450000) 
        def load_categoria_target():
           
            dict_trends_targets_geral  = rupturas_targets(df_categorias2,regional_list ,size_list,categoria_list, 'Geral' ) 
            dict_trends_targets_ofertao = rupturas_targets(df_categorias2,regional_list ,size_list,categoria_list, 'Ofertão' )
            dict_trends_targets_bau = rupturas_targets(df_categorias2,regional_list ,size_list,categoria_list, 'Bau' )
            
            df_categoria_target_geral = dict_trends_targets_geral["df_categoria_target"]
            df_categoria_target_bau =  dict_trends_targets_bau["df_categoria_target"]
            df_categoria_target_ofertao =  dict_trends_targets_ofertao["df_categoria_target"]

            df_plot_geral = dict_trends_targets_geral["df_plot_final"]
            df_plot_bau = dict_trends_targets_bau["df_plot_final"]
            df_plot_ofertao = dict_trends_targets_ofertao["df_plot_final"]

             

            df_categoria_target = pd.concat([ df_categoria_target_geral, df_categoria_target_bau,  df_categoria_target_ofertao])
            df_plot_final = pd.concat([ df_plot_geral, df_plot_bau,  df_plot_ofertao]) 
 
            return df_categoria_target, df_plot_final
        
 

        df_categoria_target , df_plot_final = load_categoria_target()
 
 
        # df_categoria_target = dict_trends_targets["df_categoria_target"]
        # df_plot = dict_trends_targets["df_plot"]
         
 
        tipo_list = ['Geral','Bau','Ofertão']
         

        for r in regional_list:     
           for s in size_list: 
                
                df_change_point = df_categoria_target.copy() 
                df_change_point = df_change_point[df_change_point['Categoria']== categoria_filters]
                df_change_point = df_change_point[df_change_point['Region'] == r]
                df_change_point = df_change_point[df_change_point['Size'] == s]
                 
                
                df_plot_trend_ruputuras = df_plot_final.copy() 
                df_plot_trend_ruputuras = df_plot_trend_ruputuras[df_plot_trend_ruputuras['Categoria']== categoria_filters]
                df_plot_trend_ruputuras = df_plot_trend_ruputuras[df_plot_trend_ruputuras['Region'] == r]
                df_plot_trend_ruputuras = df_plot_trend_ruputuras[df_plot_trend_ruputuras['Size'] == s]
                df_plot_trend_ruputuras = df_plot_trend_ruputuras.set_index('Data')
                 
                var_atual =  r + ' ' + s


                    
                if var_atual != 'BAC 5-9 Cxs':

                    st.markdown("##  Trends " + categoria_filters + ' ' + r + ' ' + s  )
    

                    col0, col1, col2  = st.columns(3)

                    with col0: 
                        
                        var_tipo = 'Geral'
                        df_geral = df_plot_trend_ruputuras[df_plot_trend_ruputuras['Tipo']==var_tipo]
                        change_point_list_geral = df_change_point[df_change_point['Tipo']==var_tipo]
                        change_point_list_geral = change_point_list_geral['Data2'].unique().tolist()
    

                        fig = go.Figure()
    
                        sales_trace = go.Scatter(
                            x=df_geral.index,
                            y=df_geral['Trend'], 
                            yaxis="y1", 
                        )

                        fig.add_trace(sales_trace)   

                        for cp in change_point_list_geral:
                            fig.add_vline(x=cp, line_width=3, line_dash="dash", line_color="red")


                        fig.update_layout(title=var_tipo, width = 350, height = 300)

                
                        st.plotly_chart(fig)

            
                    with col1:   
    
                        var_tipo = 'Bau'
                        df_bau = df_plot_trend_ruputuras[df_plot_trend_ruputuras['Tipo']==var_tipo]
                        change_point_list_bau  = df_change_point[df_change_point['Tipo']==var_tipo]
                        change_point_list_bau = change_point_list_bau['Data2'].unique().tolist()
    

                        fig = go.Figure()
    
                        sales_trace = go.Scatter(
                            x = df_bau.index,
                            y = df_bau['Trend'], 
                            yaxis="y1", 
                        )

                        fig.add_trace(sales_trace)   
                        
                        for cp in change_point_list_bau:
                            fig.add_vline(x=cp, line_width=3, line_dash="dash", line_color="red")

                        fig.update_layout(title=var_tipo, width = 350, height =300)
                        st.plotly_chart(fig)


                    with col2:   
    
                        var_tipo = 'Ofertão'
                        df_ofertao = df_plot_trend_ruputuras[df_plot_trend_ruputuras['Tipo'] == var_tipo]
                        change_point_list_ofertao  = df_change_point[df_change_point['Tipo'] == var_tipo]
                        change_point_list_ofertao = change_point_list_ofertao['Data2'].unique().tolist()
                        

                        fig = go.Figure()
    
                        sales_trace = go.Scatter(
                            x = df_ofertao.index,
                            y = df_ofertao['Trend'], 
                            yaxis="y1",  
                        )

                        fig.add_trace(sales_trace)   
                        
                        for cp in change_point_list_ofertao:
                             
                            fig.add_vline(x=cp, line_width=3, line_dash="dash", line_color="red")

                        fig.update_layout(title= var_tipo, width = 350, height =300)
                        st.plotly_chart(fig)

                        
                    # with col[1]: 

                    #     df_categoria_plot

                    # with col[2]: 

                    #     df_categoria_plot
 


        df_target = df_categoria_target.copy()
 
        df_target = df_target[['Tipo','Categoria','Region','Size','Trend Atual','Trend Top 1','Trend Top 2']].groupby(['Tipo','Categoria','Region','Size']).max()

        df_target['Delta Top 1'] =   np.where(  df_target['Trend Atual'] < df_target['Trend Top 1']  ,  df_target['Trend Atual'] -  df_target['Trend Top 1']  ,  0 )
        df_target['Delta Top 2'] =   np.where(  df_target['Trend Atual'] < df_target['Trend Top 2']  ,  df_target['Trend Atual'] -  df_target['Trend Top 2']  ,  0 )
        df_target['Delta Médio'] = (df_target['Delta Top 1'] +  df_target['Delta Top 2'])/2
         
 
        df_action = df_target.copy()
        df_action = df_action.reset_index(drop = False)
        df_action['Trend Top']  = df_action['Trend Top 2'] 
        df_action['Delta Top']  = df_action['Delta Top 2'] 

        df_action = df_action[['Tipo','Categoria','Region','Size','Trend Atual','Trend Top','Delta Top']]
        

  
        st.markdown("###  Action"  ) 

        
        tipo_list = st.radio("Tipo", ['Geral', 'Bau','Ofertão'] )


        st.markdown("#####  "  )      
        
        col0, col1  = st.columns(2)  
        
         
        
        for r in ['RJC', 'RJI','BAC']:
            
            df_action_region = df_action[df_action['Region']== r] 


            df_action_1_4 = df_action_region[['Size','Tipo','Categoria','Trend Atual','Trend Top','Delta Top']]  
            df_action_1_4 = df_action_1_4[df_action_1_4['Size']=='1-4 Cxs'] 
            df_action_1_4 = df_action_1_4[df_action_1_4['Tipo']== tipo_list] 

            
            df_action_1_4 = df_action_1_4.set_index('Size')
            df_action_1_4 = df_action_1_4.sort_values('Delta Top', ascending = True)
    
            df_action_5_9 = df_action_region[['Size','Tipo','Categoria','Trend Atual','Trend Top','Delta Top']]  
            df_action_5_9 = df_action_5_9[df_action_5_9['Size']=='5-9 Cxs']
            df_action_5_9 = df_action_5_9[df_action_5_9['Tipo']== tipo_list] 
            df_action_5_9 = df_action_5_9.set_index('Size')
            df_action_5_9 = df_action_5_9.sort_values('Delta Top', ascending = True)

        
            with col0:

                st.markdown("#####  " + r +  " 1-4 Cxs"  )      
                df_action_1_4
            
            with col1:
            
                st.markdown("#####  " +  r  +   " 5-9 Cxs"  )      
                df_action_5_9




        st.markdown("####  Action Todos"  )

        
        df_action = df_action.sort_values('Delta Top', ascending = True)
        df_action


        # df_target['Target'] = np.where(df_target['Trend'] ==  df_target['Trend Atual'], df_target['Trend Atual']*1.10,   df_target['Trend']  )
        # df_target['Target'] = np.where(df_target['Target'] >=  df_target['Trend Atual']*1.1, df_target['Trend Atual']*1.1,   df_target['Trend']  )
            


        st.markdown("####  Base"  )
        df_target = df_target.reset_index(drop = False)
        df_target


# %%  Trend Produtos


    # with Trend_Produtos:
        
    #     with st.expander('Filtros', expanded= False):

    #         with st.form(key = "my_form_produtos"):
                
    #             #col = st.columns((4, 6), gap='medium')

                
    #             col1, col2, col3,col4, col5 = st.columns(5)


    #             with col1: 

    #                 var_trends = ['% Positivação Categoria' , 'Positivação Categoria','Gmv Acum' ]
    #                 var = st.radio("Métrica", var_trends ) 

                    
    #             with col2:    

    #                 regional_list = st.multiselect("Regional", ['RJC', 'RJI','BAC']  , default = ['RJC'] ) 

    #             with col3:
                    
                    
    #                 size_list = st.radio("Size", ['1-4 Cxs' , 'size','5-9 Cxs'] )

    #             with col4:   
                        

    #                 categoria_filter_produtos = st.radio("Categoria", categoria_list,)
    #                 categoria = categoria_filter_produtos

    #             with col5:   
                
    #                 df_top_prod = df_orders[df_orders['Categoria'] == categoria_filter_produtos][['Produtos','Gmv']].groupby('Produtos').sum().sort_values('Gmv', ascending = False)
    #                 lista_produtos = df_top_prod.reset_index()[['Produtos']]['Produtos'].unique().tolist()
    #                 df_top_prod = df_top_prod.reset_index()[['Produtos']].head(5)['Produtos'].unique().tolist()
    #                 produtos_selected = st.multiselect("Produtos", lista_produtos, default = df_top_prod ) 

    #             submit_button = st.form_submit_button(label = "Submit")


    #     dicts_products = {}
    #     name_list_products = []
    #     list_dicts_products = []

    #     for k in regional_list:
    #         region = k

    #         for z in size_list:
    #             size = z

    #             for p in range(0,len(produtos_selected)):
    #                 produto = produtos_selected[p]
    #                 print_name = categoria + ' - ' + ' ' +  produto + ' ' + region + ' ' + size  

        
    #                 try:
    #                     df_product= cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list_inicial,hora_list_inicial,region_list_inicial,  [region ],[size],[categoria],[produto]) 
    #                     df_product['Categoria'] = categoria
    #                     df_product['Produtos'] = produto 
    #                     df_product['Region'] = region 
    #                     df_product['Size'] = size
                        
    #                     dicts_products = {"df_"+ f'{print_name}': df_product} 
    #                     list_dicts_products.append(dicts_products)
    #                     name_list_products.append("df_"+ f'{print_name}')
                        

    #                 except: 
    #                     pass



    #     df_fim_product = pd.DataFrame()
        
    #     for i in range(0,len(name_list_products)):

    #         if len(df_fim_product) == 0: 
    #             df_fim_product = list_dicts_products[i][name_list_products[i]]
    #         else:
    #             df_fim_product = pd.concat([df_fim_product, list_dicts_products[i][name_list_products[i]]])
            
    #     #produtos_selected = df_fim_product['Produtos'].unique().tolist() 
        
    #     df_fim_product = df_fim_product.rename(columns={'% Share Positivação Categoria':'% Positivação Categoria'})  
        
    #     df_trend_slope_final_products = pd.DataFrame()

    #     for i in regional_list:
            
            
    #         for s in size_list:

    #             df_plot = df_fim_product.copy() 
    #             df_plot = df_plot.reset_index() 
    #             df_plot['Data'] = df_plot['Date']
    #             df_plot = df_plot.set_index('Data') 
    #             df_plot = df_plot[df_plot['Hora']==23]  

    #             df_plot = df_plot[df_plot['Region']== i ]  
    #             df_plot = df_plot[df_plot['Size']== s]    

    #             if s == 'size': size_name = 'Geral' 
    #             else: size_name = s 

    #             nome_var =  i + ' - ' + size_name + ' - Trend '   

            
    #             st.markdown("##  " +  nome_var)
    #             st.markdown("##  "  )
                    
        
    #             df_plot_products, df_trend_slope_products ,var_products = df_plot_trend(df_plot,produtos_selected, var, "Produtos")
                
                
    #             col = st.columns((2,  8), gap='medium') 

    #             with col[0]:
                                
    #                 df_trend_slope_products = df_trend_slope_products.rename(columns = {'Slope 1': 'Slope'})
                    

    #                 df_trend_slope_products = df_trend_slope_products.rename(columns = {'Slope': 'Slope ' + i + ' - ' + size_name})

    #                 if len(df_trend_slope_final_products) == 0: 
    #                     df_trend_slope_final_products = df_trend_slope_products.copy()                                
    #                 else:
    #                     df_trend_slope_final_products  = df_trend_slope_final_products.merge( df_trend_slope_products, how='left', left_index=True, right_index=True)  
                
    #             with col[1]:
        
    #                 plot_trends(df_plot_products,produtos_selected, var_products, "Produtos")


    #     st.markdown("##  Resumo Slopes")
    #     st.markdown("##  "  )
        
    #     df_trend_slope_final_products




# # %% Pricing Forms



#     with Pricing: 


# # %% Pipeline 

  
#         def pipe_df_modelos(df_orders,df_ofertao, df_trafego , df_produtos, df_users, query_concorrencia,  regiao, size, lista_categorias, ean , data_inicio, data_fim ):

#             df = df_orders.copy()  
#             df['Data'] = pd.to_datetime(df['Data'])     
#             df = df[df['Região']== regiao] 

            
#             def top_sku (df):
#                 df_top_skus = df.copy() 
                
#                 df_top_skus = df_top_skus.iloc[int(df.shape[0] *0.5):,:]
#                 df_top_skus = df_top_skus[df_top_skus['Categoria'].isin(lista_categorias)]
#                 df_top_skus = df_top_skus[['Categoria','unit_ean','Unit_Description','Gmv']].groupby(['Categoria', 'unit_ean','Unit_Description']).sum().sort_values(by =['Categoria','Gmv'],ascending= False)
                
                

#                 df_top_skus = df_top_skus.reset_index(drop=False)
#                 df_gmv_categoria = df_top_skus[['Categoria','Gmv']].groupby('Categoria').sum().reset_index(drop = False) 
#                 df_gmv_categoria = df_gmv_categoria.sort_values('Gmv',ascending = False) 
#                 #df_gmv_categoria.head(10)['Categoria'].to_list()

                
#                 df_gmv_categoria = df_gmv_categoria.rename(columns = {'Gmv':'Gmv_Categoria'})
                
                

#                 df_top_skus  = df_top_skus.merge(df_gmv_categoria , how ='left', left_on='Categoria', right_on='Categoria', suffixes=(False, False))
#                 df_top_skus['Share_Produto'] = df_top_skus['Gmv']/df_top_skus['Gmv_Categoria']
#                 df_top_skus['Share_Acumulado'] = df_top_skus.groupby(['Categoria'])['Share_Produto'].cumsum()
#                 df_top_skus['Ranking'] = 1 
#                 df_top_skus['Ranking'] = df_top_skus.groupby(['Categoria'])['Ranking'].cumsum()
#                 df_top_skus = df_top_skus[df_top_skus['Ranking']<=5]
#                 df_top_skus = df_top_skus[df_top_skus['Share_Acumulado']<=0.9].sort_values('Gmv',ascending=False).reset_index(drop=True).reset_index(drop=False)
                
#                 top_skus = df_top_skus['unit_ean'].unique().tolist()
#                 return top_skus 

#             top_skus = top_sku(df)


#             def ofertao(df_ofertao_inicial, lista_categorias, top_skus, regiao):


#                 df_ofertao = df_ofertao_inicial.copy() 
#               #  if lista_categorias[0] != 'lista_categorias': df_ofertao = df_ofertao[df_ofertao['category'].isin(lista_categorias)]
#                 #/if regiao[0] != 'regiao': df_ofertao = df_ofertao[df_ofertao['Região'] == 'RJC']  
#                 df_ofertao['Ofertão'] = 1 

#                 df_ofertao = df_ofertao[df_ofertao['Região'].isin(['RJC','RJI','BAC'])]  
#                 df_ofertao_dia = df_ofertao.copy()  
#                 df_ofertao_prod = df_ofertao.copy()  
#                 if lista_categorias[0] != 'lista_categorias': df_ofertao_prod = df_ofertao_prod[df_ofertao_prod['category'].isin(lista_categorias)]
                
#                 df_ofertao['category'] =  df_ofertao['category'] + ' ' + df_ofertao['Região'] + ' ' + df_ofertao['tipo_ofertao']
#                 df_ofertao_prod['category'] =  df_ofertao_prod['category'] + ' ' + df_ofertao_prod['Região'] + ' ' + df_ofertao_prod['tipo_ofertao']
            


#                 df_ofertao = df_ofertao[['Data','category','price']].groupby(['Data','category']).min('price')
#                 df_ofertao = df_ofertao.reset_index(drop = False)
#                 df_ofertao = df_ofertao.set_index('Data') 

#                 df_ofertao = pd.get_dummies(df_ofertao['category']).astype(float)
#                 df_ofertao = df_ofertao.groupby('Data').max() 
#                 df_ofertao.columns=[ "Ofertão " + str(df_ofertao.columns[k-1])   for k in range(1, df_ofertao.shape[1] + 1)]



#                 #ofertao_columns = list(map(lambda x: 'Ofertão ' + x, lista_categorias ))
#                 #ofertao_columns  = ofertao_columns + ['Ofertão']
#                 #df_ofertao = df_ofertao[ofertao_columns]


#                 #df_ofertao_dia = df_ofertao_dia.iloc[df_ofertao_dia.shape[0]-10:,:]
#                 #df_ofertao_dia  = df_ofertao_dia.set_index('Data')
#                 #df_ofertao_dia = df_ofertao_dia[df_ofertao_dia['Ofertão']==1].T.astype(float)
#                 #df_ofertao_dia['Total'] =   df_ofertao_dia.iloc[:, :].sum(axis=1)
#                 #df_ofertao_dia.sort_values('Total',ascending= False) 



#                 df_ofertao_prod = df_ofertao_prod[['Data','category','Ean','Description','price']].groupby(['Data','category','Ean','Description']).min('price')
#                 df_ofertao_prod = df_ofertao_prod.reset_index(drop = False)
#                 df_ofertao_prod = df_ofertao_prod.set_index('Data') 
#                 df_ofertao_prod['Ean'] = df_ofertao_prod['Ean'].astype(str)
#                 df_ofertao_prod = df_ofertao_prod[df_ofertao_prod['Ean'].isin(top_skus)]
#                 df_ofertao_prod = df_ofertao_prod.reset_index(drop = False)
#                 df_ofertao_prod = df_ofertao_prod.merge(df_produtos  ,how ='left', left_on='Ean', right_on='unit_ean_prod', suffixes=(False, False))


#                 df_ofertao_prod = df_ofertao_prod[df_ofertao_prod['Description'] ==  df_ofertao_prod['Unit_Description']]
#                 df_ofertao_prod = df_ofertao_prod[['Data','category','Ean','Description','price']]
#                 df_ofertao_prod['Product_Ofertao'] = 'Ofertão ' +  df_ofertao_prod['category'].astype(str) + ' ' + df_ofertao_prod['Description'].astype(str) + ' - ' + df_ofertao_prod['Ean'].astype(str)
#                 df_ofertao_prod = df_ofertao_prod[['Data']].merge( pd.get_dummies(df_ofertao_prod['Product_Ofertao']).astype(float)  ,how ='left', left_index= True, right_index= True, suffixes=(False, False))
#                 df_ofertao_prod = df_ofertao_prod.groupby('Data').max()
                 

#                 df_ofertao = df_ofertao.merge(df_ofertao_prod  ,how ='left', left_index = True , right_index = True, suffixes=(False, False))
#                 df_ofertao =  pd.DataFrame(df_ofertao.asfreq('D').index).set_index('Data').merge(df_ofertao, left_index = True, right_index=True,how = "left") 
#                 df_ofertao = df_ofertao.replace(np.nan, 0) 
#             #     df_ofertao['weekday'] = df_ofertao.index.weekday 

#             #     #ofertao_columns = df_ofertao.columns.to_list()

#             # #    df_ofertao_teste = df_ofertao.copy() 
                
#             #     weekday_list = ['Segunda','Terça','Quarta','Quinta','Sexta','Sábado','Domingo']

#             #     for i in range(0, len(df_ofertao['weekday'].unique().tolist())):
                
#             #         if i == 0:

#             #             df_ofertao_teste = df_ofertao[df_ofertao['weekday']==i]           
#             #             df_ofertao_teste = df_ofertao_teste.drop(columns=['weekday'])
#             #             df_ofertao_teste.columns=[ str(df_ofertao_teste.columns[k-1]) + " - weekday " +  weekday_list[i]   for k in range(1, df_ofertao_teste.shape[1] + 1)]            
#             #             df_ofertao_teste = df_ofertao[[]].merge(df_ofertao_teste  ,how ='left', left_index = True , right_index = True, suffixes=(False, False))
                        

#             #         else:

#             #             df_ofertao2 = df_ofertao[df_ofertao['weekday']==i]           
#             #             df_ofertao2 = df_ofertao2.drop(columns=['weekday'])
#             #             df_ofertao2.columns=[ str(df_ofertao2.columns[k-1]) + " - weekday " +  weekday_list[i]   for k in range(1, df_ofertao2.shape[1] + 1)]            
#             #             df_ofertao_teste = df_ofertao_teste.merge(df_ofertao2  ,how ='left', left_index = True , right_index = True, suffixes=(False, False))


#             #     df_ofertao = df_ofertao.merge(df_ofertao_teste  ,how ='left', left_index = True , right_index = True, suffixes=(False, False))
#             #     df_ofertao = df_ofertao.replace(np.nan,0)
#             #     df_ofertao = df_ofertao.drop(columns= ['weekday'])


#                 ofertao_cate_cols = df_ofertao.filter(regex= lista_categorias[0]).columns.to_list() 
                
                
#                 df_ofertao = df_ofertao[ofertao_cate_cols + df_ofertao.drop(columns = ofertao_cate_cols).columns.to_list()]
                 


#                 return df_ofertao 

        
#             df_ofertao = ofertao(df_ofertao, lista_categorias , top_skus, regiao)


#             def df_vendass(df):
                
#                 df_vendas = df.copy() 
#                 df_vendas['key'] = df_vendas['Data'].astype(str) + df_vendas['customer_id'].astype(str)
#                 df_vendas['Pedidos'] = df_vendas['key'] 
#                 df_vendas['Positivação'] = df_vendas['Data'].astype(str) + df_vendas['customer_id'].astype(str) + df_vendas['Categoria'].astype(str)
#                 df_vendas = df_vendas.merge( df_users[['client_site_code','Region Name','Tipo_Cliente','5-9 Cxs','1-4 Cxs']] ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False))
#                 df_vendas = df_vendas.drop(columns=['client_site_code'])
#                 df_vendas['Gmv 5-9 Cxs'] = df_vendas['Gmv'].multiply(df_vendas['5-9 Cxs'] , axis=0)
#                 df_vendas['Gmv 1-4 Cxs'] = df_vendas['Gmv'].multiply(df_vendas['1-4 Cxs'] , axis=0)
#                 df_vendas = df_vendas.groupby( df_vendas['key']).agg({'Gmv':'sum', 'Quantity':'sum' , 'Pedidos': pd.Series.nunique , 'Gmv 5-9 Cxs':'sum','Gmv 1-4 Cxs':'sum' }).reset_index(drop= False)
#                 df_vendas = df_vendas[['key','Gmv','Pedidos','Quantity']]
#                 return df_vendas
            
#             df_vendas = df_vendass(df) 

#             def concorrencia(query_concorrencia, top_skus):
            
#                 # Df Concorrencia 

#                 df_concorrencia = query_concorrencia.copy()
#                 df_concorrencia = df_concorrencia[df_concorrencia['ean'].isin(top_skus)].drop(columns=['categoria','description','region_name'])
#                 price_cols = df_concorrencia.drop(columns = ['data','ean']).columns.to_list()
#                 df_concorrencia = df_concorrencia.merge(df_produtos  ,how ='left', left_on='ean', right_on='unit_ean_prod', suffixes=(False, False))
#                 #df_concorrencia = df_concorrencia[['data','ean','Unit_Description']] 

#                 cols_conco = df_concorrencia.drop(columns=['Unit_Description','Categoria']).columns.to_list()
#                 df_concorrencia['Produtos'] = df_concorrencia['Unit_Description'].astype(str)  + " - " + df_concorrencia['ean'].astype(str)
#                 df_concorrencia = df_concorrencia[['Produtos','Unit_Description'] + cols_conco]
#                 df_concorrencia = df_concorrencia.set_index('data')
                    

#                 price_cols = ['price_avg_concorrencia','price_guanabara','price_ceasa','price_mundial']
#                 price_cols = ['price_mundial','price_ceasa','price_torre_cia','price_nova_coqueiro_alimentos']
#                 price_cols = ['price_mundial','price_ceasa','price_torre_cia']

#                 #price_cols = ['price_mundial']
#                 #df_concorrencia = df_concorrencia[['data','ean','Unit_Description'] + price_cols] 

#                 df_concorrencia.iloc[:,4:] = df_concorrencia.iloc[:,4:].astype(float)
#                 df_concorrencia_final = df_concorrencia.copy()  
#                 df_concorrencia_final = df_concorrencia_final[  price_cols]

                
#                 df_concorrencia_dummies = pd.get_dummies(df_concorrencia['Produtos'] )
#                 df_concorrencia_dummies = df_concorrencia_dummies.astype(float)
#                 df_concorrencia_dummies = df_concorrencia_dummies.replace(0,np.nan)
                
                
#                 for k in range(1, df_concorrencia.shape[1] + 1): 
                
#                     if df_concorrencia.columns[k-1] in price_cols:
#                         print(df_concorrencia.columns[k-1])
#                         concorrente = df_concorrencia.columns[k-1][6:]
#                         df_concorrencia_prices = df_concorrencia_dummies.multiply(df_concorrencia['price_' + concorrente ], axis=0)     
                    
#                         df_concorrencia_prices.columns=[concorrente.title() + " " + str(df_concorrencia_prices.columns[k-1])  for k in range(1, df_concorrencia_prices.shape[1] + 1)]
#                         df_concorrencia_final =  df_concorrencia_final.merge(df_concorrencia_prices, left_index = True, right_index=True,how = "left") 
#                         df_concorrencia_final  = df_concorrencia_final.drop(columns = [ df_concorrencia.columns[k-1] ])

                
#                 df_concorrencia_final = df_concorrencia_final.astype(float) 
#                 #df_concorrencia_final = df_concorrencia_final.drop(columns=['ean','Produtos','Unit_Description','unit_ean_prod'])
#                 df_concorrencia_final  = df_concorrencia_final.interpolate().bfill() 
#                 df_concorrencia_final = df_concorrencia_final.interpolate(limit_direction='both')
#                 df_concorrencia_final =  df_concorrencia_final.dropna(axis='columns') 
#                 df_concorrencia_final = df_concorrencia_final.groupby( df_concorrencia_final.index).median()
                
#                 df_concorrencia_final.columns=[ "Price Concorrência " + str(df_concorrencia_final.columns[k-1]).title()  for k in range(1, df_concorrencia_final.shape[1] + 1)]
                                
        
#                 return df_concorrencia_final

#             df_concorrencia = concorrencia(query_concorrencia, top_skus)
        
#             def price_top_skus(df, top_skus):
        
#                 # Df Prices Top SKus 
#                 # Prices 
#                 df_price_categories = df.copy() 
#                 df_price_categories['Produtos'] = df_price_categories['Unit_Description'].astype(str) + " - " + df_price_categories['unit_ean'].astype(str)
#                 df_price_categories = df_price_categories[df_price_categories['unit_ean_prod'].isin(top_skus)]

#                 df_price_categories = df_price_categories[['DateHour','order_id','order_item_id','Data','customer_id','Categoria','unit_ean','Unit_Description','Produtos','region_id','store_id','price_managers','Original_Price','Price']]
#                 df_price_categories = df_price_categories.merge( df_users[['client_site_code','Region Name','pricing_group_id','size','Tipo_Cliente','tipo da loja','Não_Mercado','5-9 Cxs','1-4 Cxs']] ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False)).drop(columns=['client_site_code'])
#                 df_price_categories = df_price_categories[df_price_categories['Data'] >= pd.Timestamp('2021-01-01')]   
#             # df_price_categories = df_price_categories[df_price_categories['Data'] <= pd.Timestamp('2024-03-20')]   

#                 df_price_categories['Flag_Store'] =  np.where((df_price_categories['store_id'].str.startswith('Estoque')  ) , 'Estoque'  , '-'  )
#                 df_price_categories['Flag_Store'] =  np.where((df_price_categories['store_id'].str.startswith('Ofe')  ) , 'Ofertão'  , df_price_categories['Flag_Store']  )
#                 df_price_categories['Flag_Store'] =  np.where((df_price_categories['Flag_Store'] ==  '-'  ) , '3P'  , df_price_categories['Flag_Store']  )

#                 df_price_categories['order_item_id']= df_price_categories['order_item_id'].astype(np.int64).astype(str)
#                 #df_price_categories  = df_price_categories[df_price_categories['order_item_id']=='593']

#                 df_price_categories['unit_ean']= df_price_categories['unit_ean'].astype(np.int64).astype(str) 

                
#                 #df_price_categories['key_concorrencia'] = df_price_categories['Data'].astype(str)  +  df_price_categories['unit_ean'].astype(str) 
#                 #df_price_categories = df_price_categories.merge(   df_concorrencia  ,how ='left', left_on='key_concorrencia', right_on='key_concorrencia', suffixes=(False, False))
                
                
#                 # Prices 1-4 Cxs


#                 df_prices_1_4_cxs  = df_price_categories.copy()
#                 df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['Flag_Store']=='Estoque']
                
                
#                 df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['1-4 Cxs']==1]
#                 df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['pricing_group_id'] == 70]
#                 df_prices_1_4_cxs  = df_prices_1_4_cxs [df_prices_1_4_cxs['Não_Mercado'] == 0]
#                 df_prices_1_4_cxs  = pd.pivot_table(df_prices_1_4_cxs , values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
                
#                 df_prices_1_4_cxs.columns = df_prices_1_4_cxs .columns.droplevel(0)
#                 df_prices_1_4_cxs.columns = df_prices_1_4_cxs .columns.droplevel(0)
#                 df_prices_1_4_cxs  = df_prices_1_4_cxs .reset_index(drop = False).set_index('Data')

#                 df_prices_1_4_cxs  = pd.DataFrame(df_prices_1_4_cxs .to_records()).set_index('Data')
#                 df_prices_1_4_cxs  =  pd.DataFrame(df_prices_1_4_cxs .asfreq('D').index).set_index('Data').merge(df_prices_1_4_cxs , left_index = True, right_index=True,how = "left") 
#                 #df_prices_1_4_cxs  = df_prices_1_4_cxs .interpolate().bfill() 

#                 df_prices_1_4_cxs.columns=[ "Price BAU 1-4 Cxs - " + str(df_prices_1_4_cxs .columns[k-1]).title()  for k in range(1, df_prices_1_4_cxs .shape[1] + 1)]
                
                
#                 # Prices 5-9 Cxs

#                 df_prices_5_9_cxs = df_price_categories.copy()
#                 df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['Flag_Store']=='Estoque']
#                 df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['1-4 Cxs']==0]
#                 df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['pricing_group_id'] == 70]
#                 df_prices_5_9_cxs = df_prices_5_9_cxs[df_prices_5_9_cxs['Não_Mercado'] == 0]
#                 df_prices_5_9_cxs = pd.pivot_table(df_prices_5_9_cxs, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
#                 df_prices_5_9_cxs.columns = df_prices_5_9_cxs.columns.droplevel(0)
#                 df_prices_5_9_cxs.columns = df_prices_5_9_cxs.columns.droplevel(0)
#                 df_prices_5_9_cxs = df_prices_5_9_cxs.reset_index(drop = False).set_index('Data')
#                 df_prices_5_9_cxs = pd.DataFrame(df_prices_5_9_cxs.to_records()).set_index('Data')
                
#                 df_prices_5_9_cxs =  pd.DataFrame(df_prices_5_9_cxs.asfreq('D').index).set_index('Data').merge(df_prices_5_9_cxs, left_index = True, right_index=True,how = "left") 
#                 df_prices_5_9_cxs.columns=[ "Price BAU 5-9 Cxs " + str(df_prices_5_9_cxs.columns[k-1]).title()  for k in range(1, df_prices_5_9_cxs.shape[1] + 1)]
#                 #df_prices_5_9_cxs = df_prices_5_9_cxs.interpolate().bfill() 

                
#                 # Prices 1-4 Cxs Ofertão
                
#                 df_prices_1_4_cxs_ofertao = df_price_categories.copy()
#                 df_prices_1_4_cxs_ofertao= df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['Flag_Store']=='Estoque']
#                 df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['1-4 Cxs']==1]
#                 df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['pricing_group_id'] == 70]
#                 df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao[df_prices_1_4_cxs_ofertao['Não_Mercado'] == 0]
#                 df_prices_1_4_cxs_ofertao = pd.pivot_table(df_prices_1_4_cxs_ofertao, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
                
#                 df_prices_1_4_cxs_ofertao.columns = df_prices_1_4_cxs_ofertao.columns.droplevel(0)
#                 df_prices_1_4_cxs_ofertao.columns = df_prices_1_4_cxs_ofertao.columns.droplevel(0)
#                 df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao.reset_index(drop = False).set_index('Data')

#                 df_prices_1_4_cxs_ofertao = pd.DataFrame(df_prices_1_4_cxs_ofertao.to_records()).set_index('Data')
#                 df_prices_1_4_cxs_ofertao =  pd.DataFrame(df_prices_1_4_cxs_ofertao.asfreq('D').index).set_index('Data').merge(df_prices_1_4_cxs_ofertao, left_index = True, right_index=True,how = "left") 
#                 df_prices_1_4_cxs_ofertao.columns=[ "Price Ofertão 1-4 Cxs - " + str(df_prices_1_4_cxs_ofertao.columns[k-1]).title()  for k in range(1, df_prices_1_4_cxs_ofertao.shape[1] + 1)]
#                 #df_prices_1_4_cxs _ofertao = df_prices_1_4_cxs _ofertao.interpolate().bfill() 

#                 # Prices 5-9 Cxs

#                 df_prices_5_9_cxs_ofertao = df_price_categories.copy()
#                 df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['Flag_Store']=='Estoque']
#                 df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['1-4 Cxs']==0]
#                 df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['pricing_group_id'] == 70]
#                 df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao[df_prices_5_9_cxs_ofertao['Não_Mercado'] == 0]
#                 df_prices_5_9_cxs_ofertao = pd.pivot_table(df_prices_5_9_cxs_ofertao, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
#                 df_prices_5_9_cxs_ofertao.columns = df_prices_5_9_cxs_ofertao.columns.droplevel(0)
#                 df_prices_5_9_cxs_ofertao.columns = df_prices_5_9_cxs_ofertao.columns.droplevel(0)
#                 df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao.reset_index(drop = False).set_index('Data')
#                 df_prices_5_9_cxs_ofertao = pd.DataFrame(df_prices_5_9_cxs_ofertao.to_records()).set_index('Data')
#                 df_prices_5_9_cxs_ofertao =  pd.DataFrame(df_prices_5_9_cxs_ofertao.asfreq('D').index).set_index('Data').merge(df_prices_5_9_cxs_ofertao, left_index = True, right_index=True,how = "left") 
#                 df_prices_5_9_cxs_ofertao.columns=[ "Price Ofertão 5-9 Cxs " + str(df_prices_5_9_cxs_ofertao.columns[k-1]).title()  for k in range(1, df_prices_5_9_cxs_ofertao.shape[1] + 1)]
#                 #df_prices_5_9_cxs_ofertao = df_prices_5_8_cxs_mercado.interpolate().bfill() 


#                 df_price_mediana = df_price_categories.copy()
#                 df_price_mediana = pd.pivot_table(df_price_mediana, values=['Original_Price'], index=['Data'], columns=['Produtos'],aggfunc={ 'Original_Price': [ "median" ]})
#                 df_price_mediana = df_price_mediana.interpolate().bfill() 
#                 df_price_mediana =  df_price_mediana.interpolate(limit_direction='both')
#                 df_price_mediana.columns = df_price_mediana.columns.droplevel(0)
#                 df_price_mediana.columns = df_price_mediana.columns.droplevel(0)
#                 df_price_mediana = df_price_mediana.reset_index(drop = False).set_index('Data')
#                 df_price_mediana = pd.DataFrame(df_price_mediana.to_records()).set_index('Data')
#                 df_price_mediana.columns=[ "Price " +   str(df_price_mediana.columns[k-1]) for k in range(1, df_price_mediana.shape[1] + 1)]

#                 df_price_final = df_price_mediana.merge(df_prices_1_4_cxs, how = 'left', left_index = True, right_index=True)
#                 df_price_final = df_price_final.merge(df_prices_5_9_cxs, how = 'left', left_index = True, right_index=True) 
#                 df_price_final = df_price_final.merge(df_prices_1_4_cxs_ofertao, how = 'left', left_index = True, right_index=True) 
#                 df_price_final = df_price_final.merge(df_prices_5_9_cxs_ofertao, how = 'left', left_index = True, right_index=True) 
#                 df_price_final = df_price_final.astype(float)
#                 df_price_final = df_price_final.replace(np.nan,0)
                

#                 # Ajusta os preços de 1-4 e 5-9 para quem tava sem preço e ajusta ofertão  
                
#                 for sku in top_skus: 

#                     sku_cols = df_price_final.columns[df_price_final.columns.str.endswith(sku)].tolist()

#                 #  sku_cols_ofertao_1_4 = df_price_final.columns[df_price_final.columns.str.startswith('Price Ofertão 1-4 Cxs')].tolist()
#                 #  sku_cols_ofertao_5_9 = df_price_final.columns[df_price_final.columns.str.startswith('Price Ofertão 5-9 Cxs')].tolist()

                
#                 for col in range(0, len(sku_cols)):  
#                     # Acerta Ofertão para Preço Médio
#                     if col ==0: 
                    
#                         sku_cols_ofertao = df_price_final[sku_cols].columns[df_price_final[sku_cols].columns.str.startswith('Price Ofertão 1-4 Cxs')].tolist()    
                    
#                     if len(sku_cols_ofertao)>0:
#                         df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols_ofertao[0]] > 0  , df_price_final[sku_cols_ofertao[0]] ,  df_price_final[sku_cols[col]] )

#                     # Acerta Preços caso n tenha venda e acerta ofertão caso tenha nos 1-4 e nos 5-9        

#                     elif col >0: 
                
#                         tipo_sku = sku_cols[col][:13]

#                         if tipo_sku != 'Price Ofertão':
                    
#                             df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols[col]]  == 0  , df_price_final[sku_cols[0]]  ,  df_price_final[sku_cols[col]] )
                    
                    
#                             if tipo_sku[len(tipo_sku)-3:]  == '1-4': 
#                                 sku_cols_ofertao = df_price_final[sku_cols].columns[df_price_final[sku_cols].columns.str.startswith('Price Ofertão 1-4 Cxs')].tolist()    
                            
#                             if len(sku_cols_ofertao)>0:

                                    
#                                 df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols_ofertao[0]] > 0  , df_price_final[sku_cols_ofertao[0]] ,  df_price_final[sku_cols[col]] )
                                
#                                 #  print('aqui')
#                                 #  print(sku_cols_ofertao[0])

#                             elif tipo_sku[len(tipo_sku)-3:] == '5-9':

#                                 sku_cols_ofertao = df_price_final[sku_cols].columns[df_price_final[sku_cols].columns.str.startswith('Price Ofertão 5-9 Cxs')].tolist()    

#                             if len(sku_cols_ofertao)>0:       

#                                 df_price_final[sku_cols[col]] =  np.where(df_price_final[sku_cols_ofertao[0]]  > 0  , df_price_final[sku_cols_ofertao[0]]  ,  df_price_final[sku_cols[col]] )

                
#                 cols_df_prices_1_4_cxs = df_prices_1_4_cxs.columns.to_list()
#                 cols_df_prices_5_9_cxs = df_prices_5_9_cxs.columns.to_list() 
#                 cols_df_prices_1_4_cxs_ofertao = df_prices_1_4_cxs_ofertao.columns.to_list() 
#                 cols_df_prices_5_9_cxs_ofertao = df_prices_5_9_cxs_ofertao.columns.to_list() 
#                 cols_df_prices_mediana = df_price_mediana.columns.to_list() 
#                 df_price_final = df_price_final.drop(columns= cols_df_prices_1_4_cxs_ofertao + cols_df_prices_5_9_cxs_ofertao )
                

#                 return df_price_final 


#             df_price_top_skus = price_top_skus(df, top_skus)


#             def df_sku(df,top_skus):
#                 # DF SKUS 

#                 df_skus = df.copy()  
#                 #df_skus = df_skus[df_skus['Data'] > pd.Timestamp('2024-01-01')]  
#                 #df_skus[['Categoria','Gmv']].groupby(['Categoria']).sum().sort_values('Gmv' , ascending =  False)  
#                 df_skus['key'] = df_skus['Data'].astype(str) + df_skus['customer_id'].astype(str)
#                 df_skus = df_skus[df_skus['unit_ean_prod'].isin(top_skus)]
#                 df_skus['Produto'] = 'Gmv Categoria ' + df_skus['Categoria'].astype(str) + ' ' + df_skus['Unit_Description'].astype(str) + ' - ' +   df_skus['unit_ean_prod'].astype(str)
#                 df_skus = df_skus[['key','Produto','Gmv']]
#                 df_skus = pd.pivot_table(df_skus, values=['Gmv'], index=['key'] , columns=['Produto'],aggfunc={ 'Gmv': [ "sum" ]})
#                 df_skus.columns = df_skus .columns.droplevel(0)
#                 df_skus.columns = df_skus .columns.droplevel(0)
#                 df_skus  = df_skus .reset_index(drop = False).set_index('key')
#                 df_skus  = pd.DataFrame(df_skus .to_records()).set_index('key')
                

#                 for k in range(1, df_skus.shape[1] + 1):
                    
#                     df_skus['Positivação ' + df_skus.columns[k-1][3:]] = np.where(( df_skus.iloc[:,k-1:k]  > 0 )  ,   1  , 0  )
                

#                 df_skus = df_skus.replace(np.nan, 0 )
#                 df_skus = df_skus.reset_index(drop = False)
                
#                 return df_skus

            
#             df_skus = df_sku(df, top_skus)


#             def df_categorias(df, lista_categorias ):
                
#                 # Df Categoria

#                 df_categoria = df.copy()  
#                 #df_categoria = df_categoria[df_categoria['unit_ean_prod']== '7896079500151']
#                 #df_categoria = df_categoria[df_categoria['Data'] <  pd.Timestamp('2024-02-15')]   
#                 #df_categoria = df_categoria[df_categoria['Data'] >  pd.Timestamp('2024-02-07')]   

                
#                 #df_categoria = df_categoria[df_categoria['Data'] > pd.Timestamp('2024-01-01')]  
#                 #df_categoria[['Categoria','Gmv']].groupby(['Categoria']).sum().sort_values('Gmv' , ascending =  False)  
#                 df_categoria['key'] = df_categoria['Data'].astype(str) + df_categoria['customer_id'].astype(str)
#                 df_categoria['Pedidos'] = df_categoria['key'] 
#                 df_categoria['Positivação'] = df_categoria['Data'].astype(str) + df_categoria['customer_id'].astype(str) + df_categoria['Categoria'].astype(str)
#                 df_categoria = df_categoria[df_categoria['Categoria'].isin(lista_categorias)]
                
#                 df_categoria = df_categoria.groupby( ['key','Categoria']).agg({'Gmv':'sum', 'Quantity':'sum','Price':'mean', 'unit_ean_prod': 'nunique'  }).reset_index(drop= False)
#                 df_categoria = df_categoria.reset_index(drop = 'False')
#                 df_categoria= df_categoria.rename(columns = {'customer_id':'User','unit_ean_prod':'Produtos'})
#                 df_categoria_dummies = pd.get_dummies(df_categoria['Categoria'] ).astype(float) 
#                 df_categoria= pd.get_dummies(df_categoria, columns = ['Categoria'] , drop_first = False)


#                 df_categoria.iloc[:,1:] = df_categoria.iloc[:,1:].astype(float)
#                 df_calcula_categorias = df_categoria.copy() 
                 
                
#                 for i in range(1, df_categoria.iloc[:,5:].shape[1]+1): 
#                     col_categoria = df_categoria.columns[4+i]  
#                     df_calcula_categorias['Gmv ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Gmv']
#                     df_calcula_categorias['Positivação ' + col_categoria ] = df_categoria[col_categoria]  
#                     df_calcula_categorias['Quantity ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Quantity']
#                     df_calcula_categorias['Price ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Price']
#                     df_calcula_categorias['Top Produtos ' + col_categoria ] = df_categoria[col_categoria] * df_categoria['Produtos']
#                     df_calcula_categorias = df_calcula_categorias.drop(columns = [col_categoria])


                  

#                 df_categoria = df_categoria.merge( df_categoria_dummies ,how ='left', left_index= True, right_index=True, suffixes=(False, False))
                

#                 df_categoria = df_calcula_categorias.copy()
                
#                 cols_positivacao = df_categoria.columns[df_categoria.columns.str.startswith('Positivação')].tolist()
#                 cols_price = df_categoria.columns[df_categoria.columns.str.startswith('Price')].tolist()

#                 cols_top_produtos = df_categoria.columns[df_categoria.columns.str.startswith('Top Produtos')].tolist()


#                 cols_price.remove('Price')


#                 df_categoria_pos = df_categoria[['key'] + cols_positivacao].groupby('key').max()
#                 df_categoria_price= df_categoria[cols_price + ['key']].groupby('key').mean()
#                 df_categoria_gmv_qtd = df_categoria.drop(columns = ['Gmv','Quantity','Price'] + cols_positivacao + cols_price ).groupby('key').sum()
#                 df_categoria = df_categoria_gmv_qtd.merge( df_categoria_pos ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
#                 df_categoria = df_categoria.merge( df_skus ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
#                 #df_categoria = df_categoria.merge( df_categoria_price ,how ='left', left_on='key', right_on='key', suffixes=(False, False))


#                 df_categoria = df_categoria.reset_index(drop = False)
#                 df_categoria = df_categoria.replace(np.nan, 0)
#                 cols_positivacao = df_categoria.columns[df_categoria.columns.str.startswith('Positivação')].tolist()


#                 cols_gmv_categoria = df_categoria.columns[df_categoria.columns.str.startswith('Gmv')].tolist()

#                 return df_categoria
        
            
#             df_categoria = df_categorias(df, lista_categorias )

#             def df_groupeds(df, df_vendas, df_trafego , df_categoria, df_concorrencia , df_price_top_skus): 
        
#                 df_grouped = df.copy()  
#                 df_grouped['key'] = df_grouped['Data'].astype(str) + df_grouped['customer_id'].astype(str)
#                 df_grouped = df_grouped.rename(columns = {'customer_id':'User'})
#                 df_grouped = pd.concat([df_grouped[['Data','key','User']].groupby(['key','User']).max().reset_index(drop=False),df_trafego[['Data','key','User']].groupby(['key','User']).max().reset_index(drop=False)])
#                 df_grouped = df_grouped.groupby('key').max().reset_index(drop = False)
#                 df_grouped = df_grouped.merge( df_users[['client_site_code','Region Name','region_id']] ,how ='left', left_on='User', right_on='client_site_code', suffixes=(False, False))
#                 df_grouped = df_grouped.drop(columns=['client_site_code'])
#                 df_grouped = df_grouped.merge(df_trafego.drop(columns = ['Data','User']) ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
#                 df_grouped = df_grouped.merge( df_vendas ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
#                 df_grouped = df_grouped.merge( df_categoria ,how ='left', left_on='key', right_on='key', suffixes=(False, False))
#                 df_grouped['key_categoria'] = 1
#                 df_grouped = df_grouped.replace(np.nan, 0)

#                 df_grouped = df_grouped.drop(columns = ['key_categoria'])

#                 return df_grouped

            
#             df_grouped = df_groupeds(df, df_vendas, df_trafego, df_categoria , df_concorrencia , df_price_top_skus )


#             def df_modelos(df_grouped, df_users, df_price_top_skus , df_ofertao , size, data_inicio, data_fim):
                
#                 df_modelo = df_grouped.copy()  
#                 df_modelo = df_modelo.merge( df_users[['client_site_code','size_final']] ,how ='left', left_on='User', right_on='client_site_code', suffixes=(False, False)).drop(columns=['client_site_code'])
#                 df_modelo = df_modelo[df_modelo['size_final'] == size]
                
#                 df_modelo = df_modelo[df_modelo['Data']>=data_inicio]  
#                 df_modelo = df_modelo[df_modelo['Data']<data_fim] 
#                 df_modelo = df_modelo.set_index('Data')  
                
#                 df_modelo = df_modelo.drop(columns= ['key','User','Region Name','region_id','size_final'])
#                 df_modelo = df_modelo.replace(np.nan, 0)        
#                 df_modelo = df_modelo.groupby( df_modelo.index).sum()

#                 df_modelo = df_modelo.merge(df_price_top_skus, left_index = True, right_index=True,how = "left")  
        
                
#                 df_modelo = df_modelo.merge(df_concorrencia, left_index = True, right_index=True,how = "left") 

#                 cols_price_concorrencia = df_concorrencia.columns.to_list()
#                 cols_price_clubbi = df_price_top_skus.columns.to_list()
                

#                 # # # Métricas Trafego
                
#                 df_modelo['conversao_trafego'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego']  )
#                 df_modelo['conversao_trafego'] = df_modelo['conversao_trafego'].interpolate(limit_direction='both')

#                 df_modelo['conversao_trafego_search'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Search_Products'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego_Search_Products']  )
#                 df_modelo['conversao_trafego_search'] = df_modelo['conversao_trafego_search'].interpolate(limit_direction='both')


#                 df_modelo['conversao_trafego_add'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Add_To_Cart'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego_Add_To_Cart']  )
#                 df_modelo['conversao_trafego_add'] = df_modelo['conversao_trafego_add'].interpolate(limit_direction='both')

#                 df_modelo['conversao_trafego_checkout'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Checkout'] ) ,   np.nan  , df_modelo['Pedidos'] /df_modelo['Trafego_Checkout']  )
#                 df_modelo['conversao_trafego_checkout'] = df_modelo['conversao_trafego_add'].interpolate(limit_direction='both')


#                 df_modelo['Trafego'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego'] ) ,   df_modelo['Pedidos']/df_modelo['conversao_trafego'] , df_modelo['Trafego']  ).astype(int)
#                 df_modelo['Trafego_Search_Products'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Search_Products'] ) ,   df_modelo['Pedidos']/df_modelo['conversao_trafego_search']  , df_modelo['Trafego_Search_Products']  ).astype(int)
#                 df_modelo['Trafego_Add_To_Cart'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Add_To_Cart'] ) ,   df_modelo['Pedidos']/df_modelo['conversao_trafego_add']  , df_modelo['Trafego_Add_To_Cart']  ).astype(int)
#                 df_modelo['Trafego_Checkout'] =  np.where((df_modelo['Pedidos'] > df_modelo['Trafego_Checkout'] ) ,  df_modelo['Pedidos'] / df_modelo['conversao_trafego_checkout']  , df_modelo['Trafego_Checkout']  ).astype(int)

#                 df_modelo = df_modelo.drop(columns= ['conversao_trafego','conversao_trafego_search','conversao_trafego_add','conversao_trafego_checkout'])

                
#                 df_modelo =  pd.DataFrame(df_modelo.asfreq('D').index).set_index('Data').merge(df_modelo, left_index = True, right_index=True,how = "left") 

                
#                 # Df Futuro 
                
#                 future_dates = pd.date_range(df_modelo.index[-1], freq = 'D', periods = 2)
#                 future_dates = pd.DataFrame(future_dates, columns=['Data'])
#                 future_dates = future_dates.set_index('Data')
#                 future_dates['Gmv'] = np.nan
#                 future_dates = future_dates.iloc[1:,:]
                 
                
#                 # Df Modelo Final

#                 df_modelo = pd.concat([df_modelo, future_dates])  

                    
#                 cols_price_clubbi = df_price_top_skus.columns.to_list()

#                 df_modelo[cols_price_clubbi + cols_price_concorrencia]  = df_modelo[cols_price_clubbi + cols_price_concorrencia].interpolate().bfill() 
#                 df_modelo[cols_price_clubbi + cols_price_concorrencia] = df_modelo[cols_price_clubbi + cols_price_concorrencia].interpolate(limit_direction='both')
                
#              #   df_modelo = df_modelo.merge(df_eventos, left_index = True, right_index=True,how = "left") 


#                 df_modelo = df_modelo.merge(  df_ofertao  , how='left', left_index=True, right_index=True)   
#                 df_modelo = df_modelo.replace(np.nan,0)
                
#                 df_modelo = df_modelo.dropna()  
                
#                 df_modelo = df_modelo.sort_index(ascending=True)

#                 # df_modelo['Gmv_Shift'] = df_modelo['Gmv'].shift(periods=  70, freq="D")
#                 # df_modelo['Quantity_Shift'] = df_modelo['Quantity'].shift(periods=  70, freq="D")
#                 # df_modelo['Pedidos_Shift'] = df_modelo['Pedidos'].shift(periods=   70, freq="D")
#                 # df_modelo['Trafego_Shift'] = df_modelo['Trafego'].shift(periods=   70, freq="D")



#                 # df_modelo['Gmv'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,   df_modelo['Gmv_Shift']  ,df_modelo['Gmv'] )
#                 # df_modelo['Quantity'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Quantity_Shift']   , df_modelo['Quantity'] )
#                 # df_modelo['Pedidos'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Pedidos_Shift']  , df_modelo['Pedidos'] ) 
#                 # df_modelo['Trafego'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Trafego_Shift']  , df_modelo['Trafego'] ) 


#                 # df_modelo = df_modelo.drop(columns=['Gmv_Shift','Quantity_Shift','Pedidos_Shift','Trafego_Shift'])
                



#                 # Não Me xer Nessa Parte * Atenção 
#                 df_delta_prices =  df_modelo[cols_price_clubbi + cols_price_concorrencia]
#                 #df_delta_prices = df_delta_prices[df_delta_prices.columns[df_delta_prices.columns.str.endswith('7891107101621')].tolist()]   # Aqui pode # Filtrar
#                 df_delta_final = df_delta_prices.copy() 
#                 df_delta_final['Flag'] = 1  
#                 df_delta_final = df_delta_final[['Flag']]
                

#                 # Não Me xer Nessa Parte * Atenção 
                
#                 # Lags, Windowns, Deltas, Preços Clubbi e Concorrentes 
                
#                 for k in range(0, len(top_skus)):  
#                     # Loop SKU
                    
#                     prod = top_skus[k]  
#                     cols_produto = df_delta_prices.columns[df_delta_prices.columns.str.endswith(prod)].tolist()
                    
#                     df_produto_prices = df_delta_prices[cols_produto]
#                     cols_clubbi = df_produto_prices.columns[df_produto_prices.columns.str.startswith('Price')].tolist() 
                    
                    
#                     cols_prices_concorrencia = df_produto_prices.drop(columns=cols_clubbi).columns.tolist()
#                     flag_conc = len(cols_prices_concorrencia)


                    
#                     cols_positivacao = df_modelo.columns[df_modelo.columns.str.endswith(prod)].tolist()
#                     cols_positivacao = df_modelo[cols_positivacao].columns[df_modelo[cols_positivacao].columns.str.startswith('Positivação')].tolist()

                    
#                     df_modelo['% Share ' + cols_positivacao[0]] =  np.where((df_modelo['Pedidos'] == 0 ) , 0 , df_modelo[cols_positivacao[0]]/ df_modelo['Pedidos']  )
                    
                        

#                     for c in range(0, len(cols_clubbi)):    

#                     #    price_clubbi =  cols_clubbi[c][:17][6:]           
#                         price_clubbi =  cols_clubbi[c]
                    
#                         lag_list = [1,5,7,14,21,28]
                        
#                         # Lags e Delta Lag 
                        
#                         for lag in lag_list:

#                             # Lag  
#                             delta_clubbi = 'Lag ' + str(lag) +  ' ' + price_clubbi 
#                             df_delta_final[delta_clubbi] = df_delta_prices[cols_clubbi[c]].shift(periods=  lag, freq="D") 
#                             # Delta Lag
#                             delta_clubbi = 'Delta Lag ' + str(lag) +  ' ' +  price_clubbi 
#                             df_delta_final[delta_clubbi] = (df_delta_prices[cols_clubbi[c]]/ df_delta_prices[cols_clubbi[c]].shift(periods=  lag, freq="D"))-1


#                             # delta_clubbi = 'Delta Lag 1 '   +  price_clubbi 

#                             # delta_clubbi_percent = 'Delta Wind 1 Redução 0-5% ' + price_clubbi       
#                             # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi] >-0.20) &  (df_delta_final[delta_clubbi]<0) , 1 , 0 )

#                             # delta_clubbi_percent = 'Delta Wind 1 Redução 0,5% ' + price_clubbi       
#                             # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi]<-0.05) , 1 , 0 )




#                         # Média dos Lags 
                        
#                         delta_clubbi = 'Lag- Mean 7/14 ' + price_clubbi      
#                         df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi ]  +  df_delta_final['Lag 14 ' + price_clubbi ])/2

#                         delta_clubbi = 'Lag- Mean 7/14/21 ' + price_clubbi  
#                         df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi ]  +  df_delta_final['Lag 14 ' + price_clubbi ] +  df_delta_final['Lag 21 ' +  price_clubbi ]      )/3
                        
#                         delta_clubbi = 'Lag- Mean 7/14/21/28 ' +  price_clubbi     
#                         df_delta_final[delta_clubbi] = (df_delta_final['Lag 7 ' + price_clubbi]  +  df_delta_final['Lag 14 ' + price_clubbi ] +  df_delta_final['Lag 21 ' +  price_clubbi ]  +  df_delta_final['Lag 28 ' + price_clubbi ]  )/4 

#                         # Média Windown 
#                         delta_clubbi = 'Wind 2D ' + price_clubbi     
#                         df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(2).mean()
#                         delta_clubbi = 'Wind 5D ' + price_clubbi     
#                         df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(5).mean()
#                         delta_clubbi = 'Wind 7D ' + price_clubbi     
#                         df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(7).mean()
#                         delta_clubbi = 'Wind 14D ' + price_clubbi     
#                         df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(14).mean()
#                         delta_clubbi = 'Wind 21D ' + price_clubbi     
#                         df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(21).mean()
#                         delta_clubbi = 'Wind 28D ' + price_clubbi     
#                         df_delta_final[delta_clubbi] = df_delta_prices[price_clubbi].shift(periods=  1, freq="D").rolling(28).mean()
                        
#                         # Delta Média Lags  
#                         delta_clubbi_var = 'Lag- Mean 7/14 ' + price_clubbi         
#                         delta_clubbi = 'Delta Lag- Mean 7/14 ' + price_clubbi       
#                         df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1 




#                         # delta_clubbi = 'Delta Lag- Mean 7/14/21 ' + price_clubbi       
#                         # df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1


#                         # delta_clubbi = 'Delta Lag- Mean 7/14/21/28 ' + price_clubbi       
#                         # df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

                        

#                         # Delta Média Wind
#                         delta_clubbi_var = 'Wind 2D ' + price_clubbi         
#                         delta_clubbi = 'Delta Wind 2D ' + price_clubbi       
#                         df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

#                         delta_clubbi_var = 'Wind 5D ' + price_clubbi         
#                         delta_clubbi = 'Delta Wind 5D ' + price_clubbi       
#                         df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

#                         # delta_clubbi_var = 'Wind 7D ' + price_clubbi         
#                         # delta_clubbi = 'Delta Wind 7D ' + price_clubbi       
#                     #    df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

#                         delta_clubbi_var = 'Wind 14D ' + price_clubbi         
#                         delta_clubbi = 'Delta Wind 14D ' + price_clubbi       
#                         df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

                        

#                         delta_clubbi_var = 'Wind 21D ' + price_clubbi         
#                         delta_clubbi = 'Delta Wind 21D ' + price_clubbi       
#                         df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1

                        

#                         delta_clubbi_var = 'Wind 28D ' + price_clubbi         
#                         delta_clubbi = 'Delta Wind 28D ' + price_clubbi       
#                         df_delta_final[delta_clubbi] = (df_delta_prices[price_clubbi]/df_delta_final[delta_clubbi_var])-1
                    

#                         # delta_clubbi_percent = 'Delta Wind 28D Redução 0-5% ' + price_clubbi       
#                         # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi] >-0.20) &  (df_delta_final[delta_clubbi]<0) , 1 , 0 )

#                         # delta_clubbi_percent = 'Delta Wind 28D Redução 0,5% ' + price_clubbi       
#                         # df_delta_final[delta_clubbi_percent] = np.where((df_delta_final[delta_clubbi]<-0.05) , 1 , 0 )

                
#                     #  df_delta_final[delta] =   df_delta_prices.apply(lambda row: ((row[cols_clubbi[z]] /row[concorrente_var])-1), axis=1) 
#                     #  df_delta_final[delta_clubbi] =   df_delta_prices.apply(lambda row: ((row[cols_clubbi[c]].shift (periods= 1  , freq="D") )), axis=1) 

#                     # Checa se tem concorrente 

#                     if flag_conc >0:
                                
#                         df_conc = df_produto_prices[cols_prices_concorrencia]  
#                 # Se tem concorrente loop os concorrentes 

#                         for j in range(1,   df_conc.shape[1] +1):
#                         # Loop Concorrente 
                            
#                             concorrente_var = df_delta_prices[cols_prices_concorrencia].columns[j-1] 
#                             for z in range(0, len(cols_clubbi)):    

#                         #        preco_clubbi =  cols_clubbi[z][:17][6:]         
#                                 preco_clubbi =  cols_clubbi[z][:17][6:] 
#                                 delta = 'Delta Concorrencia ' +  preco_clubbi + ' ' + concorrente_var  
#                                 df_delta_final[delta] =   df_delta_prices.apply(lambda row: ((row[cols_clubbi[z]] /row[concorrente_var])-1), axis=1)

#                         # Lags, Windowns, Deltas, Preços Clubbi e Concorrentes 
                            
                
#                 df_delta_final = df_delta_final.drop(columns=['Flag']) 



#                 #df_modelo = df_modelo.drop(columns =  cols_price_concorrencia).merge(df_delta_final[delta_wind + delta_lag], left_index = True, right_index=True,how = "left") 
#                 df_modelo = df_modelo.merge(df_delta_final ,  left_index = True, right_index=True,how = "left") 

         

#                 return df_modelo

            
#             df_modelo = df_modelos(df_grouped, df_users, df_price_top_skus, df_ofertao, size, data_inicio, data_fim)


#             def df_modelo_posit_conversao(df_modelo):
                    
#                 columns_positivacao = df_modelo.filter(regex='Positivação').columns.to_list() 
#                 columns_positivacao = [i for i in columns_positivacao if i.find('% Share' ,0)<0 ]   

#                 for i in columns_positivacao:
                    
#                     posit_column = '% ' + i  
#                     conversion_column = '% Conversão ' +  i[12:].lstrip()

#                     df_modelo[posit_column] = df_modelo[i]/df_modelo['Positivação']
#                     df_modelo[conversion_column] = df_modelo[i]/df_modelo['Trafego'] 


#                 return df_modelo
            
#             df_modelo = df_modelo.rename(columns = {'Pedidos': 'Positivação'})
#             df_modelo_final = df_modelo_posit_conversao(df_modelo)



#             return df_modelo_final
        

 
            
#         def listas_modelo(df_modelo):
            
#             # Colunas BAU 

#             columns_trafego = df_modelo.filter(regex='Trafego').columns.to_list()
#             columns_gmv = df_modelo.filter(regex='Gmv').columns.to_list()
#             columns_conversao = df_modelo.filter(regex='% Conversão').columns.to_list() 
#             columns_positivacao = df_modelo.filter(regex='Positivação').columns.to_list() 
#             columns_positivacao = [i for i in columns_positivacao if i.find('% Share' ,0)<0 ]   
#             columns_positivacao_share = [i for i in df_modelo.filter(regex='Positivação').columns.to_list()  if i.find('% Share' ,0)>=0 ]   
#             columns_qtd  = df_modelo.filter(regex='Quantity').columns.to_list()
#          #   columns_pedidos = df_modelo.filter(regex='Pedidos').columns.to_list() 
#             columns_ofertao = df_modelo.filter(regex='Ofertão').columns.to_list()
#             columns_ofertao_weekday = [i for i in  columns_ofertao  if i.find('weekday' ,0)>=0 ]
#             columns_ofertao = [i for i in  columns_ofertao  if i.find('weekday' ,0)<0 ]   

#             # Prices Concorrencia 

#             columns_price_concorrencia = df_modelo.filter(regex='Price Concorrência').columns.to_list()  
#             columns_price_concorrencia_lag = [i for i in columns_price_concorrencia if i.find('Lag' ,0)==0 and i.find('Mean' ,0)<=0 ] 
#             columns_price_concorrencia_lag_delta = [i for i in columns_price_concorrencia if i.find('Lag' ,0)>=0 and i.find('Delta' ,0)>=0 and i.find('Mean' ,0)<=0 ] 
#             columns_price_concorrencia_lag_mean = [i for i in columns_price_concorrencia if i.find('Lag' ,0)==0 and i.find('Mean' ,0)>=0 ]
#             columns_price_concorrencia_lag_delta_mean = [i for i in columns_price_concorrencia if i.find('Lag' ,0)>=0  and i.find('Delta' ,0)>=0  and i.find('Mean' ,0)>=0 ]
#             columns_price_concorrencia = [i for i in columns_price_concorrencia if i.find('Price' ,0)==0 ]
        

#             # Prices Clubbi 
        
#             columns_price_clubbi = df_modelo.drop(columns = columns_price_concorrencia).filter(regex='Price').columns.to_list() 
#             columns_price_clubbi_geral = [i for i in columns_price_clubbi if i.find('Lag' ,0)<0 and  i.find('Wind' ,0)<0 and i.find('Mean' ,0)<=0 ] 
#             columns_price_clubbi_geral_1_4_cxs = [i for i in columns_price_clubbi_geral if i.find('1-4' ,0)>=0 ] 
#             columns_price_clubbi_geral_5_4_cxs = [i for i in columns_price_clubbi_geral if i.find('5-9' ,0)>=0 ] 

#             # Price Lags Clubbi 

#             columns_price_clubbi_lag = [i for i in columns_price_clubbi if i.find('Lag' ,0)==0 and i.find('Mean' ,0)<=0 ] 
#             columns_price_clubbi_lag_1_4_cxs = [i for i in columns_price_clubbi_lag if i.find('1-4' ,0)>=0 ] 
#             columns_price_clubbi_lag_5_4_cxs = [i for i in columns_price_clubbi_lag if i.find('5-9' ,0)>=0 ] 
#             columns_price_clubbi_lag_geral =  [i for i in columns_price_clubbi_lag if i.find('1-4' ,0)<=0]
#             columns_price_clubbi_lag_geral =  [i for i in columns_price_clubbi_lag_geral  if i.find('5-9' ,0)<=0]


#             # Price Lags Clubbi Delta
            
#             columns_price_clubbi_lag_delta = [i for i in columns_price_clubbi if i.find('Lag' ,0)>=0 and i.find('Delta' ,0)>=0 and i.find('Mean' ,0)<=0 ] 
#             columns_price_clubbi_lag_delta_1_4_cxs = [i for i in columns_price_clubbi_lag_delta if i.find('1-4' ,0)>=0 ] 
#             columns_price_clubbi_lag_delta_5_4_cxs = [i for i in columns_price_clubbi_lag_delta if i.find('5-9' ,0)>=0 ] 
#             columns_price_clubbi_lag_delta_geral =  [i for i in columns_price_clubbi_lag_delta if i.find('1-4' ,0)<=0]
#             columns_price_clubbi_lag_delta_geral =  [i for i in columns_price_clubbi_lag_delta_geral  if i.find('5-9' ,0)<=0]


#             # Price Lag Clubbi Means 
            
#             columns_price_clubbi_lag_mean = [i for i in columns_price_clubbi if i.find('Lag' ,0)==0 and i.find('Mean' ,0)>=0 ]
#             columns_price_clubbi_lag_mean_1_4_cxs = [i for i in columns_price_clubbi_lag_mean if i.find('1-4' ,0)>=0 ] 
#             columns_price_clubbi_lag_mean_5_4_cxs = [i for i in columns_price_clubbi_lag_mean if i.find('5-9' ,0)>=0 ] 
#             columns_price_clubbi_lag_mean_geral =  [i for i in columns_price_clubbi_lag_mean if i.find('1-4' ,0)<=0]
#             columns_price_clubbi_lag_mean_geral =  [i for i in columns_price_clubbi_lag_mean_geral if i.find('5-9' ,0)<=0]

#             # Price Lag Means Delta 

#             columns_price_clubbi_lag_delta_mean = [i for i in columns_price_clubbi if i.find('Lag' ,0)>=0  and i.find('Delta' ,0)>=0  and i.find('Mean' ,0)>=0 ]
#             columns_price_clubbi_lag_delta_mean_1_4_cxs = [i for i in columns_price_clubbi_lag_delta_mean if i.find('1-4' ,0)>=0 ] 
#             columns_price_clubbi_lag_delta_mean_5_4_cxs = [i for i in columns_price_clubbi_lag_delta_mean if i.find('5-9' ,0)>=0 ] 
#             columns_price_clubbi_lag_delta_mean_geral =  [i for i in columns_price_clubbi_lag_delta_mean if i.find('1-4' ,0)<=0]
#             columns_price_clubbi_lag_delta_mean_geral =  [i for i in columns_price_clubbi_lag_delta_mean_geral if i.find('5-9' ,0)<=0]
            

#             # Price Wind Delta Mean

#             columns_price_clubbi_wind_delta_mean = [i for i in columns_price_clubbi if i.find('Wind' ,0)>=0  and i.find('Delta' ,0)>=0   ]
#             columns_price_clubbi_wind_delta_mean_1_4_cxs = [i for i in columns_price_clubbi_wind_delta_mean if i.find('1-4' ,0)>=0 ] 
#             columns_price_clubbi_wind_delta_mean_5_4_cxs = [i for i in columns_price_clubbi_wind_delta_mean if i.find('5-9' ,0)>=0 ] 
#             columns_price_clubbi_wind_delta_mean_geral =  [i for i in columns_price_clubbi_wind_delta_mean if i.find('1-4' ,0)<=0]
#             columns_price_clubbi_wind_delta_mean_geral =  [i for i in columns_price_clubbi_wind_delta_mean_geral if i.find('5-9' ,0)<=0]


        
#             data = {
                
                
#                     # Colunas BAU 

#                     "columns_trafego": columns_trafego, 
#                     "columns_gmv": columns_gmv, 
#                     "columns_positivacao": columns_positivacao, 
#                     "columns_positivacao_share": columns_positivacao_share, 
#                     "columns_qtd": columns_qtd, 
#                    # "columns_pedidos": columns_pedidos, 
#                     "columns_ofertao": columns_ofertao, 
#                     "columns_ofertao_weekday": columns_ofertao_weekday,  

#                     # Prices Concorrencia 

#                     "columns_price_concorrencia": columns_price_concorrencia, 
#                     "columns_price_concorrencia_lag": columns_price_concorrencia_lag, 
#                     "columns_price_concorrencia_lag_delta": columns_price_concorrencia_lag_delta, 
#                     "columns_price_concorrencia_lag_mean": columns_price_concorrencia_lag_mean, 
#                     "columns_price_concorrencia_lag_delta_mean": columns_price_concorrencia_lag_delta_mean, 


#                     # Prices Clubbi  

#                     "columns_price_clubbi_geral": columns_price_clubbi_geral, 
#                     "columns_price_clubbi_geral_1_4_cxs": columns_price_clubbi_geral_1_4_cxs, 
#                     "columns_price_clubbi_geral_5_4_cxs": columns_price_clubbi_geral_5_4_cxs, 

#                     # Prices Lag Clubbi  

#                     "columns_price_clubbi_lag": columns_price_clubbi_lag, 
#                     "columns_price_clubbi_lag_1_4_cxs": columns_price_clubbi_lag_1_4_cxs, 
#                     "columns_price_clubbi_lag_5_4_cxs": columns_price_clubbi_lag_5_4_cxs, 
#                     "columns_price_clubbi_lag_geral": columns_price_clubbi_lag_geral, 

#                     # Price Lags Clubbi 

#                     "columns_price_clubbi_lag": columns_price_clubbi_lag, 
#                     "columns_price_clubbi_lag_1_4_cxs": columns_price_clubbi_lag_1_4_cxs, 
#                     "columns_price_clubbi_lag_5_4_cxs": columns_price_clubbi_lag_5_4_cxs, 
#                     "columns_price_clubbi_lag_geral": columns_price_clubbi_lag_geral,  


#                     # Price Lags Clubbi Delta

#                     "columns_price_clubbi_lag_delta": columns_price_clubbi_lag_delta,  
#                     "columns_price_clubbi_lag_delta_1_4_cxs": columns_price_clubbi_lag_delta_1_4_cxs,  
#                     "columns_price_clubbi_lag_delta_5_4_cxs": columns_price_clubbi_lag_delta_5_4_cxs,  
#                     "columns_price_clubbi_lag_delta_geral": columns_price_clubbi_lag_delta_geral,  


#                     # Price Lag Clubbi Means 
        

#                     "columns_price_clubbi_lag_mean": columns_price_clubbi_lag_mean,  
#                     "columns_price_clubbi_lag_delta_mean_1_4_cxs": columns_price_clubbi_lag_delta_mean_1_4_cxs,  
#                     "columns_price_clubbi_lag_delta_mean_5_4_cxs": columns_price_clubbi_lag_delta_mean_5_4_cxs,  
#                     "columns_price_clubbi_lag_delta_mean_geral": columns_price_clubbi_lag_delta_mean_geral,  


#                     # Price Lag Means Delta
                    
#                     "columns_price_clubbi_lag_delta_mean": columns_price_clubbi_lag_delta_mean,  
#                     "columns_price_clubbi_wind_delta_mean_1_4_cxs": columns_price_clubbi_wind_delta_mean_1_4_cxs,  
#                     "columns_price_clubbi_wind_delta_mean_5_4_cxs": columns_price_clubbi_wind_delta_mean_5_4_cxs,  
#                     "columns_price_clubbi_wind_delta_mean_geral": columns_price_clubbi_wind_delta_mean_geral,  
        

#             }



#             return data
 

 
#                  # AOV
#                     # for i in listas['columns_gmv']:
                        
#                     #     if i!= 'Gmv':

#                     #         share_gmv_column = '% Gmv Categoria ' + i[13:].lstrip() + '/Gmv Total'   
#                     #         aov_column = 'AOV ' +  i[13:].lstrip()

#                     #         df_modelo_pipe[share_gmv_column] = df_modelo_pipe[i]/df_modelo_pipe['Gmv']

                    
#                     #         col_modelo = i[13:].lstrip()

#                     #         if col_modelo[0:1] == "_":
                    
#                     #             df_modelo_pipe['Aov ' + col_modelo[1:]] = df_modelo_pipe[i]/df_modelo_pipe['Positivação Categoria_' + col_modelo[1:]]            
#                     #             df_modelo_pipe['Aov ' + col_modelo[1:]] = df_modelo_pipe['Aov ' + col_modelo[1:]].replace(np.nan,0)
#                     #         else:
#                     #             df_modelo_pipe['Aov ' + col_modelo] = df_modelo_pipe[i]/df_modelo_pipe['Positivação  Categoria ' + col_modelo]            
#                     #             df_modelo_pipe['Aov ' + col_modelo]  = df_modelo_pipe['Aov ' + col_modelo].replace(np.nan,0)
 
 
 

# # %% Base Modelo 
 

#         with st.form(key = "form_modelos"):


#             col0, col1, col2    = st.columns(3)
            

#             with col0:

#                 categorias_forecasting =  st.multiselect('Categoria', categoria_list, default =  'Açúcar e Adoçante')[0]  


#             with col1:

#                 regional_modelo =  st.multiselect('Regional', regional_list, default =  ['RJC'])[0]  
                
#             with col2:

#                 size_modelo =  st.multiselect('Size', size_list, default =  ['1-4 Cxs'])[0]
                

#             submit_buttons = st.form_submit_button(label = "Submit")



#         with st.expander('Base', expanded= False):


#             @st.cache_resource( ttl = 45000)
#             def df_modelos(): 
                    
#                 df_modelo = pipe_df_modelos(df_order_previsao, df_ofertao_inicial, df_trafego_previsao , df_produtos_previsao , df_users_previsao,  query_concorrencia , regional_modelo, size_modelo , [categorias_forecasting], 'ean', '2022-01-01',  date.today().strftime('%Y-%m-%d'))

#                 return df_modelo 
        

#             df_modelo = df_modelos()  
#             listas = listas_modelo(df_modelo)
            
#             df_modelo   


    
#         columns_gmv = listas['columns_gmv']
#         columns_positivacao = listas['columns_positivacao'] 
#         columns_preco = listas['columns_price_clubbi_geral_1_4_cxs'] 
#         columns_preco_concorrente = listas['columns_price_concorrencia']  
#         columns_ofertao = listas['columns_ofertao'] 

#         columns_ofertao = [i for i in columns_ofertao if i.find( regional_modelo ,0)>=0 ]
#         columns_ofertao_size = [i for i in columns_ofertao if i.find( size_modelo ,0)>=0 ]


#         sku_list = df_modelo[columns_gmv].filter(regex='-').columns.to_list() 

#         sku_list = [ i[len(i)-13:] for i in sku_list if i ] + ['categoria']

                

# # %% Feature Modelo 


#         # col = st.columns((3, 6), gap='medium')

    
#         with st.expander('Feature Selection', expanded= False):


#             with st.form(key = "features_modelo"):


#                 col = st.columns((1.2, 9), gap='medium')
                    
#                 with col[0]: 
                    
#                     metricas_forecasting = st.radio("Métrica Forecast", ['Gmv', 'Positivação','Trafego'] )  

#                 with col[1]:

#                     col , col1, col2   = st.columns(3)


#                     with col:

                        
#                         sku_forecasting = st.multiselect("Forecast", sku_list , default = '7891910000197' )  

                    
#                     with col1: 

#                         columns_gmv_modelo = st.multiselect("Gmv Columns", columns_gmv , default =   df_modelo[columns_gmv].filter(regex=sku_forecasting[0]).columns.to_list()  )  

                    
#                     with col2:

#                         columns_positivacao = st.multiselect("Positivação Columns", columns_positivacao , default = df_modelo[columns_positivacao].filter(regex=sku_forecasting[0]).columns.to_list() )           
            

#                     col3,col4,col5   = st.columns(3)
        
#                     with col3:
                    
#                         columns_ofertao_modelo = st.multiselect("Ofertão Columns", columns_ofertao  , default = df_modelo[columns_ofertao_size].filter(regex=sku_forecasting[0]).columns.to_list() )           
                    

#                     with col4:
        
#                         columns_preco_modelo = st.multiselect("Preço Columns", columns_preco ,  default = df_modelo[columns_preco].filter(regex=sku_forecasting[0]).columns.to_list() )    


#                     with col5: 

#                         columns_preco_concorrente = st.multiselect("Preço Concorrente Columns", columns_preco_concorrente ,  default = df_modelo[columns_preco_concorrente].filter(regex=sku_forecasting[0]).columns.to_list() )    
        

#                 submit_buttonss = st.form_submit_button(label = "Submit")

        
#             var_forecasting = [i for i in df_modelo.columns.to_list() if i.find( metricas_forecasting ,0)==0 ]
#             var_forecasting = [i for i in var_forecasting if i.find( sku_forecasting[0] ,0)>=0 ]
            
#             df_modelo_final = df_modelo.copy()
#             df_modelo_final = df_modelo_final[ columns_gmv_modelo + columns_positivacao + columns_ofertao_modelo + columns_preco_modelo  + columns_preco_concorrente]
#             df_modelo_final


#             # col = st.columns((4, 6), gap='medium')

#             # with col[0]:
                
#             #     gmv_cols, posit_cols, col3   = st.columns(3)
                
#             #     with gmv_cols:

                    
#             #         columns_gmv_modelo = st.multiselect("Gmv Columns", columns_gmv )  

#             #     with posit_cols:
                        
#             #         columns_positivacao = st.multiselect("Positivação Columns", columns_positivacao )           



# # %% Etl Transform  


#     #  col = st.columns((3, 6), gap='medium')


#         # df_modelo['Gmv_Shift'] = df_modelo['Gmv'].shift(periods=  70, freq="D")
#         # df_modelo['Quantity_Shift'] = df_modelo['Quantity'].shift(periods=  70, freq="D")
#         # df_modelo['Pedidos_Shift'] = df_modelo['Pedidos'].shift(periods=   70, freq="D")
#         # df_modelo['Trafego_Shift'] = df_modelo['Trafego'].shift(periods=   70, freq="D")



#         # df_modelo['Gmv'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,   df_modelo['Gmv_Shift']  ,df_modelo['Gmv'] )
#         # df_modelo['Quantity'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Quantity_Shift']   , df_modelo['Quantity'] )
#         # df_modelo['Pedidos'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Pedidos_Shift']  , df_modelo['Pedidos'] ) 
#         # df_modelo['Trafego'] = np.where((df_modelo['Evento_Carnaval'] == 1) ,  df_modelo['Trafego_Shift']  , df_modelo['Trafego'] ) 


#         drop_outliers_date_default = []


#         if var_forecasting[0] == 'Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197':

            
#             drop_outliers_date_default.append({ 'Trafego 1-4 Cxs':'2024-03-28'})
#             drop_outliers_date_default.append({ 'Trafego 1-4 Cxs':'2024-03-29'})
#             drop_outliers_date_default.append({ 'Restr Vol.':'2024-04-01'})
#             drop_outliers_date_default.append({ 'Restr Vol.':'2024-04-05'})
#             drop_outliers_date_default.append({ 'Venda':'2024-04-16'})


#             drop_outliers_dates_default = []
#             drop_outliers_dates_default.append(pd.to_datetime('2024-03-28'))
#             drop_outliers_dates_default.append(pd.to_datetime('2024-03-29'))
#             drop_outliers_dates_default.append(pd.to_datetime('2024-04-01'))
#             drop_outliers_dates_default.append(pd.to_datetime('2024-04-05'))
#             drop_outliers_dates_default.append(pd.to_datetime('2024-04-16'))

#         if len(drop_outliers_date_default) !=0:

#             df_date_outliers = pd.DataFrame(drop_outliers_date_default)
            
#             date_index = pd.DataFrame(df_date_outliers.values.flatten())  

#             date_index[0] = pd.to_datetime(date_index[0]) 
#             date_index = date_index.dropna().reset_index()
#             date_index = date_index.groupby(0).max()
#             date_index = date_index[[]].reset_index(drop = False)   
#             date_index = date_index.rename(columns = {0:'Date'}) 
            
#             df_date_outliers = date_index.merge(  df_date_outliers, how='left', left_index=True, right_index=True)   
            
#             df_date_outliers = df_date_outliers.set_index('Date')
            
#             df_date_outliers[df_date_outliers.columns.to_list()] =  np.where(df_date_outliers[df_date_outliers.columns.to_list()].isna(), 0, 1)  
#             cols = df_date_outliers.columns.to_list()
#             df_date_outliers['Outlier'] = 1
#             df_date_outliers = df_date_outliers[['Outlier'] + cols]
            
#             df_modelo_final = df_modelo_final.merge(  df_date_outliers[['Outlier']], how='left', left_index=True, right_index=True)  
#             df_modelo_final['Outlier'] = df_modelo_final['Outlier'].replace(np.nan,0) 
#            # df_modelo_final[var_forecasting[0] + ' V0'] = df_modelo_final[var_forecasting[0]] 
#            # df_modelo_final[var_forecasting[0]] = np.where( df_modelo_final['Outlier']>0  , df_modelo_final[var_forecasting[0]].shift(periods=  7, freq="D") ,  df_modelo_final[var_forecasting[0]])



    
#         with st.expander('ETL Transformation', expanded= False):


#             with st.form(key = "pipeline_modelo"):
        
                
                    
#                 day_lag = [ "1D","2D","5D","6D","7D","12D","14D","21D","28D","35D","84D" ]
#                 day_wind = [ "1D","2D","5D","7D","12D","14D","21D","28D","35D","84D"  ] 
                
                
#                 day_lag_modelo =  st.multiselect('Lag Days', day_lag, default = ["14D","21D","28D"])
                        
#                 day_wind_modelo =  st.multiselect('Wind Days', day_wind, default = ["14D","21D","28D"])
                        

#                 not_scaled = columns_ofertao_modelo + [var_forecasting]    
#                 not_lag = columns_preco_modelo + columns_preco_concorrente

            

#                 lag_features =   st.multiselect('Lag/Wind Features',  columns_gmv_modelo + columns_positivacao + columns_preco + columns_preco_concorrente , default = columns_gmv_modelo + columns_positivacao )
                        
                
#                 submit_buttons_pipe = st.form_submit_button(label = "Submit")


                
#                 pipe_date = Pipeline([

#                     ("datetime_features",DatetimeFeatures(variables = "index",features_to_extract = ["month","week","day_of_week","day_of_month","weekend"]) ), 
#                     ("Periodic",CyclicalFeatures(variables = ["month","week","day_of_week", "day_of_month"],drop_original=False,)),
        
#                 ]) 

                
#                 st.text('Date Features') 
#                 cols_pre_lag = df_modelo_final.columns.to_list()
#                 df_datetime =  pipe_date.fit_transform(df_modelo_final)   
#                 df_datetime = df_datetime.drop(columns = cols_pre_lag)  
#                 df_datetime



#                 pipe = Pipeline([ 
#                     ("lagf",LagFeatures(variables =  lag_features ,freq = day_lag_modelo  ,missing_values = "ignore",) ), 
#                     ("winf",WindowFeatures(variables = lag_features ,  window =  day_wind_modelo ,freq = "1D", missing_values = "ignore",) ),
#                     ("drop_ts", DropFeatures(features_to_drop =  lag_features )),
#                 ]) 



#                 pipe_dropna = Pipeline([     
#                     ("dropna",DropMissingData()), 
#                 ])


#                 st.text('Lags e Windows')
#                 df_lags =  pipe.fit_transform(df_modelo_final[lag_features])    
#                 df_lags =  pipe_dropna.fit_transform(df_lags)    
#                 df_lags



#                 st.text('Modelo Features') 

#                 df_modelo_final = df_modelo_final.merge(  df_datetime, how='left', left_index=True, right_index=True) 
#                 df_modelo_final = df_modelo_final.merge(  df_lags, how='left', left_index=True, right_index=True) 

#                 df_modelo_final = df_modelo_final.replace(np.nan,0)

#                 try:
#                     drop_vars =  columns_gmv_modelo 
#                     drop_vars = [i for i in drop_vars if i != var_forecasting[0]]              
#                     df_modelo_final =  df_modelo_final.drop(columns = drop_vars ) 
#                 except:
#                     pass

#                 try:
#                     drop_vars =  columns_positivacao 
#                     drop_vars = [i for i in drop_vars if i != var_forecasting[0]]  
#                     df_modelo_final =  df_modelo_final.drop(columns = drop_vars ) 

#                 except:
#                     pass


#                 df_modelo_final = df_modelo_final.replace(to_replace=np.inf, value=0)
#                 df_modelo_final


#                 st.text('Modelo Scaled') 
                
#                 df_scaled = df_modelo_final.drop(columns = var_forecasting + columns_ofertao_modelo +  df_datetime.columns.to_list()).reset_index(drop = True)
#                 scaler = preprocessing.StandardScaler()
#                 arr_scaled = scaler.fit_transform( df_scaled)
#                 df_arr = pd.DataFrame(arr_scaled, columns = df_scaled.columns.to_list()) 
#                 df_arr

                
#                 st.text('Modelo Final') 
                

#                 df_modelo_final = df_modelo_final[ var_forecasting + columns_ofertao_modelo + df_datetime.columns.to_list()  ].reset_index()


#                 df_modelo_final = df_modelo_final.merge( df_arr , left_index = True, right_index=True,how = "left")#.set_index('Data')
#                 df_modelo_final = df_modelo_final.set_index('Data')
#                 df_modelo_final



# # %% Size Previsão

 
#         with st.expander('Drop Features and Point Outliers', expanded= False):


#             with st.form(key = "drop_outliers"):
                 
                

#                 size =   st.radio('Dias',  [30,15,45,60,90,120])
                
                 


# # %% Features Outliers 


                
#                 columns_list_modelo = df_modelo_final.columns.to_list()
#                 columns_list_modelo.remove(var_forecasting[0])

 

#                 if var_forecasting[0] == 'Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197':
 
                     

#                     drop_outliers_features_default = []
#                     drop_outliers_features_default.append('% Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_28D') 
#                     drop_outliers_features_default.append('Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_window_14D_mean')
#                     drop_outliers_features_default.append('% Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_window_14D_mean')
#                     drop_outliers_features_default.append('Price Concorrência Mundial Açúcar Refinado União 1Kg - 7891910000197')
#                     drop_outliers_features_default.append('Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_window_21D_mean')
#                     drop_outliers_features_default.append('% Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_window_21D_mean')
#                     drop_outliers_features_default.append('Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_window_28D_mean')
#                     drop_outliers_features_default.append('Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_14D')
#                     drop_outliers_features_default.append('Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_5D')

#                     drop_outliers_features_default.append('Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_5D')
#                     drop_outliers_features_default.append('% Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_1D')
#                     drop_outliers_features_default.append('Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_2D')
#                     drop_outliers_features_default.append('% Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_2D')
 
                    
#                     drop_outliers_features_default.append('weekend')
#                     drop_outliers_features_default.append('Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_14D')
#                     drop_outliers_features_default.append('Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_window_28D_mean')
#                     drop_outliers_features_default.append('month_sin')
#                     drop_outliers_features_default.append('month_cos')
#                     drop_outliers_features_default.append('Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_28D')
#                     drop_outliers_features_default.append('% Positivação  Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_21D')
#                     drop_outliers_features_default.append('week_cos')
#                     drop_outliers_features_default.append('Gmv Categoria Açúcar e Adoçante Açúcar Refinado União 1kg - 7891910000197_lag_21D')
# #                    drop_outliers_features_default.append('')




#                     drop_outliers_features_default_final = []

#                     for k in drop_outliers_features_default:
                        
#                         for col in columns_list_modelo:
#                             if k == col:

#                                 drop_outliers_features_default_final.append(k)

 


#                 if len(drop_outliers_features_default) != 0:
 
#                     drop_features =   st.multiselect('Drop Features',  columns_list_modelo , default =  drop_outliers_features_default_final )

#                 else: 
                    
#                     drop_features =  st.multiselect('Drop Features',  columns_list_modelo )


 
                        


# # %% Date Outliers

#                 # Date Outliers 
                           
#                 date_columns = df_modelo_final.sort_index(ascending = False)
#                 date_columns = date_columns.index.to_list()
 

 
#                 if len(drop_outliers_dates_default) != 0:

#                    drop_outliers_date=   st.multiselect('Drop Outliers Dates',  date_columns, default = drop_outliers_dates_default )
#                    df_outliers_date = pd.DataFrame({'Data':drop_outliers_date})
#                    df_outliers_date['Outlier'] = df_outliers_date 
                      
#                 else: 
                    
#                    drop_outliers_date=   st.multiselect('Drop Outliers Dates',  date_columns)


#                 submit_buttons_drop = st.form_submit_button(label = "Submit")




# # %% Train Test Split

    
#         with st.expander('Train Test Split', expanded= False):


#             # with st.form(key = "treino_teste_modelo"):
                    
                
#             #     submit_buttons_treino_teste = st.form_submit_button(label = "Submit")
      
 


#             df_modelo_final = df_modelo_final.drop(columns = drop_features) 
#             #df_modelo_final[var_forecasting[0] + '_']  = df_modelo_final[var_forecasting[0]] 
#            # df_modelo_final[var_forecasting[0]] = np.where(df_modelo_final['Outlier']>0,df_modelo_final[var_forecasting[0]].shift(periods=  7, freq="D") ,df_modelo_final[var_forecasting[0]])


#             data_ref = df_modelo_final.reset_index(drop=False)['Data'].dt.date.max() 
#             data_corte = pd.Timestamp(data_ref)  - pd.offsets.Day(size) 
                    

#             x_train = df_modelo_final[df_modelo_final.index <= data_corte ].drop(columns =var_forecasting)
#             x_train = x_train.dropna()
            
#             x_test = df_modelo_final[df_modelo_final.index > data_corte].drop(columns = var_forecasting)
#             y_train = df_modelo_final[df_modelo_final.index <=  data_corte][var_forecasting]
#             y_test = df_modelo_final[df_modelo_final.index > data_corte ][ var_forecasting ]
#             y_test = y_test.replace(np.nan, 0)
#             x_test = x_test.dropna( )
#             x_test['Outlier'] = 0

#             y_train = y_train.loc[x_train.index] 

#             y_train = y_train.iloc[:,0:1]

#             y_test = y_test.iloc[:,0:1]



#             st.markdown('#### Treino')     

#             y_treino, x_treino  = st.columns(2) 


#             with y_treino:
                
#                 st.text('Y Target') 
#                 y_train

                
#             with x_treino:

                        
#                 st.text('X Features') 
#                 x_train


#             st.markdown('#### Teste')   
#             y_teste, x_teste = st.columns(2)

#             with y_teste: 
                
#                 st.text('Y Target') 
#                 y_test

#             with x_teste:
                
#                 st.text('X Features')    
                        
#                 x_test


# # %%  Modelo Machine Learning


# lasso = Lasso( alpha = 100, random_state = 0 )
# lasso.fit(x_train, y_train)
# preds2 = lasso.predict(x_test)


# # %% Erro 
 
 
# # Output e Erro 

# coef = pd.DataFrame(pd.Series( lasso.coef_, index = x_train.columns), columns=['Coef']) 
# coef['Coef Abs'] = np.abs(coef['Coef'])
# df_predito = pd.DataFrame(preds2,columns=[var_forecasting[0] + ' Predito']) 

 
 
# y_test = y_test.reset_index(drop= False)
# y_test = y_test.merge(  df_predito, how='left', left_index=True, right_index=True)  
# y_test = y_test.set_index('Data')
# df_erro = y_test[:len(y_test) +1 ] #- size2]
# df_erro = df_erro.dropna() 
# df_erro = df_erro.where(df_erro >  0, 0)



# # df_erro  = df_erro.merge(df_modelo_segundo_final[[var_predicao + ' Sem Ajuste']], left_index = True, right_index=True,how = "left") 
# df_plot = df_erro.copy()

# df_erro = df_erro.iloc[:df_erro.shape[0]-1,:] 
# df_erro = df_erro[df_erro[var_forecasting[0]]>0]

# rmse = sqrt(mean_squared_error(df_erro[var_forecasting[0]], df_erro[var_forecasting[0] + ' Predito']))
# mape =  mean_absolute_percentage_error(df_erro[var_forecasting[0]], df_erro[var_forecasting[0] + ' Predito'])
 


# def calculate_mape(actual, predicted):
#     return np.mean(np.abs((actual - predicted) / actual) * 100)  # Assuming actual values are non-zero

# def calculate_rmse(actual, predicted):
#     return np.sqrt(np.mean((actual - predicted) ** 2))



 
 
# df_erro2 = df_erro.copy()
# df_erro2 = df_erro2[df_erro2.index.isin(drop_outliers_date) == False] 

# # rmse2 = calculate_rmse(df_erro2[var_forecasting[0]], df_erro2[var_forecasting[0] + ' Predito'])
# # mape2 = calculate_mape(df_erro2[var_forecasting[0]], df_erro2[var_forecasting[0] + ' Predito'])



# rmse2 = sqrt(mean_squared_error(df_erro2[var_forecasting[0]], df_erro2[var_forecasting[0] + ' Predito']))
# mape2 =  mean_absolute_percentage_error(df_erro2[var_forecasting[0]], df_erro2[var_forecasting[0] + ' Predito'])
  
 
  
# tail_columns = coef.tail(30).index.to_list()
# coef = coef.sort_values('Coef Abs', ascending = False)
# #coef.head(30) 

# #coef.tail(20).T.columns 


# df_plot = df_modelo_final[df_modelo_final.index > pd.Timestamp('2023-11-01')].merge(df_plot[var_forecasting[0] + ' Predito'], left_index = True, right_index=True,how = "left")
# #df_plot = df_plot[[  var_forecasting[0],  var_forecasting[0] + ' Predito'] ]

# df_plot = df_plot.sort_index(ascending = False)



# # rmse = sqrt(mean_squared_error(df_erro[var_forecasting[0] ], df_erro[var_forecasting[0]  + ' Predito']))
# # mape =  mean_absolute_percentage_error(df_erro[var_forecasting[0] ], df_erro[var_forecasting[0]  + ' Predito'])
# # coef = coef.sort_values(by=['Coef Abs'], ascending= False)


# # rmse2 = sqrt(mean_squared_error(df_erro2[var_forecasting[0] ], df_erro2[var_forecasting[0]  + ' Predito']))
# # mape2 =  mean_absolute_percentage_error(df_erro2[var_forecasting[0] ], df_erro2[var_forecasting[0]  + ' Predito'])
# # coef = coef.sort_values(by=['Coef Abs'], ascending= False)
 



# # %% Plot Erros e Coef

# st.markdown("### Predições")

# fig = go.Figure()

# sales_trace = go.Scatter(
#     x = df_plot.index,
#     y = df_plot[var_forecasting[0] ], 
#     yaxis="y1", 
#     name= var_forecasting[0]  ,
#     line_color = "#7900F1",
# )
# sales_trace2 = go.Scatter(
#     x = df_plot.index,
#     y = df_plot[var_forecasting[0] + ' Predito'], 
#     yaxis="y1", 
#     name= var_forecasting[0]  + ' Predito' , 
#     line_color = "#CD4C46",
# )

# fig.add_trace(sales_trace)   
# fig.add_trace(sales_trace2)   

# for outlier in drop_outliers_date:
    
#     date_outlier = outlier
#     # fig.add_vline(x=date_outlier, line_width=3,name = name_outlier , annotation_text = df_plot.loc[date_outlier, var_forecasting[0]] , line_dash="dash", line_color="white")


#     # Vertical line
#     fig.add_vline(x=date_outlier, line_width=0.5, line_dash="dash", line_color="white")

#     # Text annotation near the line
#     fig.add_annotation(
#         x=date_outlier,  # X-coordinate (same as vline)
#         y=  0,  # Y-coordinate (adjust based on data or preference)
#         xref='x',  # Reference for x-axis
#         yref='y',  # Reference for y-axis
#         text= '', 
#         showarrow=False,  # Hide arrow (optional)
#         xanchor='center',  # Horizontal anchor for text
#         yanchor='top',  # Vertical anchor for text (adjust as needed)
#         #font=dict(size=6, color='white')  # Customize font (optional)
#     )


# # for i in range(0,len(drop_outliers_date_default)):
# #     for outlier in drop_outliers_date_default[i]:
      
# #         date_outlier = drop_outliers_date_default[i][outlier]
# #        # fig.add_vline(x=date_outlier, line_width=3,name = name_outlier , annotation_text = df_plot.loc[date_outlier, var_forecasting[0]] , line_dash="dash", line_color="white")
 

# #         # Vertical line
# #         fig.add_vline(x=date_outlier, line_width=0.3, line_dash="dash", line_color="white")

# #         # Text annotation near the line
# #         fig.add_annotation(
# #             x=date_outlier,  # X-coordinate (same as vline)
# #             y= -2000,  # Y-coordinate (adjust based on data or preference)
# #             xref='x',  # Reference for x-axis
# #             yref='y',  # Reference for y-axis
# #             text= outlier, 
# #             showarrow=False,  # Hide arrow (optional)
# #             xanchor='center',  # Horizontal anchor for text
# #             yanchor='bottom',  # Vertical anchor for text (adjust as needed)
# #             font=dict(size=6.5, color='white')  # Customize font (optional)
# #         )

   
# #fig.update_layout(annotations=[{**a, **{"y":.5}}  for a in fig.to_dict()["layout"]["annotations"]])

# fig.update_layout(
    
#     width = 1200, height = 500,
#     legend=dict(x=0, y= 1.3, traceorder="reversed"), 
#    # xaxis_title='Date',
#    # yaxis_title= var_forecasting[0] ,
# )


# st.plotly_chart(fig)
 

# st.markdown("##### Erro e Coeficientes")
# st.markdown("#####  ")


# col = st.columns((6, 9 ), gap='medium')


# with col[0]:

#     st.markdown("##### Erro")          
#     st.text('Rmse: ' +  "{:,.2f}".format(rmse)  )
#     st.text('Mape: ' +  f"{mape:.2%}"  )
#     st.text('Rmse (Sem Outliers): ' +  "{:,.2f}".format(rmse2)  )
#     st.text('Mape (Sem Outliers): ' +  f"{mape2:.2%}"  ) 
 
#  #   df_date_outliers['Date'] = pd.to_datetime(df_date_outliers.values.flatten()) 
#   #  df_date_outliers = df_date_outliers.set_index('Date')

#     df_date_outliers

#    # df_date_outliers =  df_date_outliers.T 
     
#    # df_date_outliers.columns=[ 'Outlier ' + str(df_date_outliers.columns[k-1] +1)   for k in range(1, df_date_outliers.shape[1] + 1)]
#    # df_date_outliers


# with col[1]:
#     st.markdown("##### Coeficiente")

#     coef


 



# st.markdown("##### Base Predição")

# #df_plot['Delta Predição'] = df_plot[var_forecasting[0]]- df_plot[var_forecasting[0] + ' Predição']
# df_plot

  
 


#     # if cont >=5:
#     #     button = "Produtos " + cached_data[key]['Categoria'].unique()[0]
#     #     button_produto.append(button)
#     #     buttons_dic[button] = False 

  

# %% Forcasts D0 


with tab2:
     

    df_forecast = pd.concat([ df_rjc, df_rji,df_rj1,df_rj7,df_rj19,df_rj36,df_rj37,df_rj31,df_rj27,df_rj29,df_rj30,df_rj24, df_rj25,df_rj22  ])
    
    df_forecast = df_forecast.reset_index(drop = False)
    df_forecast['weekday'] = df_forecast['DateHour'].dt.weekday 
    df_forecast['Date'] =   pd.to_datetime(df_forecast['DateHour'].dt.date)
    df_forecast = df_forecast.set_index('DateHour')   
 
    df_forecast = df_forecast[['Date','Hora','Region Final','Forecast Gmv','Forecast Peso']]
    data_fim =  data_max -  pd.offsets.Day(1) 
    st.header("Forecast Gmv " + str(data_fim.strftime('%d/%m/%Y')) ) 
    print(df_forecast['Region Final'].unique().tolist())


   #df_forecast = df_forecast.sort_values('Forecast Peso',ascending= False)

    col = st.columns((2,  2), gap='medium')
    with col[0]:
            
                        
        st.markdown('#### RJC') 

        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJC' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]  

        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')

        fig     

  
    with col[1]:
  
        st.markdown('#### RJI')  
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJI' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    col = st.columns((2,  2), gap='medium')
    with col[0]:
        
                    
          
        st.markdown('#### RJC - Benfica - 1')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ1' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    with col[1]:
  
        st.markdown('#### RJC - SJM - 7')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ7' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    col = st.columns((2,  2), gap='medium')
    with col[0]:
        
                    
          
        st.markdown('#### RJC - Barra - 19')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ19' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    with col[1]:
  
        st.markdown('#### RJC - Niterói - 36')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ36' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    col = st.columns((2,  2), gap='medium')

    with col[0]:
        
                    
          
        st.markdown('#### RJM - Campo Grande - 37')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ37' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    with col[1]:
  
        st.markdown('#### RJM - Seropédica - 31')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ31' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    col = st.columns((2,  2), gap='medium')

    with col[0]:
        
                    
          
        st.markdown('#### RJM - Itaguai - 27')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ27' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig       

    with col[1]:
  
        st.markdown('#### RJM - Maricá - 29')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ29' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    col = st.columns((2,  2), gap='medium')

    with col[0]:
        
                    
          
        st.markdown('#### RJM - Rio Bonito - 30')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ30' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     
    with col[1]:
  
        st.markdown('#### RJI - Petrópolis - 23')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ24' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    col = st.columns((2,  2), gap='medium')


    with col[0]:
        
          
        st.markdown('#### RJI - Volta Redonda - 25')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ25' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     
    with col[1]:
  
        st.markdown('#### RJI - Cabo Frio - 22')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ22' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Gmv'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    
    df_forecast = df_forecast.reset_index(drop = False)
        

    df_dummies = pd.get_dummies(df_forecast['Region Final'])   
    lista_region = df_dummies.columns.tolist()
    lista_region.remove('RJC')
    lista_region.remove('RJI')
    lista_region_final = ['RJC','RJI'] + lista_region
    df_dummies = df_dummies[lista_region_final]    
    df_dummies.columns=['Forecast Gmv '  + str(df_dummies.columns[k-1])  for k in range(1, df_dummies.shape[1] + 1)]
    df_dummies = df_dummies.multiply(df_forecast['Forecast Gmv'], axis=0) 
    df_forecast = df_forecast.merge(  df_dummies, how='left', left_index=True, right_index=True)  
    df_forecast = df_forecast[df_forecast['Date'] == data_fim ]  
    df_forecast = df_forecast.drop(columns = ['Date','Region Final','Forecast Peso','Forecast Gmv']).groupby(['DateHour','Hora']).sum()
    df_forecast = df_forecast.sort_values(by=['DateHour'], ascending = False)
    df_forecast = df_forecast.apply(lambda x: round(x))
    df_forecast

with tab3:

 

    df_forecast = pd.concat([ df_rjc, df_rji,df_rj1,df_rj7,df_rj19,df_rj36,df_rj37,df_rj31,df_rj27,df_rj29,df_rj30,df_rj24, df_rj25,df_rj22  ])
    
    df_forecast = df_forecast.reset_index(drop = False)
    df_forecast['weekday'] = df_forecast['DateHour'].dt.weekday 
    df_forecast['Date'] =   pd.to_datetime(df_forecast['DateHour'].dt.date)
    df_forecast = df_forecast.set_index('DateHour')   
 
    df_forecast = df_forecast[['Date','Hora','Region Final','Forecast Gmv','Forecast Peso']]
    data_fim =  data_max -  pd.offsets.Day(1) 
    st.header("Forecast Peso " + str(data_fim.strftime('%d/%m/%Y')) ) 
    print(df_forecast['Region Final'].unique().tolist())


   #df_forecast = df_forecast.sort_values('Forecast Peso',ascending= False)

    col = st.columns((2,  2), gap='medium')
    with col[0]:
            
                        
        st.markdown('#### RJC') 

        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJC' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]  

        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')

        fig     




    with col[1]:
  
        st.markdown('#### RJI')  
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJI' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    col = st.columns((2,  2), gap='medium')
    with col[0]:
        
                    
          
        st.markdown('#### RJC - Benfica - 1')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ1' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    with col[1]:
  
        st.markdown('#### RJC - SJM - 7')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ7' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    col = st.columns((2,  2), gap='medium')
    with col[0]:
        
                    
          
        st.markdown('#### RJC - Barra - 19')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ19' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     
    with col[1]:
  
        st.markdown('#### RJC - Niterói - 36')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ36' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    col = st.columns((2,  2), gap='medium')
    with col[0]:
        
                    
          
        st.markdown('#### RJM - Campo Grande - 37')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ37' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    with col[1]:
  
        st.markdown('#### RJM - Seropédica - 31')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ31' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    col = st.columns((2,  2), gap='medium')
    with col[0]:
        
                    
          
        st.markdown('#### RJM - Itaguai - 27')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ27' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig       

    with col[1]:
  
        st.markdown('#### RJM - Maricá - 29')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ29' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     


    col = st.columns((2,  2), gap='medium')

    with col[0]:
        
                    
          
        st.markdown('#### RJM - Rio Bonito - 30')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ30' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     
    with col[1]:
  
        st.markdown('#### RJI - Petrópolis - 23')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ24' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    col = st.columns((2,  2), gap='medium')


    with col[0]:
        
          
        st.markdown('#### RJI - Volta Redonda - 25')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ25' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     
    with col[1]:
  
        st.markdown('#### RJI - Cabo Frio - 22')
        df_plot = df_forecast.copy()
        df_plot = df_plot[df_plot['Date'] == data_fim ] 
        df_plot = df_plot[df_plot['Region Final'] == 'RJ22' ]
        df_plot = df_plot[df_plot['Hora'] >= 6]   
        dados_x =  df_plot.Hora
        dados_y =  df_plot['Forecast Peso'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Hora", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')
        fig     

    
    df_forecast = df_forecast.reset_index(drop = False)
        

    df_dummies = pd.get_dummies(df_forecast['Region Final'])   
    lista_region = df_dummies.columns.tolist()
    lista_region.remove('RJC')
    lista_region.remove('RJI')
    lista_region_final = ['RJC','RJI'] + lista_region
    df_dummies = df_dummies[lista_region_final]    
    df_dummies.columns=['Forecast Peso '  + str(df_dummies.columns[k-1])  for k in range(1, df_dummies.shape[1] + 1)]
    df_dummies = df_dummies.multiply(df_forecast['Forecast Peso'], axis=0)
    df_forecast = df_forecast.merge(  df_dummies, how='left', left_index=True, right_index=True)   
    df_forecast = df_forecast[df_forecast['Date'] == data_fim ] 
    df_forecast = df_forecast.drop(columns = ['Date','Region Final','Forecast Peso','Forecast Gmv']).groupby(['DateHour','Hora']).sum()
    df_forecast = df_forecast.sort_values(by=['DateHour'], ascending = False)
    df_forecast = df_forecast.apply(lambda x: round(x))
    df_forecast
 




# with tab5:

#     st.markdown('###### Atualizado em: ' + str(hora_atualizacao_trafego) + ' / Hora Filtrada: ' + str(max_hora_trafego))  
    

#     st.header("Variação Trafego")
# #    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

#     with st.container(): 

#        # col = st.columns((9.5,  2), gap='small')

#         # with col[0]:
            
                    
#         #   st.markdown('#### Variação Tráfego')

#         df_view = df_view.sort_values('DateHour',ascending= False)
                    
#         df_view = df_view[['Date','Hora','trafego Acum', 'Orders Acum','% Conversão Acum', '% Var trafego Acum', '% Var % Conversão Acum' ,'Gmv Acum','Forecast Gmv', '% Var Gmv Acum' , 'Peso Acum',  'Forecast Peso', '% Var Peso Acum']]
        
        

            
#         delta_columns = ['% Var trafego Acum','% Var % Conversão Acum'] #,'% Var Gmv Acum' ,   '% Var Peso Acum']


    
#     #   for i in delta_columns:

# #             df_view[i] = df_view[i].apply(lambda x: f"{x:.2%}")
#     #      df_view[i] = df_view[i].apply(lambda x: '{:.2%}'.format(x))
#         #  df_view[i] = df_view[i].apply(lambda x: float(x.strip('%'))  ) 
#         #  df_view[i] = df_view[i].apply(lambda x: x[4:]  ) 
        
# #           styled_df = df_view[df_view['Date'] == data_max].style.apply(highlight_negative, subset=delta_columns)
# #           styled_df

#         df_style = df_view.copy()
#         df_style = df_style[['Gmv Acum','trafego Acum','Orders Acum','% Conversão Acum','% Var trafego Acum', '% Var % Conversão Acum']]


# #            df_style  = df_style[df_style['Date'] == data_max] 
        

#         def try_convert(value):
    
#             try:
#                 value = value.rstrip('%')
#                 value = float(value) / 100
#                 return value
#             except ValueError:
#                 return None

#         def highlight_negative(s):
#             return ['color: #CD4C46 ' if try_convert(v) < 0 else 'color: #7900F1' for v in s]
                
#             # Remove the trailing '%' sign (if present)
# #               percent_str = percent_str

#             # Convert the string to a float and divide by 100
# #                return float(percent_str) / 100
# #                return 


#         df_styled = df_style.style\
#                     .apply(highlight_negative, subset=delta_columns)

#         df_styled







#         # with col[1]:

                
#         #     st.markdown('#### Takeaways')
#         # #     df_trafego_query
#         #     with st.expander('About', expanded=True):
#         #         st.write('''
#         #             - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
#         #             - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
#         #             - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
#         #             ''')

#     col = st.columns((2,  2), gap='medium')

#     with col[0]:
            
                    
#         st.markdown('#### Trafego Acum') 

#         df_plot =  df_view.copy()

#         df_plot = df_plot.reset_index(drop = False)
#         df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
#         df_plot = df_plot.set_index('DateHour') 
        

#         if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
#         if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
 
 
#         dados_x =  df_plot.index
#         dados_y =  df_plot['trafego Acum']
#         dados_y2 =  df_plot['% Conversão Acum'] 
#         fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Data", y="Trafego") , height=300, width= 600, markers = True,    line_shape='spline')

#         fig 

#     with col[1]:


                    
#         st.markdown('#### Conversão') 
#         fig=py.line(x=dados_x, y=dados_y2, height=300, width= 600, markers = True,    line_shape='spline')
#         fig
#         #  labels=dict(x="Data", y="% Conversão") ,


#     col = st.columns((2,  2), gap='medium')

#     with col[0]:
            
                    
#         st.markdown('#### Gmv Acum') 

#         df_plot =  df_view.copy()

#         df_plot = df_plot.reset_index(drop = False)
#         df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
#         df_plot = df_plot.set_index('DateHour') 
        

#         if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
#         if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]


#         dados_x =  df_plot.index
#         dados_y =  df_plot['Gmv Acum']
#         dados_y2 =  df_plot['Orders Acum'] 
#         fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Data", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')

#         fig 

#     with col[1]:


                    
#         st.markdown('#### Pedidos Acum') 
#         fig=py.line(x=dados_x, y=dados_y2,  labels=dict(x="Data", y="Pedidos") , height=300, width= 600, markers = True,    line_shape='spline')
#         fig

# %%
