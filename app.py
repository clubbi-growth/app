
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


 

@st.cache_resource( ttl = 1800) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
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
 
  
    df_vendas['% Positivação Categoria'] = df_vendas['Positivação Categoria'] /  df_vendas['Positivação Geral']

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



 

 
# %% Load Categoria

print('Load Categoria')

#@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
@st.cache_resource( ttl = 1800) 
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
 
@st.cache_resource( ttl = 1800) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
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
df_top_categorias = df_top_categorias.head(15)
df_top_categorias = df_top_categorias.reset_index()
categoria_list = df_top_categorias['Categoria'].unique().tolist() 
 

dicts = {}
name_list = []
list_dicts = []

regional_list =  ['RJC','RJI','BAC']
size_list = ['size','1-4 Cxs','5-9 Cxs']    
 


@st.cache_resource( ttl = 1800) 
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
    
 
tab0   , tab1   = st.tabs(["Performance D0","Categorias" ])
df_count = 0 

button_count = 0   

# %% Categorias 

with tab0: 

    df_categorias2['Date'] = df_categorias2.index.date
 

    if len (df_categorias2[df_categorias2['Date'] == datetime.date.today() ]) >0 :
            
        date_ref = datetime.date.today()   
    else: 
        date_ref = datetime.date.today()    - pd.offsets.Day(1)  


    hora_atual = df_categorias2[df_categorias2['Date'] == date_ref ]['Hora'].max()
     

    if hora_atual <=6:
        
        date_ref = datetime.date.today()    - pd.offsets.Day(1)  
        date_ref = str(date_ref)[0:10]
        date_ref = datetime.datetime.strptime( date_ref , "%Y-%m-%d")
        
        date_ref = date_ref.date() 
        hora_atual = df_categorias2[df_categorias2['Date'] == date_ref ]['Hora'].max()  
    else:
        hora_atual = df_categorias2[df_categorias2['Date'] == date_ref ]['Hora'].max()  
     

    df_categorias2 = df_categorias2.sort_index(ascending=True) 
       
    st.markdown('###### Dados até: ' + str(date_ref) + ' '  + str(hora_atual))  
  
 
    @st.cache_resource( ttl = 1800) 
    def load_df_categoria_Final():
        regional_list =  ['RJC', 'RJI','BAC']  
        size_list =     ['1-4 Cxs','5-9 Cxs' , 'size']   


        df_categoria_final = pd.DataFrame()

        for i in regional_list:
            
                
            for s in size_list:


                df_plot = df_categorias2.copy() 
                df_plot = df_plot.reset_index() 
                df_plot = df_plot.set_index('DateHour')  
                weekday = date_ref.weekday()
                
                df_plot = df_plot[df_plot['Region']== i ]  
                df_plot = df_plot[df_plot['Size']== s]   
                df_plot = df_plot[df_plot['Weekday']== weekday]  
                categoria_list_loop = df_plot['Categoria'].unique().tolist()
                

                df_plot = df_plot[['Categoria','Hora','Gmv Acum', 'Positivação Geral', 'Positivação Categoria', '% Positivação Categoria']]
                 
    
    

                for c in  categoria_list_loop:

                    df_categoria = df_plot.copy()
                    df_categoria = df_categoria[df_categoria['Categoria'] == c]
                    
                    df_categoria_fim = df_categoria[df_categoria['Hora'] == 23]  
                    df_categoria = df_categoria[df_categoria['Hora'] == hora_atual]  
                    
                    
                    df_categoria_fim = df_categoria_fim.drop(columns = ['Categoria','Hora']) 
                    df_categoria = df_categoria.drop(columns = ['Categoria','Hora'])
                    
                    cols_cate =  df_categoria.columns.to_list() 

                    
                    df_categoria['Date'] = df_categoria.index.date 
                    df_categoria = df_categoria.reset_index(drop = False)
                    df_categoria = df_categoria.set_index('Date')
    
                    df_categoria_fim['Date'] = df_categoria_fim.index.date  
                    df_categoria_fim = df_categoria_fim.reset_index(drop = False)
                    df_categoria_fim = df_categoria_fim.set_index('Date')
                    df_categoria_fim.columns=[  str(df_categoria_fim.columns[k-1]) + '_23h'   for k in range(1, df_categoria_fim.shape[1] + 1)]

 
                    cols_cate = df_categoria.columns.to_list()
                    
                    
                    df_categoria = df_categoria.merge( df_categoria_fim, how='left', left_index=True, right_index=True)  
                    


                    for g in ['Gmv Acum','Positivação Categoria','Positivação Geral']:
                        
                        df_categoria['% Share ' + g] = df_categoria[g]/df_categoria[g + '_23h'] 


                    cols_share = df_categoria.filter(regex='% Share').columns.to_list()
                    
                    df_categoria = df_categoria.set_index('DateHour')
                    
                    cols_cate = [i for i in cols_cate if i.find('DateHour' ,0)<0 ]   

                    df_categoria = df_categoria[cols_cate + cols_share]

                    pipe = Pipeline([ 
                        ("lagf",LagFeatures(variables = cols_share ,freq = ["7D","14D","21D","28D"]  ,missing_values = "ignore",) ), 
                        #  ("winf",WindowFeatures(variables = lag_features ,  window =  day_wind_modelo ,freq = "1D", missing_values = "ignore",) ),
                        #  ("drop_ts", DropFeatures(features_to_drop =  lag_features )),
                    ]) 

                    df_categoria =  pipe.fit_transform(df_categoria) 
                    df_categoria['Mean % Share Gmv'] = (df_categoria['% Share Gmv Acum_lag_7D'] + df_categoria['% Share Gmv Acum_lag_14D'] + df_categoria['% Share Gmv Acum_lag_21D'] + df_categoria['% Share Gmv Acum_lag_28D'])/4
                    df_categoria['Mean % Share Positivação Categoria'] = (df_categoria['% Share Positivação Categoria_lag_7D'] + df_categoria['% Share Positivação Categoria_lag_14D'] + df_categoria['% Share Positivação Categoria_lag_21D'] + df_categoria['% Share Positivação Categoria_lag_28D'])/4
                    df_categoria['Mean % Share Positivação Geral'] = (df_categoria['% Share Positivação Geral_lag_7D'] + df_categoria['% Share Positivação Geral_lag_14D'] + df_categoria['% Share Positivação Geral_lag_21D'] + df_categoria['% Share Positivação Geral_lag_28D'])/4
                    
                    df_categoria['Forecast Gmv'] = df_categoria['Gmv Acum']/df_categoria['Mean % Share Gmv']
 
                    
                    df_categoria['Forecast Positivação Categoria'] = df_categoria['Positivação Categoria']/df_categoria['Mean % Share Positivação Categoria']
                    
                    df_categoria['Forecast Positivação Geral'] = df_categoria['Positivação Geral']/ df_categoria['Mean % Share Positivação Geral'] 
                    
                    df_categoria['Forecast % Positivação'] =   df_categoria['Forecast Positivação Categoria']  /df_categoria['Forecast Positivação Geral']


                    columns_share = [i for i in df_categoria.columns.to_list() if i.find('% Share' ,0)<0  ]   

                    df_categoria = df_categoria[columns_share]
                    df_categoria['AOV'] = df_categoria['Gmv Acum']/df_categoria['Positivação Categoria']

                    pipe = Pipeline([ 
                        ("lagf",LagFeatures(variables = ['% Positivação Categoria','AOV'] ,freq = ["7D"]  ,missing_values = "ignore",) ), 
                        #  ("winf",WindowFeatures(variables = lag_features ,  window =  day_wind_modelo ,freq = "1D", missing_values = "ignore",) ),
                        #  ("drop_ts", DropFeatures(features_to_drop =  lag_features )),
                    ]) 

                    df_categoria =  pipe.fit_transform(df_categoria) 
                    #df_categoria = df_categoria.rename(columns = {'% Positivação Categoria_lag_7D':'% Target Positivação'})
                    df_categoria = df_categoria.rename(columns = {'AOV_lag_7D':'Target AOV'})
 
                    
                    df_categoria['Size'] = s
                    df_categoria['Regional'] = i
                    df_categoria['Categoria'] = c

                    df_categoria['Data'] =  df_categoria.index.date   
                    df_categoria = df_categoria.reset_index(drop = False)
                    df_categoria = df_categoria.set_index('Data') 
                     
    
                    df_ofertao_final = ofertao(df_ofertao_inicial,[c], [i], [s])    

                    
                    if len(df_ofertao_final) >0:

                        
                        df_ofertao_final['Date'] =  df_ofertao_final.index.date   
                        df_ofertao_final = df_ofertao_final.reset_index(drop = False)
                        df_ofertao_final = df_ofertao_final.set_index('Date')
                        df_ofertao_final = df_ofertao_final.drop(columns = ['Data'])
                    


                        df_categoria = df_categoria.merge( df_ofertao_final, how='left', left_index=True, right_index=True)  
                        df_categoria = df_categoria.rename(columns = {'Ofertão ' + c :'Ofertão'})

                    else:
                        df_categoria['Ofertão'] = 0
                    
                           
                    df_categoria['Ofertão'] = df_categoria['Ofertão'].replace(np.nan, 0)
                    df_categoria = df_categoria.set_index('DateHour')

 
                    if len(df_categoria_final) ==0: 
                        df_categoria_final = df_categoria.copy()
                    else:
                        df_categoria_final = pd.concat([df_categoria_final,  df_categoria.copy()])
                    
  

            

        df_categoria_final['Ofertão'] = np.where(df_categoria_final['Ofertão'] == 0, 'Bau','Ofertão')
        df_categoria_final['Key'] =  df_categoria_final['Ofertão'] + df_categoria_final['Categoria'] + df_categoria_final['Regional'] + df_categoria_final['Size']
    
        df_categoria_final = df_categoria_final.reset_index(drop = False)
        df_categoria_final = df_categoria_final.set_index('Key')



        # dict_targets =  {
        #                 "Tipo": {
        #                     "0": "Bau",
        #                     "1": "Bau",
        #                     "2": "Bau",
        #                     "3": "Bau",
        #                     "4": "Bau",
        #                     "5": "Bau",
        #                     "6": "Bau",
        #                     "7": "Bau",
        #                     "8": "Bau",
        #                     "9": "Bau",
        #                     "10": "Bau",
        #                     "11": "Bau",
        #                     "12": "Bau",
        #                     "13": "Bau",
        #                     "14": "Bau",
        #                     "15": "Bau",
        #                     "16": "Bau",
        #                     "17": "Bau",
        #                     "18": "Bau",
        #                     "19": "Bau",
        #                     "20": "Bau",
        #                     "21": "Bau",
        #                     "22": "Bau",
        #                     "23": "Bau",
        #                     "24": "Bau",
        #                     "25": "Bau",
        #                     "26": "Bau",
        #                     "27": "Bau",
        #                     "28": "Bau",
        #                     "29": "Bau",
        #                     "30": "Bau",
        #                     "31": "Bau",
        #                     "32": "Bau",
        #                     "33": "Bau",
        #                     "34": "Bau",
        #                     "35": "Bau",
        #                     "36": "Bau",
        #                     "37": "Bau",
        #                     "38": "Bau",
        #                     "39": "Bau",
        #                     "40": "Bau",
        #                     "41": "Bau",
        #                     "42": "Bau",
        #                     "43": "Bau",
        #                     "44": "Bau",
        #                     "45": "Bau",
        #                     "46": "Bau",
        #                     "47": "Bau",
        #                     "48": "Bau",
        #                     "49": "Bau",
        #                     "50": "Bau",
        #                     "51": "Bau",
        #                     "52": "Bau",
        #                     "53": "Bau",
        #                     "54": "Bau",
        #                     "55": "Bau",
        #                     "56": "Bau",
        #                     "57": "Bau",
        #                     "58": "Bau",
        #                     "59": "Bau",
        #                     "60": "Bau",
        #                     "61": "Bau",
        #                     "62": "Bau",
        #                     "63": "Bau",
        #                     "64": "Bau",
        #                     "65": "Bau",
        #                     "66": "Bau",
        #                     "67": "Bau",
        #                     "68": "Bau",
        #                     "69": "Bau",
        #                     "70": "Bau",
        #                     "71": "Bau",
        #                     "72": "Bau",
        #                     "73": "Bau",
        #                     "74": "Bau",
        #                     "75": "Bau",
        #                     "76": "Bau",
        #                     "77": "Bau",
        #                     "78": "Bau",
        #                     "79": "Bau",
        #                     "80": "Bau",
        #                     "81": "Bau",
        #                     "82": "Bau",
        #                     "83": "Bau",
        #                     "84": "Bau",
        #                     "85": "Bau",
        #                     "86": "Bau",
        #                     "87": "Bau",
        #                     "88": "Bau",
        #                     "89": "Bau",
        #                     "90": "Bau",
        #                     "91": "Bau",
        #                     "92": "Bau",
        #                     "93": "Bau",
        #                     "94": "Bau",
        #                     "95": "Bau",
        #                     "96": "Bau",
        #                     "97": "Bau",
        #                     "98": "Bau",
        #                     "99": "Bau",
        #                     "100": "Bau",
        #                     "101": "Bau",
        #                     "102": "Bau",
        #                     "103": "Bau",
        #                     "104": "Bau",
        #                     "105": "Bau",
        #                     "106": "Bau",
        #                     "107": "Bau",
        #                     "108": "Bau",
        #                     "109": "Bau",
        #                     "110": "Bau",
        #                     "111": "Bau",
        #                     "112": "Bau",
        #                     "113": "Bau",
        #                     "114": "Bau",
        #                     "115": "Bau",
        #                     "116": "Bau",
        #                     "117": "Bau",
        #                     "118": "Bau",
        #                     "119": "Bau",
        #                     "120": "Geral",
        #                     "121": "Geral",
        #                     "122": "Geral",
        #                     "123": "Geral",
        #                     "124": "Geral",
        #                     "125": "Geral",
        #                     "126": "Geral",
        #                     "127": "Geral",
        #                     "128": "Geral",
        #                     "129": "Geral",
        #                     "130": "Geral",
        #                     "131": "Geral",
        #                     "132": "Geral",
        #                     "133": "Geral",
        #                     "134": "Geral",
        #                     "135": "Geral",
        #                     "136": "Geral",
        #                     "137": "Geral",
        #                     "138": "Geral",
        #                     "139": "Geral",
        #                     "140": "Geral",
        #                     "141": "Geral",
        #                     "142": "Geral",
        #                     "143": "Geral",
        #                     "144": "Geral",
        #                     "145": "Geral",
        #                     "146": "Geral",
        #                     "147": "Geral",
        #                     "148": "Geral",
        #                     "149": "Geral",
        #                     "150": "Geral",
        #                     "151": "Geral",
        #                     "152": "Geral",
        #                     "153": "Geral",
        #                     "154": "Geral",
        #                     "155": "Geral",
        #                     "156": "Geral",
        #                     "157": "Geral",
        #                     "158": "Geral",
        #                     "159": "Geral",
        #                     "160": "Geral",
        #                     "161": "Geral",
        #                     "162": "Geral",
        #                     "163": "Geral",
        #                     "164": "Geral",
        #                     "165": "Geral",
        #                     "166": "Geral",
        #                     "167": "Geral",
        #                     "168": "Geral",
        #                     "169": "Geral",
        #                     "170": "Geral",
        #                     "171": "Geral",
        #                     "172": "Geral",
        #                     "173": "Geral",
        #                     "174": "Geral",
        #                     "175": "Geral",
        #                     "176": "Geral",
        #                     "177": "Geral",
        #                     "178": "Geral",
        #                     "179": "Geral",
        #                     "180": "Geral",
        #                     "181": "Geral",
        #                     "182": "Geral",
        #                     "183": "Geral",
        #                     "184": "Geral",
        #                     "185": "Geral",
        #                     "186": "Geral",
        #                     "187": "Geral",
        #                     "188": "Geral",
        #                     "189": "Geral",
        #                     "190": "Geral",
        #                     "191": "Geral",
        #                     "192": "Geral",
        #                     "193": "Geral",
        #                     "194": "Geral",
        #                     "195": "Geral",
        #                     "196": "Geral",
        #                     "197": "Geral",
        #                     "198": "Geral",
        #                     "199": "Geral",
        #                     "200": "Geral",
        #                     "201": "Geral",
        #                     "202": "Geral",
        #                     "203": "Geral",
        #                     "204": "Geral",
        #                     "205": "Geral",
        #                     "206": "Geral",
        #                     "207": "Geral",
        #                     "208": "Geral",
        #                     "209": "Geral",
        #                     "210": "Geral",
        #                     "211": "Geral",
        #                     "212": "Geral",
        #                     "213": "Geral",
        #                     "214": "Geral",
        #                     "215": "Geral",
        #                     "216": "Geral",
        #                     "217": "Geral",
        #                     "218": "Geral",
        #                     "219": "Geral",
        #                     "220": "Geral",
        #                     "221": "Geral",
        #                     "222": "Geral",
        #                     "223": "Geral",
        #                     "224": "Geral",
        #                     "225": "Geral",
        #                     "226": "Geral",
        #                     "227": "Geral",
        #                     "228": "Geral",
        #                     "229": "Geral",
        #                     "230": "Geral",
        #                     "231": "Geral",
        #                     "232": "Geral",
        #                     "233": "Geral",
        #                     "234": "Geral",
        #                     "235": "Geral",
        #                     "236": "Geral",
        #                     "237": "Geral",
        #                     "238": "Geral",
        #                     "239": "Geral",
        #                     "240": "Ofertão",
        #                     "241": "Ofertão",
        #                     "242": "Ofertão",
        #                     "243": "Ofertão",
        #                     "244": "Ofertão",
        #                     "245": "Ofertão",
        #                     "246": "Ofertão",
        #                     "247": "Ofertão",
        #                     "248": "Ofertão",
        #                     "249": "Ofertão",
        #                     "250": "Ofertão",
        #                     "251": "Ofertão",
        #                     "252": "Ofertão",
        #                     "253": "Ofertão",
        #                     "254": "Ofertão",
        #                     "255": "Ofertão",
        #                     "256": "Ofertão",
        #                     "257": "Ofertão",
        #                     "258": "Ofertão",
        #                     "259": "Ofertão",
        #                     "260": "Ofertão",
        #                     "261": "Ofertão",
        #                     "262": "Ofertão",
        #                     "263": "Ofertão",
        #                     "264": "Ofertão",
        #                     "265": "Ofertão",
        #                     "266": "Ofertão",
        #                     "267": "Ofertão",
        #                     "268": "Ofertão",
        #                     "269": "Ofertão",
        #                     "270": "Ofertão",
        #                     "271": "Ofertão",
        #                     "272": "Ofertão",
        #                     "273": "Ofertão",
        #                     "274": "Ofertão",
        #                     "275": "Ofertão",
        #                     "276": "Ofertão",
        #                     "277": "Ofertão",
        #                     "278": "Ofertão",
        #                     "279": "Ofertão",
        #                     "280": "Ofertão",
        #                     "281": "Ofertão",
        #                     "282": "Ofertão",
        #                     "283": "Ofertão",
        #                     "284": "Ofertão",
        #                     "285": "Ofertão",
        #                     "286": "Ofertão",
        #                     "287": "Ofertão",
        #                     "288": "Ofertão",
        #                     "289": "Ofertão",
        #                     "290": "Ofertão",
        #                     "291": "Ofertão",
        #                     "292": "Ofertão",
        #                     "293": "Ofertão",
        #                     "294": "Ofertão",
        #                     "295": "Ofertão",
        #                     "296": "Ofertão",
        #                     "297": "Ofertão",
        #                     "298": "Ofertão",
        #                     "299": "Ofertão",
        #                     "300": "Ofertão",
        #                     "301": "Ofertão",
        #                     "302": "Ofertão",
        #                     "303": "Ofertão",
        #                     "304": "Ofertão",
        #                     "305": "Ofertão",
        #                     "306": "Ofertão",
        #                     "307": "Ofertão",
        #                     "308": "Ofertão",
        #                     "309": "Ofertão",
        #                     "310": "Ofertão",
        #                     "311": "Ofertão",
        #                     "312": "Ofertão",
        #                     "313": "Ofertão",
        #                     "314": "Ofertão",
        #                     "315": "Ofertão",
        #                     "316": "Ofertão",
        #                     "317": "Ofertão",
        #                     "318": "Ofertão",
        #                     "319": "Ofertão",
        #                     "320": "Ofertão",
        #                     "321": "Ofertão",
        #                     "322": "Ofertão",
        #                     "323": "Ofertão",
        #                     "324": "Ofertão",
        #                     "325": "Ofertão",
        #                     "326": "Ofertão",
        #                     "327": "Ofertão",
        #                     "328": "Ofertão",
        #                     "329": "Ofertão",
        #                     "330": "Ofertão",
        #                     "331": "Ofertão",
        #                     "332": "Ofertão",
        #                     "333": "Ofertão",
        #                     "334": "Ofertão",
        #                     "335": "Ofertão",
        #                     "336": "Ofertão",
        #                     "337": "Ofertão",
        #                     "338": "Ofertão",
        #                     "339": "Ofertão",
        #                     "340": "Ofertão",
        #                     "341": "Ofertão",
        #                     "342": "Ofertão",
        #                     "343": "Ofertão",
        #                     "344": "Ofertão"
        #                 },
        #                 "Categoria": {
        #                     "0": "Arroz e Feijão",
        #                     "1": "Arroz e Feijão",
        #                     "2": "Arroz e Feijão",
        #                     "3": "Arroz e Feijão",
        #                     "4": "Arroz e Feijão",
        #                     "5": "Arroz e Feijão",
        #                     "6": "Arroz e Feijão",
        #                     "7": "Arroz e Feijão",
        #                     "8": "Açúcar e Adoçante",
        #                     "9": "Açúcar e Adoçante",
        #                     "10": "Açúcar e Adoçante",
        #                     "11": "Açúcar e Adoçante",
        #                     "12": "Açúcar e Adoçante",
        #                     "13": "Açúcar e Adoçante",
        #                     "14": "Açúcar e Adoçante",
        #                     "15": "Açúcar e Adoçante",
        #                     "16": "Biscoitos",
        #                     "17": "Biscoitos",
        #                     "18": "Biscoitos",
        #                     "19": "Biscoitos",
        #                     "20": "Biscoitos",
        #                     "21": "Biscoitos",
        #                     "22": "Biscoitos",
        #                     "23": "Biscoitos",
        #                     "24": "Cafés, Chás e Achocolatados",
        #                     "25": "Cafés, Chás e Achocolatados",
        #                     "26": "Cafés, Chás e Achocolatados",
        #                     "27": "Cafés, Chás e Achocolatados",
        #                     "28": "Cafés, Chás e Achocolatados",
        #                     "29": "Cafés, Chás e Achocolatados",
        #                     "30": "Cafés, Chás e Achocolatados",
        #                     "31": "Cafés, Chás e Achocolatados",
        #                     "32": "Cervejas",
        #                     "33": "Cervejas",
        #                     "34": "Cervejas",
        #                     "35": "Cervejas",
        #                     "36": "Cervejas",
        #                     "37": "Cervejas",
        #                     "38": "Cervejas",
        #                     "39": "Cervejas",
        #                     "40": "Cervejas Premium",
        #                     "41": "Cervejas Premium",
        #                     "42": "Cervejas Premium",
        #                     "43": "Cervejas Premium",
        #                     "44": "Cervejas Premium",
        #                     "45": "Cervejas Premium",
        #                     "46": "Cervejas Premium",
        #                     "47": "Cervejas Premium",
        #                     "48": "Derivados de Leite",
        #                     "49": "Derivados de Leite",
        #                     "50": "Derivados de Leite",
        #                     "51": "Derivados de Leite",
        #                     "52": "Derivados de Leite",
        #                     "53": "Derivados de Leite",
        #                     "54": "Derivados de Leite",
        #                     "55": "Derivados de Leite",
        #                     "56": "Grãos e Farináceos",
        #                     "57": "Grãos e Farináceos",
        #                     "58": "Grãos e Farináceos",
        #                     "59": "Grãos e Farináceos",
        #                     "60": "Grãos e Farináceos",
        #                     "61": "Grãos e Farináceos",
        #                     "62": "Grãos e Farináceos",
        #                     "63": "Grãos e Farináceos",
        #                     "64": "Leite",
        #                     "65": "Leite",
        #                     "66": "Leite",
        #                     "67": "Leite",
        #                     "68": "Leite",
        #                     "69": "Leite",
        #                     "70": "Leite",
        #                     "71": "Leite",
        #                     "72": "Limpeza de Roupa",
        #                     "73": "Limpeza de Roupa",
        #                     "74": "Limpeza de Roupa",
        #                     "75": "Limpeza de Roupa",
        #                     "76": "Limpeza de Roupa",
        #                     "77": "Limpeza de Roupa",
        #                     "78": "Limpeza de Roupa",
        #                     "79": "Limpeza de Roupa",
        #                     "80": "Limpeza em Geral",
        #                     "81": "Limpeza em Geral",
        #                     "82": "Limpeza em Geral",
        #                     "83": "Limpeza em Geral",
        #                     "84": "Limpeza em Geral",
        #                     "85": "Limpeza em Geral",
        #                     "86": "Limpeza em Geral",
        #                     "87": "Limpeza em Geral",
        #                     "88": "Massas Secas",
        #                     "89": "Massas Secas",
        #                     "90": "Massas Secas",
        #                     "91": "Massas Secas",
        #                     "92": "Massas Secas",
        #                     "93": "Massas Secas",
        #                     "94": "Massas Secas",
        #                     "95": "Massas Secas",
        #                     "96": "Refrigerantes",
        #                     "97": "Refrigerantes",
        #                     "98": "Refrigerantes",
        #                     "99": "Refrigerantes",
        #                     "100": "Refrigerantes",
        #                     "101": "Refrigerantes",
        #                     "102": "Refrigerantes",
        #                     "103": "Refrigerantes",
        #                     "104": "Sucos E Refrescos",
        #                     "105": "Sucos E Refrescos",
        #                     "106": "Sucos E Refrescos",
        #                     "107": "Sucos E Refrescos",
        #                     "108": "Sucos E Refrescos",
        #                     "109": "Sucos E Refrescos",
        #                     "110": "Sucos E Refrescos",
        #                     "111": "Sucos E Refrescos",
        #                     "112": "Óleos, Azeites e Vinagres",
        #                     "113": "Óleos, Azeites e Vinagres",
        #                     "114": "Óleos, Azeites e Vinagres",
        #                     "115": "Óleos, Azeites e Vinagres",
        #                     "116": "Óleos, Azeites e Vinagres",
        #                     "117": "Óleos, Azeites e Vinagres",
        #                     "118": "Óleos, Azeites e Vinagres",
        #                     "119": "Óleos, Azeites e Vinagres",
        #                     "120": "Arroz e Feijão",
        #                     "121": "Arroz e Feijão",
        #                     "122": "Arroz e Feijão",
        #                     "123": "Arroz e Feijão",
        #                     "124": "Arroz e Feijão",
        #                     "125": "Arroz e Feijão",
        #                     "126": "Arroz e Feijão",
        #                     "127": "Arroz e Feijão",
        #                     "128": "Açúcar e Adoçante",
        #                     "129": "Açúcar e Adoçante",
        #                     "130": "Açúcar e Adoçante",
        #                     "131": "Açúcar e Adoçante",
        #                     "132": "Açúcar e Adoçante",
        #                     "133": "Açúcar e Adoçante",
        #                     "134": "Açúcar e Adoçante",
        #                     "135": "Açúcar e Adoçante",
        #                     "136": "Biscoitos",
        #                     "137": "Biscoitos",
        #                     "138": "Biscoitos",
        #                     "139": "Biscoitos",
        #                     "140": "Biscoitos",
        #                     "141": "Biscoitos",
        #                     "142": "Biscoitos",
        #                     "143": "Biscoitos",
        #                     "144": "Cafés, Chás e Achocolatados",
        #                     "145": "Cafés, Chás e Achocolatados",
        #                     "146": "Cafés, Chás e Achocolatados",
        #                     "147": "Cafés, Chás e Achocolatados",
        #                     "148": "Cafés, Chás e Achocolatados",
        #                     "149": "Cafés, Chás e Achocolatados",
        #                     "150": "Cafés, Chás e Achocolatados",
        #                     "151": "Cafés, Chás e Achocolatados",
        #                     "152": "Cervejas",
        #                     "153": "Cervejas",
        #                     "154": "Cervejas",
        #                     "155": "Cervejas",
        #                     "156": "Cervejas",
        #                     "157": "Cervejas",
        #                     "158": "Cervejas",
        #                     "159": "Cervejas",
        #                     "160": "Cervejas Premium",
        #                     "161": "Cervejas Premium",
        #                     "162": "Cervejas Premium",
        #                     "163": "Cervejas Premium",
        #                     "164": "Cervejas Premium",
        #                     "165": "Cervejas Premium",
        #                     "166": "Cervejas Premium",
        #                     "167": "Cervejas Premium",
        #                     "168": "Derivados de Leite",
        #                     "169": "Derivados de Leite",
        #                     "170": "Derivados de Leite",
        #                     "171": "Derivados de Leite",
        #                     "172": "Derivados de Leite",
        #                     "173": "Derivados de Leite",
        #                     "174": "Derivados de Leite",
        #                     "175": "Derivados de Leite",
        #                     "176": "Grãos e Farináceos",
        #                     "177": "Grãos e Farináceos",
        #                     "178": "Grãos e Farináceos",
        #                     "179": "Grãos e Farináceos",
        #                     "180": "Grãos e Farináceos",
        #                     "181": "Grãos e Farináceos",
        #                     "182": "Grãos e Farináceos",
        #                     "183": "Grãos e Farináceos",
        #                     "184": "Leite",
        #                     "185": "Leite",
        #                     "186": "Leite",
        #                     "187": "Leite",
        #                     "188": "Leite",
        #                     "189": "Leite",
        #                     "190": "Leite",
        #                     "191": "Leite",
        #                     "192": "Limpeza de Roupa",
        #                     "193": "Limpeza de Roupa",
        #                     "194": "Limpeza de Roupa",
        #                     "195": "Limpeza de Roupa",
        #                     "196": "Limpeza de Roupa",
        #                     "197": "Limpeza de Roupa",
        #                     "198": "Limpeza de Roupa",
        #                     "199": "Limpeza de Roupa",
        #                     "200": "Limpeza em Geral",
        #                     "201": "Limpeza em Geral",
        #                     "202": "Limpeza em Geral",
        #                     "203": "Limpeza em Geral",
        #                     "204": "Limpeza em Geral",
        #                     "205": "Limpeza em Geral",
        #                     "206": "Limpeza em Geral",
        #                     "207": "Limpeza em Geral",
        #                     "208": "Massas Secas",
        #                     "209": "Massas Secas",
        #                     "210": "Massas Secas",
        #                     "211": "Massas Secas",
        #                     "212": "Massas Secas",
        #                     "213": "Massas Secas",
        #                     "214": "Massas Secas",
        #                     "215": "Massas Secas",
        #                     "216": "Refrigerantes",
        #                     "217": "Refrigerantes",
        #                     "218": "Refrigerantes",
        #                     "219": "Refrigerantes",
        #                     "220": "Refrigerantes",
        #                     "221": "Refrigerantes",
        #                     "222": "Refrigerantes",
        #                     "223": "Refrigerantes",
        #                     "224": "Sucos E Refrescos",
        #                     "225": "Sucos E Refrescos",
        #                     "226": "Sucos E Refrescos",
        #                     "227": "Sucos E Refrescos",
        #                     "228": "Sucos E Refrescos",
        #                     "229": "Sucos E Refrescos",
        #                     "230": "Sucos E Refrescos",
        #                     "231": "Sucos E Refrescos",
        #                     "232": "Óleos, Azeites e Vinagres",
        #                     "233": "Óleos, Azeites e Vinagres",
        #                     "234": "Óleos, Azeites e Vinagres",
        #                     "235": "Óleos, Azeites e Vinagres",
        #                     "236": "Óleos, Azeites e Vinagres",
        #                     "237": "Óleos, Azeites e Vinagres",
        #                     "238": "Óleos, Azeites e Vinagres",
        #                     "239": "Óleos, Azeites e Vinagres",
        #                     "240": "Arroz e Feijão",
        #                     "241": "Arroz e Feijão",
        #                     "242": "Arroz e Feijão",
        #                     "243": "Arroz e Feijão",
        #                     "244": "Arroz e Feijão",
        #                     "245": "Arroz e Feijão",
        #                     "246": "Arroz e Feijão",
        #                     "247": "Açúcar e Adoçante",
        #                     "248": "Açúcar e Adoçante",
        #                     "249": "Açúcar e Adoçante",
        #                     "250": "Açúcar e Adoçante",
        #                     "251": "Açúcar e Adoçante",
        #                     "252": "Açúcar e Adoçante",
        #                     "253": "Açúcar e Adoçante",
        #                     "254": "Biscoitos",
        #                     "255": "Biscoitos",
        #                     "256": "Biscoitos",
        #                     "257": "Biscoitos",
        #                     "258": "Biscoitos",
        #                     "259": "Biscoitos",
        #                     "260": "Biscoitos",
        #                     "261": "Cafés, Chás e Achocolatados",
        #                     "262": "Cafés, Chás e Achocolatados",
        #                     "263": "Cafés, Chás e Achocolatados",
        #                     "264": "Cafés, Chás e Achocolatados",
        #                     "265": "Cafés, Chás e Achocolatados",
        #                     "266": "Cafés, Chás e Achocolatados",
        #                     "267": "Cafés, Chás e Achocolatados",
        #                     "268": "Cervejas",
        #                     "269": "Cervejas",
        #                     "270": "Cervejas",
        #                     "271": "Cervejas",
        #                     "272": "Cervejas",
        #                     "273": "Cervejas",
        #                     "274": "Cervejas",
        #                     "275": "Cervejas Premium",
        #                     "276": "Cervejas Premium",
        #                     "277": "Cervejas Premium",
        #                     "278": "Cervejas Premium",
        #                     "279": "Cervejas Premium",
        #                     "280": "Cervejas Premium",
        #                     "281": "Cervejas Premium",
        #                     "282": "Derivados de Leite",
        #                     "283": "Derivados de Leite",
        #                     "284": "Derivados de Leite",
        #                     "285": "Derivados de Leite",
        #                     "286": "Derivados de Leite",
        #                     "287": "Derivados de Leite",
        #                     "288": "Derivados de Leite",
        #                     "289": "Grãos e Farináceos",
        #                     "290": "Grãos e Farináceos",
        #                     "291": "Grãos e Farináceos",
        #                     "292": "Grãos e Farináceos",
        #                     "293": "Grãos e Farináceos",
        #                     "294": "Grãos e Farináceos",
        #                     "295": "Grãos e Farináceos",
        #                     "296": "Leite",
        #                     "297": "Leite",
        #                     "298": "Leite",
        #                     "299": "Leite",
        #                     "300": "Leite",
        #                     "301": "Leite",
        #                     "302": "Leite",
        #                     "303": "Limpeza de Roupa",
        #                     "304": "Limpeza de Roupa",
        #                     "305": "Limpeza de Roupa",
        #                     "306": "Limpeza de Roupa",
        #                     "307": "Limpeza de Roupa",
        #                     "308": "Limpeza de Roupa",
        #                     "309": "Limpeza de Roupa",
        #                     "310": "Limpeza em Geral",
        #                     "311": "Limpeza em Geral",
        #                     "312": "Limpeza em Geral",
        #                     "313": "Limpeza em Geral",
        #                     "314": "Limpeza em Geral",
        #                     "315": "Limpeza em Geral",
        #                     "316": "Limpeza em Geral",
        #                     "317": "Massas Secas",
        #                     "318": "Massas Secas",
        #                     "319": "Massas Secas",
        #                     "320": "Massas Secas",
        #                     "321": "Massas Secas",
        #                     "322": "Massas Secas",
        #                     "323": "Massas Secas",
        #                     "324": "Refrigerantes",
        #                     "325": "Refrigerantes",
        #                     "326": "Refrigerantes",
        #                     "327": "Refrigerantes",
        #                     "328": "Refrigerantes",
        #                     "329": "Refrigerantes",
        #                     "330": "Refrigerantes",
        #                     "331": "Sucos E Refrescos",
        #                     "332": "Sucos E Refrescos",
        #                     "333": "Sucos E Refrescos",
        #                     "334": "Sucos E Refrescos",
        #                     "335": "Sucos E Refrescos",
        #                     "336": "Sucos E Refrescos",
        #                     "337": "Sucos E Refrescos",
        #                     "338": "Óleos, Azeites e Vinagres",
        #                     "339": "Óleos, Azeites e Vinagres",
        #                     "340": "Óleos, Azeites e Vinagres",
        #                     "341": "Óleos, Azeites e Vinagres",
        #                     "342": "Óleos, Azeites e Vinagres",
        #                     "343": "Óleos, Azeites e Vinagres",
        #                     "344": "Óleos, Azeites e Vinagres"
        #                 },
        #                 "Region": {
        #                     "0": "BAC",
        #                     "1": "BAC",
        #                     "2": "RJC",
        #                     "3": "RJC",
        #                     "4": "RJC",
        #                     "5": "RJI",
        #                     "6": "RJI",
        #                     "7": "RJI",
        #                     "8": "BAC",
        #                     "9": "BAC",
        #                     "10": "RJC",
        #                     "11": "RJC",
        #                     "12": "RJC",
        #                     "13": "RJI",
        #                     "14": "RJI",
        #                     "15": "RJI",
        #                     "16": "BAC",
        #                     "17": "BAC",
        #                     "18": "RJC",
        #                     "19": "RJC",
        #                     "20": "RJC",
        #                     "21": "RJI",
        #                     "22": "RJI",
        #                     "23": "RJI",
        #                     "24": "BAC",
        #                     "25": "BAC",
        #                     "26": "RJC",
        #                     "27": "RJC",
        #                     "28": "RJC",
        #                     "29": "RJI",
        #                     "30": "RJI",
        #                     "31": "RJI",
        #                     "32": "BAC",
        #                     "33": "BAC",
        #                     "34": "RJC",
        #                     "35": "RJC",
        #                     "36": "RJC",
        #                     "37": "RJI",
        #                     "38": "RJI",
        #                     "39": "RJI",
        #                     "40": "BAC",
        #                     "41": "BAC",
        #                     "42": "RJC",
        #                     "43": "RJC",
        #                     "44": "RJC",
        #                     "45": "RJI",
        #                     "46": "RJI",
        #                     "47": "RJI",
        #                     "48": "BAC",
        #                     "49": "BAC",
        #                     "50": "RJC",
        #                     "51": "RJC",
        #                     "52": "RJC",
        #                     "53": "RJI",
        #                     "54": "RJI",
        #                     "55": "RJI",
        #                     "56": "BAC",
        #                     "57": "BAC",
        #                     "58": "RJC",
        #                     "59": "RJC",
        #                     "60": "RJC",
        #                     "61": "RJI",
        #                     "62": "RJI",
        #                     "63": "RJI",
        #                     "64": "BAC",
        #                     "65": "BAC",
        #                     "66": "RJC",
        #                     "67": "RJC",
        #                     "68": "RJC",
        #                     "69": "RJI",
        #                     "70": "RJI",
        #                     "71": "RJI",
        #                     "72": "BAC",
        #                     "73": "BAC",
        #                     "74": "RJC",
        #                     "75": "RJC",
        #                     "76": "RJC",
        #                     "77": "RJI",
        #                     "78": "RJI",
        #                     "79": "RJI",
        #                     "80": "BAC",
        #                     "81": "BAC",
        #                     "82": "RJC",
        #                     "83": "RJC",
        #                     "84": "RJC",
        #                     "85": "RJI",
        #                     "86": "RJI",
        #                     "87": "RJI",
        #                     "88": "BAC",
        #                     "89": "BAC",
        #                     "90": "RJC",
        #                     "91": "RJC",
        #                     "92": "RJC",
        #                     "93": "RJI",
        #                     "94": "RJI",
        #                     "95": "RJI",
        #                     "96": "BAC",
        #                     "97": "BAC",
        #                     "98": "RJC",
        #                     "99": "RJC",
        #                     "100": "RJC",
        #                     "101": "RJI",
        #                     "102": "RJI",
        #                     "103": "RJI",
        #                     "104": "BAC",
        #                     "105": "BAC",
        #                     "106": "RJC",
        #                     "107": "RJC",
        #                     "108": "RJC",
        #                     "109": "RJI",
        #                     "110": "RJI",
        #                     "111": "RJI",
        #                     "112": "BAC",
        #                     "113": "BAC",
        #                     "114": "RJC",
        #                     "115": "RJC",
        #                     "116": "RJC",
        #                     "117": "RJI",
        #                     "118": "RJI",
        #                     "119": "RJI",
        #                     "120": "BAC",
        #                     "121": "BAC",
        #                     "122": "RJC",
        #                     "123": "RJC",
        #                     "124": "RJC",
        #                     "125": "RJI",
        #                     "126": "RJI",
        #                     "127": "RJI",
        #                     "128": "BAC",
        #                     "129": "BAC",
        #                     "130": "RJC",
        #                     "131": "RJC",
        #                     "132": "RJC",
        #                     "133": "RJI",
        #                     "134": "RJI",
        #                     "135": "RJI",
        #                     "136": "BAC",
        #                     "137": "BAC",
        #                     "138": "RJC",
        #                     "139": "RJC",
        #                     "140": "RJC",
        #                     "141": "RJI",
        #                     "142": "RJI",
        #                     "143": "RJI",
        #                     "144": "BAC",
        #                     "145": "BAC",
        #                     "146": "RJC",
        #                     "147": "RJC",
        #                     "148": "RJC",
        #                     "149": "RJI",
        #                     "150": "RJI",
        #                     "151": "RJI",
        #                     "152": "BAC",
        #                     "153": "BAC",
        #                     "154": "RJC",
        #                     "155": "RJC",
        #                     "156": "RJC",
        #                     "157": "RJI",
        #                     "158": "RJI",
        #                     "159": "RJI",
        #                     "160": "BAC",
        #                     "161": "BAC",
        #                     "162": "RJC",
        #                     "163": "RJC",
        #                     "164": "RJC",
        #                     "165": "RJI",
        #                     "166": "RJI",
        #                     "167": "RJI",
        #                     "168": "BAC",
        #                     "169": "BAC",
        #                     "170": "RJC",
        #                     "171": "RJC",
        #                     "172": "RJC",
        #                     "173": "RJI",
        #                     "174": "RJI",
        #                     "175": "RJI",
        #                     "176": "BAC",
        #                     "177": "BAC",
        #                     "178": "RJC",
        #                     "179": "RJC",
        #                     "180": "RJC",
        #                     "181": "RJI",
        #                     "182": "RJI",
        #                     "183": "RJI",
        #                     "184": "BAC",
        #                     "185": "BAC",
        #                     "186": "RJC",
        #                     "187": "RJC",
        #                     "188": "RJC",
        #                     "189": "RJI",
        #                     "190": "RJI",
        #                     "191": "RJI",
        #                     "192": "BAC",
        #                     "193": "BAC",
        #                     "194": "RJC",
        #                     "195": "RJC",
        #                     "196": "RJC",
        #                     "197": "RJI",
        #                     "198": "RJI",
        #                     "199": "RJI",
        #                     "200": "BAC",
        #                     "201": "BAC",
        #                     "202": "RJC",
        #                     "203": "RJC",
        #                     "204": "RJC",
        #                     "205": "RJI",
        #                     "206": "RJI",
        #                     "207": "RJI",
        #                     "208": "BAC",
        #                     "209": "BAC",
        #                     "210": "RJC",
        #                     "211": "RJC",
        #                     "212": "RJC",
        #                     "213": "RJI",
        #                     "214": "RJI",
        #                     "215": "RJI",
        #                     "216": "BAC",
        #                     "217": "BAC",
        #                     "218": "RJC",
        #                     "219": "RJC",
        #                     "220": "RJC",
        #                     "221": "RJI",
        #                     "222": "RJI",
        #                     "223": "RJI",
        #                     "224": "BAC",
        #                     "225": "BAC",
        #                     "226": "RJC",
        #                     "227": "RJC",
        #                     "228": "RJC",
        #                     "229": "RJI",
        #                     "230": "RJI",
        #                     "231": "RJI",
        #                     "232": "BAC",
        #                     "233": "BAC",
        #                     "234": "RJC",
        #                     "235": "RJC",
        #                     "236": "RJC",
        #                     "237": "RJI",
        #                     "238": "RJI",
        #                     "239": "RJI",
        #                     "240": "BAC",
        #                     "241": "BAC",
        #                     "242": "RJC",
        #                     "243": "RJC",
        #                     "244": "RJC",
        #                     "245": "RJI",
        #                     "246": "RJI",
        #                     "247": "BAC",
        #                     "248": "BAC",
        #                     "249": "RJC",
        #                     "250": "RJC",
        #                     "251": "RJC",
        #                     "252": "RJI",
        #                     "253": "RJI",
        #                     "254": "BAC",
        #                     "255": "BAC",
        #                     "256": "RJC",
        #                     "257": "RJC",
        #                     "258": "RJC",
        #                     "259": "RJI",
        #                     "260": "RJI",
        #                     "261": "BAC",
        #                     "262": "BAC",
        #                     "263": "RJC",
        #                     "264": "RJC",
        #                     "265": "RJC",
        #                     "266": "RJI",
        #                     "267": "RJI",
        #                     "268": "BAC",
        #                     "269": "BAC",
        #                     "270": "RJC",
        #                     "271": "RJC",
        #                     "272": "RJC",
        #                     "273": "RJI",
        #                     "274": "RJI",
        #                     "275": "BAC",
        #                     "276": "BAC",
        #                     "277": "RJC",
        #                     "278": "RJC",
        #                     "279": "RJC",
        #                     "280": "RJI",
        #                     "281": "RJI",
        #                     "282": "BAC",
        #                     "283": "BAC",
        #                     "284": "RJC",
        #                     "285": "RJC",
        #                     "286": "RJC",
        #                     "287": "RJI",
        #                     "288": "RJI",
        #                     "289": "BAC",
        #                     "290": "BAC",
        #                     "291": "RJC",
        #                     "292": "RJC",
        #                     "293": "RJC",
        #                     "294": "RJI",
        #                     "295": "RJI",
        #                     "296": "BAC",
        #                     "297": "BAC",
        #                     "298": "RJC",
        #                     "299": "RJC",
        #                     "300": "RJC",
        #                     "301": "RJI",
        #                     "302": "RJI",
        #                     "303": "BAC",
        #                     "304": "BAC",
        #                     "305": "RJC",
        #                     "306": "RJC",
        #                     "307": "RJC",
        #                     "308": "RJI",
        #                     "309": "RJI",
        #                     "310": "BAC",
        #                     "311": "BAC",
        #                     "312": "RJC",
        #                     "313": "RJC",
        #                     "314": "RJC",
        #                     "315": "RJI",
        #                     "316": "RJI",
        #                     "317": "BAC",
        #                     "318": "BAC",
        #                     "319": "RJC",
        #                     "320": "RJC",
        #                     "321": "RJC",
        #                     "322": "RJI",
        #                     "323": "RJI",
        #                     "324": "BAC",
        #                     "325": "BAC",
        #                     "326": "RJC",
        #                     "327": "RJC",
        #                     "328": "RJC",
        #                     "329": "RJI",
        #                     "330": "RJI",
        #                     "331": "BAC",
        #                     "332": "BAC",
        #                     "333": "RJC",
        #                     "334": "RJC",
        #                     "335": "RJC",
        #                     "336": "RJI",
        #                     "337": "RJI",
        #                     "338": "BAC",
        #                     "339": "BAC",
        #                     "340": "RJC",
        #                     "341": "RJC",
        #                     "342": "RJC",
        #                     "343": "RJI",
        #                     "344": "RJI"
        #                 },
        #                 "Size": {
        #                     "0": "1-4 Cxs",
        #                     "1": "size",
        #                     "2": "1-4 Cxs",
        #                     "3": "5-9 Cxs",
        #                     "4": "size",
        #                     "5": "1-4 Cxs",
        #                     "6": "5-9 Cxs",
        #                     "7": "size",
        #                     "8": "1-4 Cxs",
        #                     "9": "size",
        #                     "10": "1-4 Cxs",
        #                     "11": "5-9 Cxs",
        #                     "12": "size",
        #                     "13": "1-4 Cxs",
        #                     "14": "5-9 Cxs",
        #                     "15": "size",
        #                     "16": "1-4 Cxs",
        #                     "17": "size",
        #                     "18": "1-4 Cxs",
        #                     "19": "5-9 Cxs",
        #                     "20": "size",
        #                     "21": "1-4 Cxs",
        #                     "22": "5-9 Cxs",
        #                     "23": "size",
        #                     "24": "1-4 Cxs",
        #                     "25": "size",
        #                     "26": "1-4 Cxs",
        #                     "27": "5-9 Cxs",
        #                     "28": "size",
        #                     "29": "1-4 Cxs",
        #                     "30": "5-9 Cxs",
        #                     "31": "size",
        #                     "32": "1-4 Cxs",
        #                     "33": "size",
        #                     "34": "1-4 Cxs",
        #                     "35": "5-9 Cxs",
        #                     "36": "size",
        #                     "37": "1-4 Cxs",
        #                     "38": "5-9 Cxs",
        #                     "39": "size",
        #                     "40": "1-4 Cxs",
        #                     "41": "size",
        #                     "42": "1-4 Cxs",
        #                     "43": "5-9 Cxs",
        #                     "44": "size",
        #                     "45": "1-4 Cxs",
        #                     "46": "5-9 Cxs",
        #                     "47": "size",
        #                     "48": "1-4 Cxs",
        #                     "49": "size",
        #                     "50": "1-4 Cxs",
        #                     "51": "5-9 Cxs",
        #                     "52": "size",
        #                     "53": "1-4 Cxs",
        #                     "54": "5-9 Cxs",
        #                     "55": "size",
        #                     "56": "1-4 Cxs",
        #                     "57": "size",
        #                     "58": "1-4 Cxs",
        #                     "59": "5-9 Cxs",
        #                     "60": "size",
        #                     "61": "1-4 Cxs",
        #                     "62": "5-9 Cxs",
        #                     "63": "size",
        #                     "64": "1-4 Cxs",
        #                     "65": "size",
        #                     "66": "1-4 Cxs",
        #                     "67": "5-9 Cxs",
        #                     "68": "size",
        #                     "69": "1-4 Cxs",
        #                     "70": "5-9 Cxs",
        #                     "71": "size",
        #                     "72": "1-4 Cxs",
        #                     "73": "size",
        #                     "74": "1-4 Cxs",
        #                     "75": "5-9 Cxs",
        #                     "76": "size",
        #                     "77": "1-4 Cxs",
        #                     "78": "5-9 Cxs",
        #                     "79": "size",
        #                     "80": "1-4 Cxs",
        #                     "81": "size",
        #                     "82": "1-4 Cxs",
        #                     "83": "5-9 Cxs",
        #                     "84": "size",
        #                     "85": "1-4 Cxs",
        #                     "86": "5-9 Cxs",
        #                     "87": "size",
        #                     "88": "1-4 Cxs",
        #                     "89": "size",
        #                     "90": "1-4 Cxs",
        #                     "91": "5-9 Cxs",
        #                     "92": "size",
        #                     "93": "1-4 Cxs",
        #                     "94": "5-9 Cxs",
        #                     "95": "size",
        #                     "96": "1-4 Cxs",
        #                     "97": "size",
        #                     "98": "1-4 Cxs",
        #                     "99": "5-9 Cxs",
        #                     "100": "size",
        #                     "101": "1-4 Cxs",
        #                     "102": "5-9 Cxs",
        #                     "103": "size",
        #                     "104": "1-4 Cxs",
        #                     "105": "size",
        #                     "106": "1-4 Cxs",
        #                     "107": "5-9 Cxs",
        #                     "108": "size",
        #                     "109": "1-4 Cxs",
        #                     "110": "5-9 Cxs",
        #                     "111": "size",
        #                     "112": "1-4 Cxs",
        #                     "113": "size",
        #                     "114": "1-4 Cxs",
        #                     "115": "5-9 Cxs",
        #                     "116": "size",
        #                     "117": "1-4 Cxs",
        #                     "118": "5-9 Cxs",
        #                     "119": "size",
        #                     "120": "1-4 Cxs",
        #                     "121": "size",
        #                     "122": "1-4 Cxs",
        #                     "123": "5-9 Cxs",
        #                     "124": "size",
        #                     "125": "1-4 Cxs",
        #                     "126": "5-9 Cxs",
        #                     "127": "size",
        #                     "128": "1-4 Cxs",
        #                     "129": "size",
        #                     "130": "1-4 Cxs",
        #                     "131": "5-9 Cxs",
        #                     "132": "size",
        #                     "133": "1-4 Cxs",
        #                     "134": "5-9 Cxs",
        #                     "135": "size",
        #                     "136": "1-4 Cxs",
        #                     "137": "size",
        #                     "138": "1-4 Cxs",
        #                     "139": "5-9 Cxs",
        #                     "140": "size",
        #                     "141": "1-4 Cxs",
        #                     "142": "5-9 Cxs",
        #                     "143": "size",
        #                     "144": "1-4 Cxs",
        #                     "145": "size",
        #                     "146": "1-4 Cxs",
        #                     "147": "5-9 Cxs",
        #                     "148": "size",
        #                     "149": "1-4 Cxs",
        #                     "150": "5-9 Cxs",
        #                     "151": "size",
        #                     "152": "1-4 Cxs",
        #                     "153": "size",
        #                     "154": "1-4 Cxs",
        #                     "155": "5-9 Cxs",
        #                     "156": "size",
        #                     "157": "1-4 Cxs",
        #                     "158": "5-9 Cxs",
        #                     "159": "size",
        #                     "160": "1-4 Cxs",
        #                     "161": "size",
        #                     "162": "1-4 Cxs",
        #                     "163": "5-9 Cxs",
        #                     "164": "size",
        #                     "165": "1-4 Cxs",
        #                     "166": "5-9 Cxs",
        #                     "167": "size",
        #                     "168": "1-4 Cxs",
        #                     "169": "size",
        #                     "170": "1-4 Cxs",
        #                     "171": "5-9 Cxs",
        #                     "172": "size",
        #                     "173": "1-4 Cxs",
        #                     "174": "5-9 Cxs",
        #                     "175": "size",
        #                     "176": "1-4 Cxs",
        #                     "177": "size",
        #                     "178": "1-4 Cxs",
        #                     "179": "5-9 Cxs",
        #                     "180": "size",
        #                     "181": "1-4 Cxs",
        #                     "182": "5-9 Cxs",
        #                     "183": "size",
        #                     "184": "1-4 Cxs",
        #                     "185": "size",
        #                     "186": "1-4 Cxs",
        #                     "187": "5-9 Cxs",
        #                     "188": "size",
        #                     "189": "1-4 Cxs",
        #                     "190": "5-9 Cxs",
        #                     "191": "size",
        #                     "192": "1-4 Cxs",
        #                     "193": "size",
        #                     "194": "1-4 Cxs",
        #                     "195": "5-9 Cxs",
        #                     "196": "size",
        #                     "197": "1-4 Cxs",
        #                     "198": "5-9 Cxs",
        #                     "199": "size",
        #                     "200": "1-4 Cxs",
        #                     "201": "size",
        #                     "202": "1-4 Cxs",
        #                     "203": "5-9 Cxs",
        #                     "204": "size",
        #                     "205": "1-4 Cxs",
        #                     "206": "5-9 Cxs",
        #                     "207": "size",
        #                     "208": "1-4 Cxs",
        #                     "209": "size",
        #                     "210": "1-4 Cxs",
        #                     "211": "5-9 Cxs",
        #                     "212": "size",
        #                     "213": "1-4 Cxs",
        #                     "214": "5-9 Cxs",
        #                     "215": "size",
        #                     "216": "1-4 Cxs",
        #                     "217": "size",
        #                     "218": "1-4 Cxs",
        #                     "219": "5-9 Cxs",
        #                     "220": "size",
        #                     "221": "1-4 Cxs",
        #                     "222": "5-9 Cxs",
        #                     "223": "size",
        #                     "224": "1-4 Cxs",
        #                     "225": "size",
        #                     "226": "1-4 Cxs",
        #                     "227": "5-9 Cxs",
        #                     "228": "size",
        #                     "229": "1-4 Cxs",
        #                     "230": "5-9 Cxs",
        #                     "231": "size",
        #                     "232": "1-4 Cxs",
        #                     "233": "size",
        #                     "234": "1-4 Cxs",
        #                     "235": "5-9 Cxs",
        #                     "236": "size",
        #                     "237": "1-4 Cxs",
        #                     "238": "5-9 Cxs",
        #                     "239": "size",
        #                     "240": "1-4 Cxs",
        #                     "241": "size",
        #                     "242": "1-4 Cxs",
        #                     "243": "5-9 Cxs",
        #                     "244": "size",
        #                     "245": "1-4 Cxs",
        #                     "246": "size",
        #                     "247": "1-4 Cxs",
        #                     "248": "size",
        #                     "249": "1-4 Cxs",
        #                     "250": "5-9 Cxs",
        #                     "251": "size",
        #                     "252": "1-4 Cxs",
        #                     "253": "size",
        #                     "254": "1-4 Cxs",
        #                     "255": "size",
        #                     "256": "1-4 Cxs",
        #                     "257": "5-9 Cxs",
        #                     "258": "size",
        #                     "259": "1-4 Cxs",
        #                     "260": "size",
        #                     "261": "1-4 Cxs",
        #                     "262": "size",
        #                     "263": "1-4 Cxs",
        #                     "264": "5-9 Cxs",
        #                     "265": "size",
        #                     "266": "1-4 Cxs",
        #                     "267": "size",
        #                     "268": "1-4 Cxs",
        #                     "269": "size",
        #                     "270": "1-4 Cxs",
        #                     "271": "5-9 Cxs",
        #                     "272": "size",
        #                     "273": "1-4 Cxs",
        #                     "274": "size",
        #                     "275": "1-4 Cxs",
        #                     "276": "size",
        #                     "277": "1-4 Cxs",
        #                     "278": "5-9 Cxs",
        #                     "279": "size",
        #                     "280": "1-4 Cxs",
        #                     "281": "size",
        #                     "282": "1-4 Cxs",
        #                     "283": "size",
        #                     "284": "1-4 Cxs",
        #                     "285": "5-9 Cxs",
        #                     "286": "size",
        #                     "287": "1-4 Cxs",
        #                     "288": "size",
        #                     "289": "1-4 Cxs",
        #                     "290": "size",
        #                     "291": "1-4 Cxs",
        #                     "292": "5-9 Cxs",
        #                     "293": "size",
        #                     "294": "1-4 Cxs",
        #                     "295": "size",
        #                     "296": "1-4 Cxs",
        #                     "297": "size",
        #                     "298": "1-4 Cxs",
        #                     "299": "5-9 Cxs",
        #                     "300": "size",
        #                     "301": "1-4 Cxs",
        #                     "302": "size",
        #                     "303": "1-4 Cxs",
        #                     "304": "size",
        #                     "305": "1-4 Cxs",
        #                     "306": "5-9 Cxs",
        #                     "307": "size",
        #                     "308": "1-4 Cxs",
        #                     "309": "size",
        #                     "310": "1-4 Cxs",
        #                     "311": "size",
        #                     "312": "1-4 Cxs",
        #                     "313": "5-9 Cxs",
        #                     "314": "size",
        #                     "315": "1-4 Cxs",
        #                     "316": "size",
        #                     "317": "1-4 Cxs",
        #                     "318": "size",
        #                     "319": "1-4 Cxs",
        #                     "320": "5-9 Cxs",
        #                     "321": "size",
        #                     "322": "1-4 Cxs",
        #                     "323": "size",
        #                     "324": "1-4 Cxs",
        #                     "325": "size",
        #                     "326": "1-4 Cxs",
        #                     "327": "5-9 Cxs",
        #                     "328": "size",
        #                     "329": "1-4 Cxs",
        #                     "330": "size",
        #                     "331": "1-4 Cxs",
        #                     "332": "size",
        #                     "333": "1-4 Cxs",
        #                     "334": "5-9 Cxs",
        #                     "335": "size",
        #                     "336": "1-4 Cxs",
        #                     "337": "size",
        #                     "338": "1-4 Cxs",
        #                     "339": "size",
        #                     "340": "1-4 Cxs",
        #                     "341": "5-9 Cxs",
        #                     "342": "size",
        #                     "343": "1-4 Cxs",
        #                     "344": "size"
        #                 },
        #                 "Trend Atual": {
        #                     "0": 0.21283967225015882,
        #                     "1": 0.2128102377102472,
        #                     "2": 0.26350640999710917,
        #                     "3": 0.2701563460756816,
        #                     "4": 0.22588527407111897,
        #                     "5": 0.2979715256599122,
        #                     "6": 0.296849608616763,
        #                     "7": 0.2926490822720907,
        #                     "8": 0.13600270693449876,
        #                     "9": 0.13521692479197586,
        #                     "10": 0.33447091978831756,
        #                     "11": 0.39250306608780444,
        #                     "12": 0.3251552769160695,
        #                     "13": 0.20687921042625912,
        #                     "14": 0.31666733546462356,
        #                     "15": 0.2187614804119686,
        #                     "16": 0.2999822086004345,
        #                     "17": 0.30636703951121447,
        #                     "18": 0.4832017933711329,
        #                     "19": 0.4602619258882162,
        #                     "20": 0.45939000587914935,
        #                     "21": 0.5827554459908473,
        #                     "22": 0.5640905527933173,
        #                     "23": 0.582161485010254,
        #                     "24": 0.3304096970702537,
        #                     "25": 0.32780667178010314,
        #                     "26": 0.17695352507034307,
        #                     "27": 0.430405528253816,
        #                     "28": 0.19943637505957598,
        #                     "29": 0.2548660389647522,
        #                     "30": 0.546970027251221,
        #                     "31": 0.2548284742538801,
        #                     "32": 0.31407047958683737,
        #                     "33": 0.3140549349806476,
        #                     "34": 0.07609412156661616,
        #                     "35": 0.14028401527246745,
        #                     "36": 0.0665643572658971,
        #                     "37": 0.19780923756984028,
        #                     "38": 0.42097992355400066,
        #                     "39": 0.22308234350187156,
        #                     "40": 0.11918973651706513,
        #                     "41": 0.1192300348455539,
        #                     "42": 0.04489105057001433,
        #                     "43": 0.14968636827726348,
        #                     "44": 0.03849959232946084,
        #                     "45": 0.08145391950554819,
        #                     "46": 0.2222462708875105,
        #                     "47": 0.07343436410616018,
        #                     "48": 0.5413940961864712,
        #                     "49": 0.5415632623771585,
        #                     "50": 0.19984364453924355,
        #                     "51": 0.3363372255726181,
        #                     "52": 0.20479331335082285,
        #                     "53": 0.13152818764744278,
        #                     "54": 0.5522955170122629,
        #                     "55": 0.09576285833690361,
        #                     "56": 0.17906514929173128,
        #                     "57": 0.18167220195075673,
        #                     "58": 0.17682944715299526,
        #                     "59": 0.20332816538864693,
        #                     "60": 0.21961456160269832,
        #                     "61": 0.20337091039301572,
        #                     "62": 0.49688400660822307,
        #                     "63": 0.18895939997932792,
        #                     "64": 0.33003404914241447,
        #                     "65": 0.33035477166893185,
        #                     "66": 0.4396037881215084,
        #                     "67": 0.5240032846193501,
        #                     "68": 0.4305474985859405,
        #                     "69": 0.3780672763296619,
        #                     "70": 0.719478273483059,
        #                     "71": 0.37988107726339443,
        #                     "72": 0.22970437608101058,
        #                     "73": 0.2341194220167455,
        #                     "74": 0.2575929379059132,
        #                     "75": 0.1425976250386788,
        #                     "76": 0.20926315434892287,
        #                     "77": 0.08663078151021289,
        #                     "78": 0.424524741095391,
        #                     "79": 0.07757117221253809,
        #                     "80": 0.24815212298265682,
        #                     "81": 0.24776771194610708,
        #                     "82": 0.20510876405214667,
        #                     "83": 0.21026356677478342,
        #                     "84": 0.20738918941033993,
        #                     "85": 0.20071044218361966,
        #                     "86": 0.42994187609594287,
        #                     "87": 0.1611579362897704,
        #                     "88": 0.2882015145466819,
        #                     "89": 0.27825410979864756,
        #                     "90": 0.17279645765354013,
        #                     "91": 0.2542494185940949,
        #                     "92": 0.35012388255988025,
        #                     "93": 0.23346488997480971,
        #                     "94": 0.4830484047121416,
        #                     "95": 0.20533106861561748,
        #                     "96": 0.25554878367921197,
        #                     "97": 0.2534613055067756,
        #                     "98": 0.22552379356114763,
        #                     "99": 0.18287288535616916,
        #                     "100": 0.21760827663875193,
        #                     "101": 0.24408664518561102,
        #                     "102": 0.4756714089024071,
        #                     "103": 0.2309906697313801,
        #                     "104": 0.29066369933829045,
        #                     "105": 0.29923059657328926,
        #                     "106": 0.12806130710074773,
        #                     "107": 0.24803316984425153,
        #                     "108": 0.11943419145566032,
        #                     "109": 0.04457988259825843,
        #                     "110": 0.27830066920435764,
        #                     "111": 0,
        #                     "112": 0.1719501345061776,
        #                     "113": 0.171562927474356,
        #                     "114": 0.21592397608869673,
        #                     "115": 0.3229012312711734,
        #                     "116": 0.22461551265521865,
        #                     "117": 0.3540133072737043,
        #                     "118": 0.3167789242686465,
        #                     "119": 0.3150307563244753,
        #                     "120": 0.20175459502164594,
        #                     "121": 0.20697478532832894,
        #                     "122": 0.2495010267837813,
        #                     "123": 0.26438793396875687,
        #                     "124": 0.2430178103694681,
        #                     "125": 0.26829082468547416,
        #                     "126": 0.296849608616763,
        #                     "127": 0.22361879201108267,
        #                     "128": 0.1681220198588986,
        #                     "129": 0.16783012475988815,
        #                     "130": 0.3848137963381494,
        #                     "131": 0.4120929834402589,
        #                     "132": 0.36640403914709485,
        #                     "133": 0.36307594704333346,
        #                     "134": 0.31666733546462356,
        #                     "135": 0.4522542685764868,
        #                     "136": 0.3684418323196522,
        #                     "137": 0.36602989104780714,
        #                     "138": 0.31813906930422065,
        #                     "139": 0.49503913377773495,
        #                     "140": 0.33884119766018234,
        #                     "141": 0.5092509802772571,
        #                     "142": 0.5640905527933173,
        #                     "143": 0.45387151380902874,
        #                     "144": 0.3763253404124172,
        #                     "145": 0.37805388750121866,
        #                     "146": 0.27937488509883474,
        #                     "147": 0.30393008148445044,
        #                     "148": 0.2757102488884806,
        #                     "149": 0.2525906417493689,
        #                     "150": 0.546970027251221,
        #                     "151": 0.25617055631886615,
        #                     "152": 0.32680114555923684,
        #                     "153": 0.3331017924968156,
        #                     "154": 0.1250562341022588,
        #                     "155": 0.15096540999886568,
        #                     "156": 0.12717321299062603,
        #                     "157": 0.16575745033224734,
        #                     "158": 0.42097992355400066,
        #                     "159": 0.14713646604861075,
        #                     "160": 0.17534144432898935,
        #                     "161": 0.17418157737433684,
        #                     "162": 0.04607231416629753,
        #                     "163": 0.1437765034423028,
        #                     "164": 0.07341460794060584,
        #                     "165": 0.056039600030374635,
        #                     "166": 0.2222462708875105,
        #                     "167": 0.03074550106806631,
        #                     "168": 0.4570997430927968,
        #                     "169": 0.4567599980313648,
        #                     "170": 0.2851319553792471,
        #                     "171": 0.3381604905298988,
        #                     "172": 0.2835469621470778,
        #                     "173": 0.44040425601534094,
        #                     "174": 0.5522955170122629,
        #                     "175": 0.3382665078369224,
        #                     "176": 0.18232900012451478,
        #                     "177": 0.18334966006905468,
        #                     "178": 0.24913875248820913,
        #                     "179": 0.17464668654545457,
        #                     "180": 0.20928928876733013,
        #                     "181": 0.22073968438261662,
        #                     "182": 0.49688400660822307,
        #                     "183": 0.21385104699395338,
        #                     "184": 0.268176143151152,
        #                     "185": 0.2744596680666146,
        #                     "186": 0.433867259687314,
        #                     "187": 0.540103578169033,
        #                     "188": 0.4417680446519972,
        #                     "189": 0.3654224390519498,
        #                     "190": 0.719478273483059,
        #                     "191": 0.3508126395947101,
        #                     "192": 0.209133216172427,
        #                     "193": 0.2022720282776591,
        #                     "194": 0.11517957481976562,
        #                     "195": 0.12912821477780753,
        #                     "196": 0.10214296201598065,
        #                     "197": 0.19190302438464024,
        #                     "198": 0.424524741095391,
        #                     "199": 0.18776267011184108,
        #                     "200": 0.2830655096410057,
        #                     "201": 0.2829922673411062,
        #                     "202": 0.2118967522972051,
        #                     "203": 0.21099673741047867,
        #                     "204": 0.19399957310892427,
        #                     "205": 0.22539294321441117,
        #                     "206": 0.42994187609594287,
        #                     "207": 0.20749671138836948,
        #                     "208": 0.213096123446751,
        #                     "209": 0.21291615085197796,
        #                     "210": 0.2506586745768572,
        #                     "211": 0.2094446977043573,
        #                     "212": 0.22647927699607753,
        #                     "213": 0.25707928985420364,
        #                     "214": 0.4830484047121416,
        #                     "215": 0.20470787308785884,
        #                     "216": 0.26505132101380435,
        #                     "217": 0.2647174490740657,
        #                     "218": 0.19516493658896855,
        #                     "219": 0.1656287482677661,
        #                     "220": 0.1737821065986529,
        #                     "221": 0.26000108991645626,
        #                     "222": 0.4756714089024071,
        #                     "223": 0.2930872592650788,
        #                     "224": 0.34610969430739597,
        #                     "225": 0.34783162767334,
        #                     "226": 0.28182283801653396,
        #                     "227": 0.2625898147819719,
        #                     "228": 0.2710438255312445,
        #                     "229": 0.26515027119161394,
        #                     "230": 0.27830066920435764,
        #                     "231": 0.2889728008865883,
        #                     "232": 0.2321160151776738,
        #                     "233": 0.2316419886303517,
        #                     "234": 0.19075882422582932,
        #                     "235": 0.30529643312175814,
        #                     "236": 0.19326523077205718,
        #                     "237": 0.3643386446469434,
        #                     "238": 0.3167789242686465,
        #                     "239": 0.25901234149286695,
        #                     "240": 0.2166578803818997,
        #                     "241": 0.2175805160111307,
        #                     "242": 0.23436018466715794,
        #                     "243": 0.05429166505535478,
        #                     "244": 0.21853019713802097,
        #                     "245": 0.24697552856803795,
        #                     "246": 0.23759180983120462,
        #                     "247": 0.2314300680710237,
        #                     "248": 0.23142891616647754,
        #                     "249": 0.39362180000643254,
        #                     "250": 0.4184364702671385,
        #                     "251": 0.3662571188228474,
        #                     "252": 0.4749137696298486,
        #                     "253": 0.4003844679345431,
        #                     "254": 0.3250532444059925,
        #                     "255": 0.3261505087113196,
        #                     "256": 0.2750050698134695,
        #                     "257": 0.39318627560518304,
        #                     "258": 0.3048355124505147,
        #                     "259": 0.43786922130556893,
        #                     "260": 0.4143682699990997,
        #                     "261": 0.35069762870594245,
        #                     "262": 0.3481167423646757,
        #                     "263": 0.2707054810605497,
        #                     "264": 0.24812821213059633,
        #                     "265": 0.27438489589105614,
        #                     "266": 0.23693799579039093,
        #                     "267": 0.2593953647869301,
        #                     "268": 0.34009945105926026,
        #                     "269": 0.33985031499703294,
        #                     "270": 0.13238529357879378,
        #                     "271": 0.14283949833341125,
        #                     "272": 0.11261623304884241,
        #                     "273": 0.12345274193394071,
        #                     "274": 0.09864903306224435,
        #                     "275": 0.17723530946553695,
        #                     "276": 0.17689578455544555,
        #                     "277": 0.06260957345998687,
        #                     "278": 0.06347721339576583,
        #                     "279": 0.08375453286475816,
        #                     "280": 0.11044875193529269,
        #                     "281": 0.09300193753935851,
        #                     "282": 0.41793396437890773,
        #                     "283": 0.4188125126369136,
        #                     "284": 0.29485195574333206,
        #                     "285": 0.30575427858951476,
        #                     "286": 0.28073411492182476,
        #                     "287": 0.48039353056600254,
        #                     "288": 0.44304853867460586,
        #                     "289": 0.23128354890509661,
        #                     "290": 0.23143572258114958,
        #                     "291": 0.26166031743077106,
        #                     "292": 0.15476273138784558,
        #                     "293": 0.24320109439383367,
        #                     "294": 0.15115079831914663,
        #                     "295": 0.19968275887085132,
        #                     "296": 0.29175007965123173,
        #                     "297": 0.29226664927207163,
        #                     "298": 0.4382981860390268,
        #                     "299": 0.39222220399932756,
        #                     "300": 0.4151533025340636,
        #                     "301": 0.3740831888745702,
        #                     "302": 0.3578251953727126,
        #                     "303": 0.25249766401032553,
        #                     "304": 0.2529088081647477,
        #                     "305": 0.14035957098914886,
        #                     "306": 0.1003031296327048,
        #                     "307": 0.1198765195013511,
        #                     "308": 0.17038860964996005,
        #                     "309": 0.12540578592244112,
        #                     "310": 0.23821877950514736,
        #                     "311": 0.23967710360697997,
        #                     "312": 0.19683851930969085,
        #                     "313": 0.1904758625393218,
        #                     "314": 0.19795786702250262,
        #                     "315": 0.27050527599986535,
        #                     "316": 0.2578024242540844,
        #                     "317": 0.22812869686160775,
        #                     "318": 0.23835946194417526,
        #                     "319": 0.2872176816162929,
        #                     "320": 0.11941426171120213,
        #                     "321": 0.23245210001409156,
        #                     "322": 0.18297004815952808,
        #                     "323": 0.25177544345712954,
        #                     "324": 0.3814022039332815,
        #                     "325": 0.3816324778094203,
        #                     "326": 0.22648469849465414,
        #                     "327": 0.09787497246166062,
        #                     "328": 0.20433131014234365,
        #                     "329": 0.2626890828686319,
        #                     "330": 0.24051000430486594,
        #                     "331": 0.2926520852922473,
        #                     "332": 0.29030566787733303,
        #                     "333": 0.315521059328352,
        #                     "334": 0.3280772547585458,
        #                     "335": 0.29095261073545897,
        #                     "336": 0.2593081974738734,
        #                     "337": 0.22723552878836176,
        #                     "338": 0.234089362875312,
        #                     "339": 0.23351176423945352,
        #                     "340": 0.3787006636081367,
        #                     "341": 0.41959317762153797,
        #                     "342": 0.36648549643363326,
        #                     "343": 0.6604727357740372,
        #                     "344": 0.5627083319723457
        #                 },
        #                 "% Target Positivação": {
        #                     "0": 0.24121569460007783,
        #                     "1": 0.2410395475256433,
        #                     "2": 0.2873652838677233,
        #                     "3": 0.6834086742880926,
        #                     "4": 0.2982846191347278,
        #                     "5": 0.40312200496933387,
        #                     "6": 0.6034492335503208,
        #                     "7": 0.37657587479113785,
        #                     "8": 0.1541420319336395,
        #                     "9": 0.15425028254547174,
        #                     "10": 0.40816037004386413,
        #                     "11": 0.43537286561283717,
        #                     "12": 0.3695122720380119,
        #                     "13": 0.5001876012436258,
        #                     "14": 0.619257064563938,
        #                     "15": 0.5315122313262813,
        #                     "16": 0.2999822086004345,
        #                     "17": 0.3085019919702773,
        #                     "18": 0.3964248572282161,
        #                     "19": 0.5397196455885527,
        #                     "20": 0.3595977882575496,
        #                     "21": 0.6073610428526613,
        #                     "22": 0.6603801863299041,
        #                     "23": 0.6071210697271965,
        #                     "24": 0.35293041076242837,
        #                     "25": 0.33676105873232515,
        #                     "26": 0.3074587275944513,
        #                     "27": 0.430405528253816,
        #                     "28": 0.30554991867480263,
        #                     "29": 0.5035275140754416,
        #                     "30": 0.6258248839254658,
        #                     "31": 0.5005921342606237,
        #                     "32": 0.4282248097362146,
        #                     "33": 0.4285439961948455,
        #                     "34": 0.17065671647077224,
        #                     "35": 0.296914492015669,
        #                     "36": 0.1655156619481438,
        #                     "37": 0.23401021253056054,
        #                     "38": 0.4296473329484008,
        #                     "39": 0.23377782812906678,
        #                     "40": 0.14007586843051356,
        #                     "41": 0.13983594182501394,
        #                     "42": 0.06042578500323643,
        #                     "43": 0.18973206420582153,
        #                     "44": 0.057396966021127156,
        #                     "45": 0.14646934208317094,
        #                     "46": 0.42044917784098074,
        #                     "47": 0.14176885016948906,
        #                     "48": 0.5413940961864712,
        #                     "49": 0.5415632623771585,
        #                     "50": 0.38187294132568134,
        #                     "51": 0.5195358325704842,
        #                     "52": 0.3651040635822726,
        #                     "53": 0.6140550398855394,
        #                     "54": 0.6259879818578182,
        #                     "55": 0.6029035720131478,
        #                     "56": 0.19199343696201127,
        #                     "57": 0.1916734649826006,
        #                     "58": 0.25081391681656173,
        #                     "59": 0.23931030004620388,
        #                     "60": 0.24080676009768426,
        #                     "61": 0.2630025369620815,
        #                     "62": 0.49688400660822307,
        #                     "63": 0.2536113256540201,
        #                     "64": 0.503445601101825,
        #                     "65": 0.5034177581248646,
        #                     "66": 0.4353348901128276,
        #                     "67": 0.5488023863318318,
        #                     "68": 0.4142998515361242,
        #                     "69": 0.3618737003280747,
        #                     "70": 0.6352029218379573,
        #                     "71": 0.34334212198146563,
        #                     "72": 0.22970437608101058,
        #                     "73": 0.2341194220167455,
        #                     "74": 0.23709486029780802,
        #                     "75": 0.30586073795986846,
        #                     "76": 0.24445533801595237,
        #                     "77": 0.2730707584672399,
        #                     "78": 0.49965361895593197,
        #                     "79": 0.26400910623056506,
        #                     "80": 0.32186237556998,
        #                     "81": 0.3308354651643145,
        #                     "82": 0.24175490157803298,
        #                     "83": 0.26674758168692114,
        #                     "84": 0.22925826664547133,
        #                     "85": 0.23968799736472451,
        #                     "86": 0.5068950667093006,
        #                     "87": 0.22202379413462098,
        #                     "88": 0.21083571875337398,
        #                     "89": 0.22818955416147696,
        #                     "90": 0.24008869426873103,
        #                     "91": 0.34634953981223987,
        #                     "92": 0.26932023741291816,
        #                     "93": 0.3248006077991298,
        #                     "94": 0.4824055318529098,
        #                     "95": 0.3078347945502027,
        #                     "96": 0.31786103579474784,
        #                     "97": 0.31745698261460553,
        #                     "98": 0.23290472625062505,
        #                     "99": 0.18750776421647167,
        #                     "100": 0.21017174154114934,
        #                     "101": 0.2515154953115813,
        #                     "102": 0.5106800749241995,
        #                     "103": 0.31020130827396414,
        #                     "104": 0.2460261628543306,
        #                     "105": 0.23388594336538712,
        #                     "106": 0.31343890300445093,
        #                     "107": 0.4744593983894059,
        #                     "108": 0.3814848225520794,
        #                     "109": 0.32319189318628777,
        #                     "110": 0.49731799911829483,
        #                     "111": 0.34086007309409994,
        #                     "112": 0.1719501345061776,
        #                     "113": 0.17064271673045736,
        #                     "114": 0.27558913300211285,
        #                     "115": 0.3720249888908496,
        #                     "116": 0.2606957676786722,
        #                     "117": 0.3540133072737043,
        #                     "118": 0.5196688334702608,
        #                     "119": 0.3284242297086908,
        #                     "120": 0.2304676214612032,
        #                     "121": 0.2299845715638623,
        #                     "122": 0.306438090257981,
        #                     "123": 0.5226090252032816,
        #                     "124": 0.29759649184651593,
        #                     "125": 0.3656621859692439,
        #                     "126": 0.6034492335503208,
        #                     "127": 0.33837784222307654,
        #                     "128": 0.1888334324355557,
        #                     "129": 0.18795634950676893,
        #                     "130": 0.4159694989704609,
        #                     "131": 0.42406297682580507,
        #                     "132": 0.3915268265628783,
        #                     "133": 0.4838022238981832,
        #                     "134": 0.619257064563938,
        #                     "135": 0.4522542685764868,
        #                     "136": 0.2978900665753283,
        #                     "137": 0.3020517253234977,
        #                     "138": 0.3689410619917076,
        #                     "139": 0.5275360906831063,
        #                     "140": 0.36679870049525054,
        #                     "141": 0.4135456422843813,
        #                     "142": 0.6603801863299041,
        #                     "143": 0.41276687012542085,
        #                     "144": 0.34514686178341164,
        #                     "145": 0.34359738702597487,
        #                     "146": 0.27937488509883474,
        #                     "147": 0.35957036598092107,
        #                     "148": 0.2749929803227029,
        #                     "149": 0.2802220770055583,
        #                     "150": 0.6258248839254658,
        #                     "151": 0.2636812059377493,
        #                     "152": 0.39309059349441144,
        #                     "153": 0.4060795569552629,
        #                     "154": 0.1250562341022588,
        #                     "155": 0.24907583067546177,
        #                     "156": 0.12174120908797903,
        #                     "157": 0.16575745033224734,
        #                     "158": 0.4296473329484008,
        #                     "159": 0.14713646604861075,
        #                     "160": 0.1526726367834974,
        #                     "161": 0.15072227105471203,
        #                     "162": 0.06704132035759852,
        #                     "163": 0.1761719259639478,
        #                     "164": 0.06754183881450311,
        #                     "165": 0.10482900568665905,
        #                     "166": 0.42044917784098074,
        #                     "167": 0.09865637656116547,
        #                     "168": 0.44030087369812315,
        #                     "169": 0.4399352252778456,
        #                     "170": 0.3668762336539977,
        #                     "171": 0.4086963422787613,
        #                     "172": 0.3510505142652546,
        #                     "173": 0.43810480296094867,
        #                     "174": 0.6259879818578182,
        #                     "175": 0.422964275881174,
        #                     "176": 0.20885652635340032,
        #                     "177": 0.20882652085720416,
        #                     "178": 0.2555430169881371,
        #                     "179": 0.24210367780464737,
        #                     "180": 0.24839188506230495,
        #                     "181": 0.2756398435006981,
        #                     "182": 0.49688400660822307,
        #                     "183": 0.2607050759368876,
        #                     "184": 0.35596982310043646,
        #                     "185": 0.36233497208823184,
        #                     "186": 0.46475198797088413,
        #                     "187": 0.5485032827595079,
        #                     "188": 0.4101533201698609,
        #                     "189": 0.3546827052365166,
        #                     "190": 0.6352029218379573,
        #                     "191": 0.3548271161622965,
        #                     "192": 0.25747910704706245,
        #                     "193": 0.25431006028003617,
        #                     "194": 0.20669257307165226,
        #                     "195": 0.2825010073385714,
        #                     "196": 0.1825327921743755,
        #                     "197": 0.20440773540462526,
        #                     "198": 0.49965361895593197,
        #                     "199": 0.18776267011184108,
        #                     "200": 0.30832953087875087,
        #                     "201": 0.30789822222741564,
        #                     "202": 0.2534901351248939,
        #                     "203": 0.2663254134130955,
        #                     "204": 0.24085469633503168,
        #                     "205": 0.2438340357103671,
        #                     "206": 0.5068950667093006,
        #                     "207": 0.23519377820035706,
        #                     "208": 0.19801152901144026,
        #                     "209": 0.20210800890601807,
        #                     "210": 0.25740427485759215,
        #                     "211": 0.34131996953556454,
        #                     "212": 0.2392785433420683,
        #                     "213": 0.25707928985420364,
        #                     "214": 0.4824055318529098,
        #                     "215": 0.2944712168699706,
        #                     "216": 0.3311880896996357,
        #                     "217": 0.33122389137173636,
        #                     "218": 0.22872423896891653,
        #                     "219": 0.1883091726443404,
        #                     "220": 0.2035567499726284,
        #                     "221": 0.22100500056300457,
        #                     "222": 0.5106800749241995,
        #                     "223": 0.25670653425365897,
        #                     "224": 0.28716856373765615,
        #                     "225": 0.29043456499510123,
        #                     "226": 0.2936599792483601,
        #                     "227": 0.3153964828044519,
        #                     "228": 0.28319732737138836,
        #                     "229": 0.3058927288842313,
        #                     "230": 0.49731799911829483,
        #                     "231": 0.2955308675597436,
        #                     "232": 0.18060067837690436,
        #                     "233": 0.1812633415562584,
        #                     "234": 0.2803614326439366,
        #                     "235": 0.3442039748613572,
        #                     "236": 0.2720385444713963,
        #                     "237": 0.3931520213942957,
        #                     "238": 0.5196688334702608,
        #                     "239": 0.38359741320543517,
        #                     "240": 0.24701039982706474,
        #                     "241": 0.24523364286187466,
        #                     "242": 0.3124223135673626,
        #                     "243": 0.2780054387753628,
        #                     "244": 0.26132156188486405,
        #                     "245": 0.3142264917495915,
        #                     "246": 0.3019173754909061,
        #                     "247": 0.36680925768576017,
        #                     "248": 0.3658473167871803,
        #                     "249": 0.4452857682917552,
        #                     "250": 0.360703580210814,
        #                     "251": 0.41823116931880283,
        #                     "252": 0.4749137696298486,
        #                     "253": 0.4814244247888441,
        #                     "254": 0.29146999265040063,
        #                     "255": 0.2948638031954366,
        #                     "256": 0.3482220144868559,
        #                     "257": 0.43843047012302494,
        #                     "258": 0.3558871168259564,
        #                     "259": 0.43786922130556893,
        #                     "260": 0.45692264891937356,
        #                     "261": 0.35069762870594245,
        #                     "262": 0.3481167423646757,
        #                     "263": 0.27789350927974926,
        #                     "264": 0.24812821213059633,
        #                     "265": 0.27438489589105614,
        #                     "266": 0.2847855965989612,
        #                     "267": 0.2593953647869301,
        #                     "268": 0.4064297956958157,
        #                     "269": 0.40535184781948197,
        #                     "270": 0.13238529357879378,
        #                     "271": 0.13460714626293788,
        #                     "272": 0.11261623304884241,
        #                     "273": 0.16240530406759768,
        #                     "274": 0.1513863733878283,
        #                     "275": 0.2235039800484648,
        #                     "276": 0.22320353387971645,
        #                     "277": 0.07130459320741245,
        #                     "278": 0.11514214352044574,
        #                     "279": 0.06217295736400658,
        #                     "280": 0.11044875193529269,
        #                     "281": 0.11097567435123208,
        #                     "282": 0.45550295941852587,
        #                     "283": 0.4574480596046775,
        #                     "284": 0.3667491628206408,
        #                     "285": 0.30575427858951476,
        #                     "286": 0.327600265882592,
        #                     "287": 0.43697811360377414,
        #                     "288": 0.41210895437444933,
        #                     "289": 0.34565546009980025,
        #                     "290": 0.34492940682258566,
        #                     "291": 0.29060288204109624,
        #                     "292": 0.2092795405454485,
        #                     "293": 0.27367836452488553,
        #                     "294": 0.2714548876196249,
        #                     "295": 0.2781398120430223,
        #                     "296": 0.3749553138094118,
        #                     "297": 0.3752911240349073,
        #                     "298": 0.5322978248715884,
        #                     "299": 0.4991515276537709,
        #                     "300": 0.4971214358134649,
        #                     "301": 0.5907570436427616,
        #                     "302": 0.594263703006119,
        #                     "303": 0.27714620884430535,
        #                     "304": 0.27500252671254155,
        #                     "305": 0.20047085320543026,
        #                     "306": 0.3192207591905116,
        #                     "307": 0.1842069164136853,
        #                     "308": 0.20049088455480818,
        #                     "309": 0.20652168288601258,
        #                     "310": 0.3424196207261294,
        #                     "311": 0.3419807882837219,
        #                     "312": 0.24410735444481052,
        #                     "313": 0.13688095923774077,
        #                     "314": 0.2084899227385989,
        #                     "315": 0.32068152514552317,
        #                     "316": 0.33261082962548805,
        #                     "317": 0.21822766486453613,
        #                     "318": 0.21663798054289485,
        #                     "319": 0.2772926196752578,
        #                     "320": 0.32548091067628265,
        #                     "321": 0.2441573832711373,
        #                     "322": 0.3238402969666319,
        #                     "323": 0.26708669888754744,
        #                     "324": 0.35242842185907397,
        #                     "325": 0.35229121267717334,
        #                     "326": 0.22648469849465414,
        #                     "327": 0.09819125342986658,
        #                     "328": 0.20433131014234365,
        #                     "329": 0.2523320687532098,
        #                     "330": 0.22718344051150482,
        #                     "331": 0.3013100766975016,
        #                     "332": 0.32688816807190557,
        #                     "333": 0.315521059328352,
        #                     "334": 0.31149365046690747,
        #                     "335": 0.29095261073545897,
        #                     "336": 0.39755025437557934,
        #                     "337": 0.33710960957742453,
        #                     "338": 0.23545919401800988,
        #                     "339": 0.23802480743107074,
        #                     "340": 0.3663347700114191,
        #                     "341": 0.3395549876256303,
        #                     "342": 0.36648549643363326,
        #                     "343": 0.5742939111505023,
        #                     "344": 0.5627083319723457
        #                 },
        #                 "Delta Top": {
        #                     "0": -0.02837602234991901,
        #                     "1": -0.028229309815396092,
        #                     "2": -0.02385887387061414,
        #                     "3": -0.413252328212411,
        #                     "4": -0.07239934506360884,
        #                     "5": -0.10515047930942167,
        #                     "6": -0.30659962493355775,
        #                     "7": -0.08392679251904717,
        #                     "8": -0.018139324999140755,
        #                     "9": -0.019033357753495878,
        #                     "10": -0.07368945025554657,
        #                     "11": -0.042869799525032726,
        #                     "12": -0.04435699512194241,
        #                     "13": -0.2933083908173667,
        #                     "14": -0.3025897290993145,
        #                     "15": -0.3127507509143127,
        #                     "16": 0,
        #                     "17": -0.002134952459062811,
        #                     "18": 0,
        #                     "19": -0.0794577197003365,
        #                     "20": 0,
        #                     "21": -0.024605596861814072,
        #                     "22": -0.09628963353658682,
        #                     "23": -0.024959584716942484,
        #                     "24": -0.022520713692174676,
        #                     "25": -0.008954386952222004,
        #                     "26": -0.13050520252410824,
        #                     "27": 0,
        #                     "28": -0.10611354361522665,
        #                     "29": -0.24866147511068942,
        #                     "30": -0.07885485667424474,
        #                     "31": -0.24576366000674355,
        #                     "32": -0.1141543301493772,
        #                     "33": -0.11448906121419788,
        #                     "34": -0.09456259490415608,
        #                     "35": -0.15663047674320155,
        #                     "36": -0.0989513046822467,
        #                     "37": -0.03620097496072025,
        #                     "38": -0.008667409394400138,
        #                     "39": -0.010695484627195218,
        #                     "40": -0.02088613191344843,
        #                     "41": -0.020605906979460034,
        #                     "42": -0.015534734433222099,
        #                     "43": -0.040045695928558056,
        #                     "44": -0.018897373691666317,
        #                     "45": -0.06501542257762276,
        #                     "46": -0.19820290695347023,
        #                     "47": -0.06833448606332888,
        #                     "48": 0,
        #                     "49": 0,
        #                     "50": -0.18202929678643778,
        #                     "51": -0.1831986069978661,
        #                     "52": -0.16031075023144975,
        #                     "53": -0.48252685223809666,
        #                     "54": -0.07369246484555536,
        #                     "55": -0.5071407136762442,
        #                     "56": -0.012928287670279981,
        #                     "57": -0.010001263031843871,
        #                     "58": -0.07398446966356648,
        #                     "59": -0.03598213465755695,
        #                     "60": -0.021192198494985937,
        #                     "61": -0.05963162656906576,
        #                     "62": 0,
        #                     "63": -0.06465192567469219,
        #                     "64": -0.17341155195941055,
        #                     "65": -0.1730629864559327,
        #                     "66": 0,
        #                     "67": -0.024799101712481675,
        #                     "68": 0,
        #                     "69": 0,
        #                     "70": 0,
        #                     "71": 0,
        #                     "72": 0,
        #                     "73": 0,
        #                     "74": 0,
        #                     "75": -0.16326311292118967,
        #                     "76": -0.0351921836670295,
        #                     "77": -0.186439976957027,
        #                     "78": -0.07512887786054095,
        #                     "79": -0.18643793401802697,
        #                     "80": -0.07371025258732317,
        #                     "81": -0.08306775321820742,
        #                     "82": -0.0366461375258863,
        #                     "83": -0.056484014912137726,
        #                     "84": -0.0218690772351314,
        #                     "85": -0.03897755518110485,
        #                     "86": -0.07695319061335776,
        #                     "87": -0.060865857844850574,
        #                     "88": 0,
        #                     "89": 0,
        #                     "90": -0.0672922366151909,
        #                     "91": -0.09210012121814498,
        #                     "92": 0,
        #                     "93": -0.09133571782432007,
        #                     "94": 0,
        #                     "95": -0.1025037259345852,
        #                     "96": -0.06231225211553587,
        #                     "97": -0.06399567710782994,
        #                     "98": -0.007380932689477426,
        #                     "99": -0.004634878860302505,
        #                     "100": 0,
        #                     "101": -0.007428850125970277,
        #                     "102": -0.03500866602179242,
        #                     "103": -0.07921063854258403,
        #                     "104": 0,
        #                     "105": 0,
        #                     "106": -0.1853775959037032,
        #                     "107": -0.22642622854515435,
        #                     "108": -0.2620506310964191,
        #                     "109": -0.2786120105880293,
        #                     "110": -0.2190173299139372,
        #                     "111": -0.34086007309409994,
        #                     "112": 0,
        #                     "113": 0,
        #                     "114": -0.05966515691341612,
        #                     "115": -0.049123757619676245,
        #                     "116": -0.03608025502345355,
        #                     "117": 0,
        #                     "118": -0.20288990920161432,
        #                     "119": -0.013393473384215482,
        #                     "120": -0.028713026439557254,
        #                     "121": -0.023009786235533347,
        #                     "122": -0.05693706347419972,
        #                     "123": -0.2582210912345247,
        #                     "124": -0.054578681477047836,
        #                     "125": -0.09737136128376972,
        #                     "126": -0.30659962493355775,
        #                     "127": -0.11475905021199387,
        #                     "128": -0.020711412576657096,
        #                     "129": -0.020126224746880778,
        #                     "130": -0.031155702632311544,
        #                     "131": -0.011969993385546163,
        #                     "132": -0.025122787415783443,
        #                     "133": -0.12072627685484977,
        #                     "134": -0.3025897290993145,
        #                     "135": 0,
        #                     "136": 0,
        #                     "137": 0,
        #                     "138": -0.050801992687486974,
        #                     "139": -0.0324969569053713,
        #                     "140": -0.027957502835068204,
        #                     "141": 0,
        #                     "142": -0.09628963353658682,
        #                     "143": 0,
        #                     "144": 0,
        #                     "145": 0,
        #                     "146": 0,
        #                     "147": -0.05564028449647063,
        #                     "148": 0,
        #                     "149": -0.027631435256189385,
        #                     "150": -0.07885485667424474,
        #                     "151": -0.007510649618883147,
        #                     "152": -0.0662894479351746,
        #                     "153": -0.07297776445844728,
        #                     "154": 0,
        #                     "155": -0.09811042067659609,
        #                     "156": 0,
        #                     "157": 0,
        #                     "158": -0.008667409394400138,
        #                     "159": 0,
        #                     "160": 0,
        #                     "161": 0,
        #                     "162": -0.020969006191300994,
        #                     "163": -0.032395422521644984,
        #                     "164": 0,
        #                     "165": -0.04878940565628442,
        #                     "166": -0.19820290695347023,
        #                     "167": -0.06791087549309915,
        #                     "168": 0,
        #                     "169": 0,
        #                     "170": -0.08174427827475056,
        #                     "171": -0.07053585174886245,
        #                     "172": -0.06750355211817677,
        #                     "173": 0,
        #                     "174": -0.07369246484555536,
        #                     "175": -0.08469776804425161,
        #                     "176": -0.026527526228885545,
        #                     "177": -0.02547686078814948,
        #                     "178": -0.006404264499927953,
        #                     "179": -0.0674569912591928,
        #                     "180": -0.03910259629497481,
        #                     "181": -0.054900159118081465,
        #                     "182": 0,
        #                     "183": -0.046854028942934195,
        #                     "184": -0.08779367994928444,
        #                     "185": -0.08787530402161725,
        #                     "186": -0.03088472828357014,
        #                     "187": -0.008399704590474899,
        #                     "188": 0,
        #                     "189": 0,
        #                     "190": 0,
        #                     "191": -0.004014476567586411,
        #                     "192": -0.04834589087463545,
        #                     "193": -0.052038032002377055,
        #                     "194": -0.09151299825188663,
        #                     "195": -0.1533727925607639,
        #                     "196": -0.08038983015839486,
        #                     "197": -0.012504711019985015,
        #                     "198": -0.07512887786054095,
        #                     "199": 0,
        #                     "200": -0.025264021237745182,
        #                     "201": -0.024905954886309423,
        #                     "202": -0.04159338282768879,
        #                     "203": -0.055328676002616844,
        #                     "204": -0.04685512322610741,
        #                     "205": -0.018441092495955935,
        #                     "206": -0.07695319061335776,
        #                     "207": -0.02769706681198758,
        #                     "208": 0,
        #                     "209": 0,
        #                     "210": -0.0067456002807349535,
        #                     "211": -0.13187527183120723,
        #                     "212": -0.012799266345990767,
        #                     "213": 0,
        #                     "214": 0,
        #                     "215": -0.08976334378211173,
        #                     "216": -0.06613676868583135,
        #                     "217": -0.06650644229767066,
        #                     "218": -0.033559302379947975,
        #                     "219": -0.022680424376574304,
        #                     "220": -0.029774643373975507,
        #                     "221": 0,
        #                     "222": -0.03500866602179242,
        #                     "223": 0,
        #                     "224": 0,
        #                     "225": 0,
        #                     "226": -0.011837141231826165,
        #                     "227": -0.05280666802248002,
        #                     "228": -0.012153501840143854,
        #                     "229": -0.04074245769261736,
        #                     "230": -0.2190173299139372,
        #                     "231": -0.006558066673155338,
        #                     "232": 0,
        #                     "233": 0,
        #                     "234": -0.08960260841810727,
        #                     "235": -0.03890754173959904,
        #                     "236": -0.07877331369933913,
        #                     "237": -0.02881337674735235,
        #                     "238": -0.20288990920161432,
        #                     "239": -0.12458507171256822,
        #                     "240": -0.030352519445165038,
        #                     "241": -0.027653126850743948,
        #                     "242": -0.07806212890020467,
        #                     "243": -0.22371377372000806,
        #                     "244": -0.042791364746843086,
        #                     "245": -0.06725096318155355,
        #                     "246": -0.06432556565970146,
        #                     "247": -0.13537918961473647,
        #                     "248": -0.13441840062070276,
        #                     "249": -0.05166396828532266,
        #                     "250": 0,
        #                     "251": -0.05197405049595544,
        #                     "252": 0,
        #                     "253": -0.08103995685430104,
        #                     "254": 0,
        #                     "255": 0,
        #                     "256": -0.07321694467338641,
        #                     "257": -0.045244194517841896,
        #                     "258": -0.051051604375441706,
        #                     "259": 0,
        #                     "260": -0.04255437892027386,
        #                     "261": 0,
        #                     "262": 0,
        #                     "263": -0.00718802821919956,
        #                     "264": 0,
        #                     "265": 0,
        #                     "266": -0.047847600808570256,
        #                     "267": 0,
        #                     "268": -0.06633034463655546,
        #                     "269": -0.06550153282244903,
        #                     "270": 0,
        #                     "271": 0,
        #                     "272": 0,
        #                     "273": -0.03895256213365697,
        #                     "274": -0.05273734032558394,
        #                     "275": -0.04626867058292786,
        #                     "276": -0.0463077493242709,
        #                     "277": -0.008695019747425578,
        #                     "278": -0.05166493012467992,
        #                     "279": 0,
        #                     "280": 0,
        #                     "281": -0.017973736811873572,
        #                     "282": -0.037568995039618136,
        #                     "283": -0.03863554696776389,
        #                     "284": -0.07189720707730873,
        #                     "285": 0,
        #                     "286": -0.04686615096076724,
        #                     "287": 0,
        #                     "288": 0,
        #                     "289": -0.11437191119470363,
        #                     "290": -0.11349368424143608,
        #                     "291": -0.028942564610325183,
        #                     "292": -0.05451680915760293,
        #                     "293": -0.030477270131051865,
        #                     "294": -0.12030408930047826,
        #                     "295": -0.07845705317217097,
        #                     "296": -0.0832052341581801,
        #                     "297": -0.08302447476283564,
        #                     "298": -0.09399963883256157,
        #                     "299": -0.10692932365444335,
        #                     "300": -0.08196813327940133,
        #                     "301": -0.2166738547681914,
        #                     "302": -0.23643850763340635,
        #                     "303": -0.024648544833979824,
        #                     "304": -0.02209371854779385,
        #                     "305": -0.0601112822162814,
        #                     "306": -0.21891762955780683,
        #                     "307": -0.0643303969123342,
        #                     "308": -0.030102274904848125,
        #                     "309": -0.08111589696357147,
        #                     "310": -0.10420084122098205,
        #                     "311": -0.10230368467674195,
        #                     "312": -0.047268835135119674,
        #                     "313": 0,
        #                     "314": -0.010532055716096278,
        #                     "315": -0.050176249145657825,
        #                     "316": -0.07480840537140365,
        #                     "317": 0,
        #                     "318": 0,
        #                     "319": 0,
        #                     "320": -0.20606664896508053,
        #                     "321": -0.011705283257045757,
        #                     "322": -0.14087024880710383,
        #                     "323": -0.015311255430417892,
        #                     "324": 0,
        #                     "325": 0,
        #                     "326": 0,
        #                     "327": -0.00031628096820596197,
        #                     "328": 0,
        #                     "329": 0,
        #                     "330": 0,
        #                     "331": -0.008657991405254306,
        #                     "332": -0.03658250019457254,
        #                     "333": 0,
        #                     "334": 0,
        #                     "335": 0,
        #                     "336": -0.13824205690170593,
        #                     "337": -0.10987408078906277,
        #                     "338": -0.0013698311426978693,
        #                     "339": -0.0045130431916172165,
        #                     "340": 0,
        #                     "341": 0,
        #                     "342": 0,
        #                     "343": 0,
        #                     "344": 0
        #                 }
        #                 }


        
        dict_targets =  {
                "Tipo": {
                    "0": "Bau",
                    "1": "Bau",
                    "2": "Bau",
                    "3": "Bau",
                    "4": "Bau",
                    "5": "Bau",
                    "6": "Bau",
                    "7": "Bau",
                    "8": "Bau",
                    "9": "Bau",
                    "10": "Bau",
                    "11": "Bau",
                    "12": "Bau",
                    "13": "Bau",
                    "14": "Bau",
                    "15": "Bau",
                    "16": "Bau",
                    "17": "Bau",
                    "18": "Bau",
                    "19": "Bau",
                    "20": "Bau",
                    "21": "Bau",
                    "22": "Bau",
                    "23": "Bau",
                    "24": "Bau",
                    "25": "Bau",
                    "26": "Bau",
                    "27": "Bau",
                    "28": "Bau",
                    "29": "Bau",
                    "30": "Bau",
                    "31": "Bau",
                    "32": "Bau",
                    "33": "Bau",
                    "34": "Bau",
                    "35": "Bau",
                    "36": "Bau",
                    "37": "Bau",
                    "38": "Bau",
                    "39": "Bau",
                    "40": "Bau",
                    "41": "Bau",
                    "42": "Bau",
                    "43": "Bau",
                    "44": "Bau",
                    "45": "Bau",
                    "46": "Bau",
                    "47": "Bau",
                    "48": "Bau",
                    "49": "Bau",
                    "50": "Bau",
                    "51": "Bau",
                    "52": "Bau",
                    "53": "Bau",
                    "54": "Bau",
                    "55": "Bau",
                    "56": "Bau",
                    "57": "Bau",
                    "58": "Bau",
                    "59": "Bau",
                    "60": "Bau",
                    "61": "Bau",
                    "62": "Bau",
                    "63": "Bau",
                    "64": "Bau",
                    "65": "Bau",
                    "66": "Bau",
                    "67": "Bau",
                    "68": "Bau",
                    "69": "Bau",
                    "70": "Bau",
                    "71": "Bau",
                    "72": "Bau",
                    "73": "Bau",
                    "74": "Bau",
                    "75": "Bau",
                    "76": "Bau",
                    "77": "Bau",
                    "78": "Bau",
                    "79": "Bau",
                    "80": "Bau",
                    "81": "Bau",
                    "82": "Bau",
                    "83": "Bau",
                    "84": "Bau",
                    "85": "Bau",
                    "86": "Bau",
                    "87": "Bau",
                    "88": "Bau",
                    "89": "Bau",
                    "90": "Bau",
                    "91": "Bau",
                    "92": "Bau",
                    "93": "Bau",
                    "94": "Bau",
                    "95": "Bau",
                    "96": "Bau",
                    "97": "Bau",
                    "98": "Bau",
                    "99": "Bau",
                    "100": "Bau",
                    "101": "Bau",
                    "102": "Bau",
                    "103": "Bau",
                    "104": "Bau",
                    "105": "Bau",
                    "106": "Bau",
                    "107": "Bau",
                    "108": "Bau",
                    "109": "Bau",
                    "110": "Bau",
                    "111": "Bau",
                    "112": "Bau",
                    "113": "Bau",
                    "114": "Bau",
                    "115": "Bau",
                    "116": "Bau",
                    "117": "Bau",
                    "118": "Bau",
                    "119": "Bau",
                    "120": "Geral",
                    "121": "Geral",
                    "122": "Geral",
                    "123": "Geral",
                    "124": "Geral",
                    "125": "Geral",
                    "126": "Geral",
                    "127": "Geral",
                    "128": "Geral",
                    "129": "Geral",
                    "130": "Geral",
                    "131": "Geral",
                    "132": "Geral",
                    "133": "Geral",
                    "134": "Geral",
                    "135": "Geral",
                    "136": "Geral",
                    "137": "Geral",
                    "138": "Geral",
                    "139": "Geral",
                    "140": "Geral",
                    "141": "Geral",
                    "142": "Geral",
                    "143": "Geral",
                    "144": "Geral",
                    "145": "Geral",
                    "146": "Geral",
                    "147": "Geral",
                    "148": "Geral",
                    "149": "Geral",
                    "150": "Geral",
                    "151": "Geral",
                    "152": "Geral",
                    "153": "Geral",
                    "154": "Geral",
                    "155": "Geral",
                    "156": "Geral",
                    "157": "Geral",
                    "158": "Geral",
                    "159": "Geral",
                    "160": "Geral",
                    "161": "Geral",
                    "162": "Geral",
                    "163": "Geral",
                    "164": "Geral",
                    "165": "Geral",
                    "166": "Geral",
                    "167": "Geral",
                    "168": "Geral",
                    "169": "Geral",
                    "170": "Geral",
                    "171": "Geral",
                    "172": "Geral",
                    "173": "Geral",
                    "174": "Geral",
                    "175": "Geral",
                    "176": "Geral",
                    "177": "Geral",
                    "178": "Geral",
                    "179": "Geral",
                    "180": "Geral",
                    "181": "Geral",
                    "182": "Geral",
                    "183": "Geral",
                    "184": "Geral",
                    "185": "Geral",
                    "186": "Geral",
                    "187": "Geral",
                    "188": "Geral",
                    "189": "Geral",
                    "190": "Geral",
                    "191": "Geral",
                    "192": "Geral",
                    "193": "Geral",
                    "194": "Geral",
                    "195": "Geral",
                    "196": "Geral",
                    "197": "Geral",
                    "198": "Geral",
                    "199": "Geral",
                    "200": "Geral",
                    "201": "Geral",
                    "202": "Geral",
                    "203": "Geral",
                    "204": "Geral",
                    "205": "Geral",
                    "206": "Geral",
                    "207": "Geral",
                    "208": "Geral",
                    "209": "Geral",
                    "210": "Geral",
                    "211": "Geral",
                    "212": "Geral",
                    "213": "Geral",
                    "214": "Geral",
                    "215": "Geral",
                    "216": "Geral",
                    "217": "Geral",
                    "218": "Geral",
                    "219": "Geral",
                    "220": "Geral",
                    "221": "Geral",
                    "222": "Geral",
                    "223": "Geral",
                    "224": "Geral",
                    "225": "Geral",
                    "226": "Geral",
                    "227": "Geral",
                    "228": "Geral",
                    "229": "Geral",
                    "230": "Geral",
                    "231": "Geral",
                    "232": "Geral",
                    "233": "Geral",
                    "234": "Geral",
                    "235": "Geral",
                    "236": "Geral",
                    "237": "Geral",
                    "238": "Geral",
                    "239": "Geral",
                    "240": "Ofertão",
                    "241": "Ofertão",
                    "242": "Ofertão",
                    "243": "Ofertão",
                    "244": "Ofertão",
                    "245": "Ofertão",
                    "246": "Ofertão",
                    "247": "Ofertão",
                    "248": "Ofertão",
                    "249": "Ofertão",
                    "250": "Ofertão",
                    "251": "Ofertão",
                    "252": "Ofertão",
                    "253": "Ofertão",
                    "254": "Ofertão",
                    "255": "Ofertão",
                    "256": "Ofertão",
                    "257": "Ofertão",
                    "258": "Ofertão",
                    "259": "Ofertão",
                    "260": "Ofertão",
                    "261": "Ofertão",
                    "262": "Ofertão",
                    "263": "Ofertão",
                    "264": "Ofertão",
                    "265": "Ofertão",
                    "266": "Ofertão",
                    "267": "Ofertão",
                    "268": "Ofertão",
                    "269": "Ofertão",
                    "270": "Ofertão",
                    "271": "Ofertão",
                    "272": "Ofertão",
                    "273": "Ofertão",
                    "274": "Ofertão",
                    "275": "Ofertão",
                    "276": "Ofertão",
                    "277": "Ofertão",
                    "278": "Ofertão",
                    "279": "Ofertão",
                    "280": "Ofertão",
                    "281": "Ofertão",
                    "282": "Ofertão",
                    "283": "Ofertão",
                    "284": "Ofertão",
                    "285": "Ofertão",
                    "286": "Ofertão",
                    "287": "Ofertão",
                    "288": "Ofertão",
                    "289": "Ofertão",
                    "290": "Ofertão",
                    "291": "Ofertão",
                    "292": "Ofertão",
                    "293": "Ofertão",
                    "294": "Ofertão",
                    "295": "Ofertão",
                    "296": "Ofertão",
                    "297": "Ofertão",
                    "298": "Ofertão",
                    "299": "Ofertão",
                    "300": "Ofertão",
                    "301": "Ofertão",
                    "302": "Ofertão",
                    "303": "Ofertão",
                    "304": "Ofertão",
                    "305": "Ofertão",
                    "306": "Ofertão",
                    "307": "Ofertão",
                    "308": "Ofertão",
                    "309": "Ofertão",
                    "310": "Ofertão",
                    "311": "Ofertão",
                    "312": "Ofertão",
                    "313": "Ofertão",
                    "314": "Ofertão",
                    "315": "Ofertão",
                    "316": "Ofertão",
                    "317": "Ofertão",
                    "318": "Ofertão",
                    "319": "Ofertão",
                    "320": "Ofertão",
                    "321": "Ofertão",
                    "322": "Ofertão",
                    "323": "Ofertão",
                    "324": "Ofertão",
                    "325": "Ofertão",
                    "326": "Ofertão",
                    "327": "Ofertão",
                    "328": "Ofertão",
                    "329": "Ofertão",
                    "330": "Ofertão",
                    "331": "Ofertão",
                    "332": "Ofertão",
                    "333": "Ofertão",
                    "334": "Ofertão",
                    "335": "Ofertão",
                    "336": "Ofertão",
                    "337": "Ofertão",
                    "338": "Ofertão",
                    "339": "Ofertão",
                    "340": "Ofertão",
                    "341": "Ofertão",
                    "342": "Ofertão",
                    "343": "Ofertão",
                    "344": "Ofertão"
                },
                "Categoria": {
                    "0": "Arroz e Feijão",
                    "1": "Arroz e Feijão",
                    "2": "Arroz e Feijão",
                    "3": "Arroz e Feijão",
                    "4": "Arroz e Feijão",
                    "5": "Arroz e Feijão",
                    "6": "Arroz e Feijão",
                    "7": "Arroz e Feijão",
                    "8": "Açúcar e Adoçante",
                    "9": "Açúcar e Adoçante",
                    "10": "Açúcar e Adoçante",
                    "11": "Açúcar e Adoçante",
                    "12": "Açúcar e Adoçante",
                    "13": "Açúcar e Adoçante",
                    "14": "Açúcar e Adoçante",
                    "15": "Açúcar e Adoçante",
                    "16": "Biscoitos",
                    "17": "Biscoitos",
                    "18": "Biscoitos",
                    "19": "Biscoitos",
                    "20": "Biscoitos",
                    "21": "Biscoitos",
                    "22": "Biscoitos",
                    "23": "Biscoitos",
                    "24": "Cafés, Chás e Achocolatados",
                    "25": "Cafés, Chás e Achocolatados",
                    "26": "Cafés, Chás e Achocolatados",
                    "27": "Cafés, Chás e Achocolatados",
                    "28": "Cafés, Chás e Achocolatados",
                    "29": "Cafés, Chás e Achocolatados",
                    "30": "Cafés, Chás e Achocolatados",
                    "31": "Cafés, Chás e Achocolatados",
                    "32": "Cervejas",
                    "33": "Cervejas",
                    "34": "Cervejas",
                    "35": "Cervejas",
                    "36": "Cervejas",
                    "37": "Cervejas",
                    "38": "Cervejas",
                    "39": "Cervejas",
                    "40": "Cervejas Premium",
                    "41": "Cervejas Premium",
                    "42": "Cervejas Premium",
                    "43": "Cervejas Premium",
                    "44": "Cervejas Premium",
                    "45": "Cervejas Premium",
                    "46": "Cervejas Premium",
                    "47": "Cervejas Premium",
                    "48": "Derivados de Leite",
                    "49": "Derivados de Leite",
                    "50": "Derivados de Leite",
                    "51": "Derivados de Leite",
                    "52": "Derivados de Leite",
                    "53": "Derivados de Leite",
                    "54": "Derivados de Leite",
                    "55": "Derivados de Leite",
                    "56": "Grãos e Farináceos",
                    "57": "Grãos e Farináceos",
                    "58": "Grãos e Farináceos",
                    "59": "Grãos e Farináceos",
                    "60": "Grãos e Farináceos",
                    "61": "Grãos e Farináceos",
                    "62": "Grãos e Farináceos",
                    "63": "Grãos e Farináceos",
                    "64": "Leite",
                    "65": "Leite",
                    "66": "Leite",
                    "67": "Leite",
                    "68": "Leite",
                    "69": "Leite",
                    "70": "Leite",
                    "71": "Leite",
                    "72": "Limpeza de Roupa",
                    "73": "Limpeza de Roupa",
                    "74": "Limpeza de Roupa",
                    "75": "Limpeza de Roupa",
                    "76": "Limpeza de Roupa",
                    "77": "Limpeza de Roupa",
                    "78": "Limpeza de Roupa",
                    "79": "Limpeza de Roupa",
                    "80": "Limpeza em Geral",
                    "81": "Limpeza em Geral",
                    "82": "Limpeza em Geral",
                    "83": "Limpeza em Geral",
                    "84": "Limpeza em Geral",
                    "85": "Limpeza em Geral",
                    "86": "Limpeza em Geral",
                    "87": "Limpeza em Geral",
                    "88": "Massas Secas",
                    "89": "Massas Secas",
                    "90": "Massas Secas",
                    "91": "Massas Secas",
                    "92": "Massas Secas",
                    "93": "Massas Secas",
                    "94": "Massas Secas",
                    "95": "Massas Secas",
                    "96": "Refrigerantes",
                    "97": "Refrigerantes",
                    "98": "Refrigerantes",
                    "99": "Refrigerantes",
                    "100": "Refrigerantes",
                    "101": "Refrigerantes",
                    "102": "Refrigerantes",
                    "103": "Refrigerantes",
                    "104": "Sucos E Refrescos",
                    "105": "Sucos E Refrescos",
                    "106": "Sucos E Refrescos",
                    "107": "Sucos E Refrescos",
                    "108": "Sucos E Refrescos",
                    "109": "Sucos E Refrescos",
                    "110": "Sucos E Refrescos",
                    "111": "Sucos E Refrescos",
                    "112": "Óleos, Azeites e Vinagres",
                    "113": "Óleos, Azeites e Vinagres",
                    "114": "Óleos, Azeites e Vinagres",
                    "115": "Óleos, Azeites e Vinagres",
                    "116": "Óleos, Azeites e Vinagres",
                    "117": "Óleos, Azeites e Vinagres",
                    "118": "Óleos, Azeites e Vinagres",
                    "119": "Óleos, Azeites e Vinagres",
                    "120": "Arroz e Feijão",
                    "121": "Arroz e Feijão",
                    "122": "Arroz e Feijão",
                    "123": "Arroz e Feijão",
                    "124": "Arroz e Feijão",
                    "125": "Arroz e Feijão",
                    "126": "Arroz e Feijão",
                    "127": "Arroz e Feijão",
                    "128": "Açúcar e Adoçante",
                    "129": "Açúcar e Adoçante",
                    "130": "Açúcar e Adoçante",
                    "131": "Açúcar e Adoçante",
                    "132": "Açúcar e Adoçante",
                    "133": "Açúcar e Adoçante",
                    "134": "Açúcar e Adoçante",
                    "135": "Açúcar e Adoçante",
                    "136": "Biscoitos",
                    "137": "Biscoitos",
                    "138": "Biscoitos",
                    "139": "Biscoitos",
                    "140": "Biscoitos",
                    "141": "Biscoitos",
                    "142": "Biscoitos",
                    "143": "Biscoitos",
                    "144": "Cafés, Chás e Achocolatados",
                    "145": "Cafés, Chás e Achocolatados",
                    "146": "Cafés, Chás e Achocolatados",
                    "147": "Cafés, Chás e Achocolatados",
                    "148": "Cafés, Chás e Achocolatados",
                    "149": "Cafés, Chás e Achocolatados",
                    "150": "Cafés, Chás e Achocolatados",
                    "151": "Cafés, Chás e Achocolatados",
                    "152": "Cervejas",
                    "153": "Cervejas",
                    "154": "Cervejas",
                    "155": "Cervejas",
                    "156": "Cervejas",
                    "157": "Cervejas",
                    "158": "Cervejas",
                    "159": "Cervejas",
                    "160": "Cervejas Premium",
                    "161": "Cervejas Premium",
                    "162": "Cervejas Premium",
                    "163": "Cervejas Premium",
                    "164": "Cervejas Premium",
                    "165": "Cervejas Premium",
                    "166": "Cervejas Premium",
                    "167": "Cervejas Premium",
                    "168": "Derivados de Leite",
                    "169": "Derivados de Leite",
                    "170": "Derivados de Leite",
                    "171": "Derivados de Leite",
                    "172": "Derivados de Leite",
                    "173": "Derivados de Leite",
                    "174": "Derivados de Leite",
                    "175": "Derivados de Leite",
                    "176": "Grãos e Farináceos",
                    "177": "Grãos e Farináceos",
                    "178": "Grãos e Farináceos",
                    "179": "Grãos e Farináceos",
                    "180": "Grãos e Farináceos",
                    "181": "Grãos e Farináceos",
                    "182": "Grãos e Farináceos",
                    "183": "Grãos e Farináceos",
                    "184": "Leite",
                    "185": "Leite",
                    "186": "Leite",
                    "187": "Leite",
                    "188": "Leite",
                    "189": "Leite",
                    "190": "Leite",
                    "191": "Leite",
                    "192": "Limpeza de Roupa",
                    "193": "Limpeza de Roupa",
                    "194": "Limpeza de Roupa",
                    "195": "Limpeza de Roupa",
                    "196": "Limpeza de Roupa",
                    "197": "Limpeza de Roupa",
                    "198": "Limpeza de Roupa",
                    "199": "Limpeza de Roupa",
                    "200": "Limpeza em Geral",
                    "201": "Limpeza em Geral",
                    "202": "Limpeza em Geral",
                    "203": "Limpeza em Geral",
                    "204": "Limpeza em Geral",
                    "205": "Limpeza em Geral",
                    "206": "Limpeza em Geral",
                    "207": "Limpeza em Geral",
                    "208": "Massas Secas",
                    "209": "Massas Secas",
                    "210": "Massas Secas",
                    "211": "Massas Secas",
                    "212": "Massas Secas",
                    "213": "Massas Secas",
                    "214": "Massas Secas",
                    "215": "Massas Secas",
                    "216": "Refrigerantes",
                    "217": "Refrigerantes",
                    "218": "Refrigerantes",
                    "219": "Refrigerantes",
                    "220": "Refrigerantes",
                    "221": "Refrigerantes",
                    "222": "Refrigerantes",
                    "223": "Refrigerantes",
                    "224": "Sucos E Refrescos",
                    "225": "Sucos E Refrescos",
                    "226": "Sucos E Refrescos",
                    "227": "Sucos E Refrescos",
                    "228": "Sucos E Refrescos",
                    "229": "Sucos E Refrescos",
                    "230": "Sucos E Refrescos",
                    "231": "Sucos E Refrescos",
                    "232": "Óleos, Azeites e Vinagres",
                    "233": "Óleos, Azeites e Vinagres",
                    "234": "Óleos, Azeites e Vinagres",
                    "235": "Óleos, Azeites e Vinagres",
                    "236": "Óleos, Azeites e Vinagres",
                    "237": "Óleos, Azeites e Vinagres",
                    "238": "Óleos, Azeites e Vinagres",
                    "239": "Óleos, Azeites e Vinagres",
                    "240": "Arroz e Feijão",
                    "241": "Arroz e Feijão",
                    "242": "Arroz e Feijão",
                    "243": "Arroz e Feijão",
                    "244": "Arroz e Feijão",
                    "245": "Arroz e Feijão",
                    "246": "Arroz e Feijão",
                    "247": "Açúcar e Adoçante",
                    "248": "Açúcar e Adoçante",
                    "249": "Açúcar e Adoçante",
                    "250": "Açúcar e Adoçante",
                    "251": "Açúcar e Adoçante",
                    "252": "Açúcar e Adoçante",
                    "253": "Açúcar e Adoçante",
                    "254": "Biscoitos",
                    "255": "Biscoitos",
                    "256": "Biscoitos",
                    "257": "Biscoitos",
                    "258": "Biscoitos",
                    "259": "Biscoitos",
                    "260": "Biscoitos",
                    "261": "Cafés, Chás e Achocolatados",
                    "262": "Cafés, Chás e Achocolatados",
                    "263": "Cafés, Chás e Achocolatados",
                    "264": "Cafés, Chás e Achocolatados",
                    "265": "Cafés, Chás e Achocolatados",
                    "266": "Cafés, Chás e Achocolatados",
                    "267": "Cafés, Chás e Achocolatados",
                    "268": "Cervejas",
                    "269": "Cervejas",
                    "270": "Cervejas",
                    "271": "Cervejas",
                    "272": "Cervejas",
                    "273": "Cervejas",
                    "274": "Cervejas",
                    "275": "Cervejas Premium",
                    "276": "Cervejas Premium",
                    "277": "Cervejas Premium",
                    "278": "Cervejas Premium",
                    "279": "Cervejas Premium",
                    "280": "Cervejas Premium",
                    "281": "Cervejas Premium",
                    "282": "Derivados de Leite",
                    "283": "Derivados de Leite",
                    "284": "Derivados de Leite",
                    "285": "Derivados de Leite",
                    "286": "Derivados de Leite",
                    "287": "Derivados de Leite",
                    "288": "Derivados de Leite",
                    "289": "Grãos e Farináceos",
                    "290": "Grãos e Farináceos",
                    "291": "Grãos e Farináceos",
                    "292": "Grãos e Farináceos",
                    "293": "Grãos e Farináceos",
                    "294": "Grãos e Farináceos",
                    "295": "Grãos e Farináceos",
                    "296": "Leite",
                    "297": "Leite",
                    "298": "Leite",
                    "299": "Leite",
                    "300": "Leite",
                    "301": "Leite",
                    "302": "Leite",
                    "303": "Limpeza de Roupa",
                    "304": "Limpeza de Roupa",
                    "305": "Limpeza de Roupa",
                    "306": "Limpeza de Roupa",
                    "307": "Limpeza de Roupa",
                    "308": "Limpeza de Roupa",
                    "309": "Limpeza de Roupa",
                    "310": "Limpeza em Geral",
                    "311": "Limpeza em Geral",
                    "312": "Limpeza em Geral",
                    "313": "Limpeza em Geral",
                    "314": "Limpeza em Geral",
                    "315": "Limpeza em Geral",
                    "316": "Limpeza em Geral",
                    "317": "Massas Secas",
                    "318": "Massas Secas",
                    "319": "Massas Secas",
                    "320": "Massas Secas",
                    "321": "Massas Secas",
                    "322": "Massas Secas",
                    "323": "Massas Secas",
                    "324": "Refrigerantes",
                    "325": "Refrigerantes",
                    "326": "Refrigerantes",
                    "327": "Refrigerantes",
                    "328": "Refrigerantes",
                    "329": "Refrigerantes",
                    "330": "Refrigerantes",
                    "331": "Sucos E Refrescos",
                    "332": "Sucos E Refrescos",
                    "333": "Sucos E Refrescos",
                    "334": "Sucos E Refrescos",
                    "335": "Sucos E Refrescos",
                    "336": "Sucos E Refrescos",
                    "337": "Sucos E Refrescos",
                    "338": "Óleos, Azeites e Vinagres",
                    "339": "Óleos, Azeites e Vinagres",
                    "340": "Óleos, Azeites e Vinagres",
                    "341": "Óleos, Azeites e Vinagres",
                    "342": "Óleos, Azeites e Vinagres",
                    "343": "Óleos, Azeites e Vinagres",
                    "344": "Óleos, Azeites e Vinagres"
                },
                "Region": {
                    "0": "BAC",
                    "1": "BAC",
                    "2": "RJC",
                    "3": "RJC",
                    "4": "RJC",
                    "5": "RJI",
                    "6": "RJI",
                    "7": "RJI",
                    "8": "BAC",
                    "9": "BAC",
                    "10": "RJC",
                    "11": "RJC",
                    "12": "RJC",
                    "13": "RJI",
                    "14": "RJI",
                    "15": "RJI",
                    "16": "BAC",
                    "17": "BAC",
                    "18": "RJC",
                    "19": "RJC",
                    "20": "RJC",
                    "21": "RJI",
                    "22": "RJI",
                    "23": "RJI",
                    "24": "BAC",
                    "25": "BAC",
                    "26": "RJC",
                    "27": "RJC",
                    "28": "RJC",
                    "29": "RJI",
                    "30": "RJI",
                    "31": "RJI",
                    "32": "BAC",
                    "33": "BAC",
                    "34": "RJC",
                    "35": "RJC",
                    "36": "RJC",
                    "37": "RJI",
                    "38": "RJI",
                    "39": "RJI",
                    "40": "BAC",
                    "41": "BAC",
                    "42": "RJC",
                    "43": "RJC",
                    "44": "RJC",
                    "45": "RJI",
                    "46": "RJI",
                    "47": "RJI",
                    "48": "BAC",
                    "49": "BAC",
                    "50": "RJC",
                    "51": "RJC",
                    "52": "RJC",
                    "53": "RJI",
                    "54": "RJI",
                    "55": "RJI",
                    "56": "BAC",
                    "57": "BAC",
                    "58": "RJC",
                    "59": "RJC",
                    "60": "RJC",
                    "61": "RJI",
                    "62": "RJI",
                    "63": "RJI",
                    "64": "BAC",
                    "65": "BAC",
                    "66": "RJC",
                    "67": "RJC",
                    "68": "RJC",
                    "69": "RJI",
                    "70": "RJI",
                    "71": "RJI",
                    "72": "BAC",
                    "73": "BAC",
                    "74": "RJC",
                    "75": "RJC",
                    "76": "RJC",
                    "77": "RJI",
                    "78": "RJI",
                    "79": "RJI",
                    "80": "BAC",
                    "81": "BAC",
                    "82": "RJC",
                    "83": "RJC",
                    "84": "RJC",
                    "85": "RJI",
                    "86": "RJI",
                    "87": "RJI",
                    "88": "BAC",
                    "89": "BAC",
                    "90": "RJC",
                    "91": "RJC",
                    "92": "RJC",
                    "93": "RJI",
                    "94": "RJI",
                    "95": "RJI",
                    "96": "BAC",
                    "97": "BAC",
                    "98": "RJC",
                    "99": "RJC",
                    "100": "RJC",
                    "101": "RJI",
                    "102": "RJI",
                    "103": "RJI",
                    "104": "BAC",
                    "105": "BAC",
                    "106": "RJC",
                    "107": "RJC",
                    "108": "RJC",
                    "109": "RJI",
                    "110": "RJI",
                    "111": "RJI",
                    "112": "BAC",
                    "113": "BAC",
                    "114": "RJC",
                    "115": "RJC",
                    "116": "RJC",
                    "117": "RJI",
                    "118": "RJI",
                    "119": "RJI",
                    "120": "BAC",
                    "121": "BAC",
                    "122": "RJC",
                    "123": "RJC",
                    "124": "RJC",
                    "125": "RJI",
                    "126": "RJI",
                    "127": "RJI",
                    "128": "BAC",
                    "129": "BAC",
                    "130": "RJC",
                    "131": "RJC",
                    "132": "RJC",
                    "133": "RJI",
                    "134": "RJI",
                    "135": "RJI",
                    "136": "BAC",
                    "137": "BAC",
                    "138": "RJC",
                    "139": "RJC",
                    "140": "RJC",
                    "141": "RJI",
                    "142": "RJI",
                    "143": "RJI",
                    "144": "BAC",
                    "145": "BAC",
                    "146": "RJC",
                    "147": "RJC",
                    "148": "RJC",
                    "149": "RJI",
                    "150": "RJI",
                    "151": "RJI",
                    "152": "BAC",
                    "153": "BAC",
                    "154": "RJC",
                    "155": "RJC",
                    "156": "RJC",
                    "157": "RJI",
                    "158": "RJI",
                    "159": "RJI",
                    "160": "BAC",
                    "161": "BAC",
                    "162": "RJC",
                    "163": "RJC",
                    "164": "RJC",
                    "165": "RJI",
                    "166": "RJI",
                    "167": "RJI",
                    "168": "BAC",
                    "169": "BAC",
                    "170": "RJC",
                    "171": "RJC",
                    "172": "RJC",
                    "173": "RJI",
                    "174": "RJI",
                    "175": "RJI",
                    "176": "BAC",
                    "177": "BAC",
                    "178": "RJC",
                    "179": "RJC",
                    "180": "RJC",
                    "181": "RJI",
                    "182": "RJI",
                    "183": "RJI",
                    "184": "BAC",
                    "185": "BAC",
                    "186": "RJC",
                    "187": "RJC",
                    "188": "RJC",
                    "189": "RJI",
                    "190": "RJI",
                    "191": "RJI",
                    "192": "BAC",
                    "193": "BAC",
                    "194": "RJC",
                    "195": "RJC",
                    "196": "RJC",
                    "197": "RJI",
                    "198": "RJI",
                    "199": "RJI",
                    "200": "BAC",
                    "201": "BAC",
                    "202": "RJC",
                    "203": "RJC",
                    "204": "RJC",
                    "205": "RJI",
                    "206": "RJI",
                    "207": "RJI",
                    "208": "BAC",
                    "209": "BAC",
                    "210": "RJC",
                    "211": "RJC",
                    "212": "RJC",
                    "213": "RJI",
                    "214": "RJI",
                    "215": "RJI",
                    "216": "BAC",
                    "217": "BAC",
                    "218": "RJC",
                    "219": "RJC",
                    "220": "RJC",
                    "221": "RJI",
                    "222": "RJI",
                    "223": "RJI",
                    "224": "BAC",
                    "225": "BAC",
                    "226": "RJC",
                    "227": "RJC",
                    "228": "RJC",
                    "229": "RJI",
                    "230": "RJI",
                    "231": "RJI",
                    "232": "BAC",
                    "233": "BAC",
                    "234": "RJC",
                    "235": "RJC",
                    "236": "RJC",
                    "237": "RJI",
                    "238": "RJI",
                    "239": "RJI",
                    "240": "BAC",
                    "241": "BAC",
                    "242": "RJC",
                    "243": "RJC",
                    "244": "RJC",
                    "245": "RJI",
                    "246": "RJI",
                    "247": "BAC",
                    "248": "BAC",
                    "249": "RJC",
                    "250": "RJC",
                    "251": "RJC",
                    "252": "RJI",
                    "253": "RJI",
                    "254": "BAC",
                    "255": "BAC",
                    "256": "RJC",
                    "257": "RJC",
                    "258": "RJC",
                    "259": "RJI",
                    "260": "RJI",
                    "261": "BAC",
                    "262": "BAC",
                    "263": "RJC",
                    "264": "RJC",
                    "265": "RJC",
                    "266": "RJI",
                    "267": "RJI",
                    "268": "BAC",
                    "269": "BAC",
                    "270": "RJC",
                    "271": "RJC",
                    "272": "RJC",
                    "273": "RJI",
                    "274": "RJI",
                    "275": "BAC",
                    "276": "BAC",
                    "277": "RJC",
                    "278": "RJC",
                    "279": "RJC",
                    "280": "RJI",
                    "281": "RJI",
                    "282": "BAC",
                    "283": "BAC",
                    "284": "RJC",
                    "285": "RJC",
                    "286": "RJC",
                    "287": "RJI",
                    "288": "RJI",
                    "289": "BAC",
                    "290": "BAC",
                    "291": "RJC",
                    "292": "RJC",
                    "293": "RJC",
                    "294": "RJI",
                    "295": "RJI",
                    "296": "BAC",
                    "297": "BAC",
                    "298": "RJC",
                    "299": "RJC",
                    "300": "RJC",
                    "301": "RJI",
                    "302": "RJI",
                    "303": "BAC",
                    "304": "BAC",
                    "305": "RJC",
                    "306": "RJC",
                    "307": "RJC",
                    "308": "RJI",
                    "309": "RJI",
                    "310": "BAC",
                    "311": "BAC",
                    "312": "RJC",
                    "313": "RJC",
                    "314": "RJC",
                    "315": "RJI",
                    "316": "RJI",
                    "317": "BAC",
                    "318": "BAC",
                    "319": "RJC",
                    "320": "RJC",
                    "321": "RJC",
                    "322": "RJI",
                    "323": "RJI",
                    "324": "BAC",
                    "325": "BAC",
                    "326": "RJC",
                    "327": "RJC",
                    "328": "RJC",
                    "329": "RJI",
                    "330": "RJI",
                    "331": "BAC",
                    "332": "BAC",
                    "333": "RJC",
                    "334": "RJC",
                    "335": "RJC",
                    "336": "RJI",
                    "337": "RJI",
                    "338": "BAC",
                    "339": "BAC",
                    "340": "RJC",
                    "341": "RJC",
                    "342": "RJC",
                    "343": "RJI",
                    "344": "RJI"
                },
                "Size": {
                    "0": "1-4 Cxs",
                    "1": "size",
                    "2": "1-4 Cxs",
                    "3": "5-9 Cxs",
                    "4": "size",
                    "5": "1-4 Cxs",
                    "6": "5-9 Cxs",
                    "7": "size",
                    "8": "1-4 Cxs",
                    "9": "size",
                    "10": "1-4 Cxs",
                    "11": "5-9 Cxs",
                    "12": "size",
                    "13": "1-4 Cxs",
                    "14": "5-9 Cxs",
                    "15": "size",
                    "16": "1-4 Cxs",
                    "17": "size",
                    "18": "1-4 Cxs",
                    "19": "5-9 Cxs",
                    "20": "size",
                    "21": "1-4 Cxs",
                    "22": "5-9 Cxs",
                    "23": "size",
                    "24": "1-4 Cxs",
                    "25": "size",
                    "26": "1-4 Cxs",
                    "27": "5-9 Cxs",
                    "28": "size",
                    "29": "1-4 Cxs",
                    "30": "5-9 Cxs",
                    "31": "size",
                    "32": "1-4 Cxs",
                    "33": "size",
                    "34": "1-4 Cxs",
                    "35": "5-9 Cxs",
                    "36": "size",
                    "37": "1-4 Cxs",
                    "38": "5-9 Cxs",
                    "39": "size",
                    "40": "1-4 Cxs",
                    "41": "size",
                    "42": "1-4 Cxs",
                    "43": "5-9 Cxs",
                    "44": "size",
                    "45": "1-4 Cxs",
                    "46": "5-9 Cxs",
                    "47": "size",
                    "48": "1-4 Cxs",
                    "49": "size",
                    "50": "1-4 Cxs",
                    "51": "5-9 Cxs",
                    "52": "size",
                    "53": "1-4 Cxs",
                    "54": "5-9 Cxs",
                    "55": "size",
                    "56": "1-4 Cxs",
                    "57": "size",
                    "58": "1-4 Cxs",
                    "59": "5-9 Cxs",
                    "60": "size",
                    "61": "1-4 Cxs",
                    "62": "5-9 Cxs",
                    "63": "size",
                    "64": "1-4 Cxs",
                    "65": "size",
                    "66": "1-4 Cxs",
                    "67": "5-9 Cxs",
                    "68": "size",
                    "69": "1-4 Cxs",
                    "70": "5-9 Cxs",
                    "71": "size",
                    "72": "1-4 Cxs",
                    "73": "size",
                    "74": "1-4 Cxs",
                    "75": "5-9 Cxs",
                    "76": "size",
                    "77": "1-4 Cxs",
                    "78": "5-9 Cxs",
                    "79": "size",
                    "80": "1-4 Cxs",
                    "81": "size",
                    "82": "1-4 Cxs",
                    "83": "5-9 Cxs",
                    "84": "size",
                    "85": "1-4 Cxs",
                    "86": "5-9 Cxs",
                    "87": "size",
                    "88": "1-4 Cxs",
                    "89": "size",
                    "90": "1-4 Cxs",
                    "91": "5-9 Cxs",
                    "92": "size",
                    "93": "1-4 Cxs",
                    "94": "5-9 Cxs",
                    "95": "size",
                    "96": "1-4 Cxs",
                    "97": "size",
                    "98": "1-4 Cxs",
                    "99": "5-9 Cxs",
                    "100": "size",
                    "101": "1-4 Cxs",
                    "102": "5-9 Cxs",
                    "103": "size",
                    "104": "1-4 Cxs",
                    "105": "size",
                    "106": "1-4 Cxs",
                    "107": "5-9 Cxs",
                    "108": "size",
                    "109": "1-4 Cxs",
                    "110": "5-9 Cxs",
                    "111": "size",
                    "112": "1-4 Cxs",
                    "113": "size",
                    "114": "1-4 Cxs",
                    "115": "5-9 Cxs",
                    "116": "size",
                    "117": "1-4 Cxs",
                    "118": "5-9 Cxs",
                    "119": "size",
                    "120": "1-4 Cxs",
                    "121": "size",
                    "122": "1-4 Cxs",
                    "123": "5-9 Cxs",
                    "124": "size",
                    "125": "1-4 Cxs",
                    "126": "5-9 Cxs",
                    "127": "size",
                    "128": "1-4 Cxs",
                    "129": "size",
                    "130": "1-4 Cxs",
                    "131": "5-9 Cxs",
                    "132": "size",
                    "133": "1-4 Cxs",
                    "134": "5-9 Cxs",
                    "135": "size",
                    "136": "1-4 Cxs",
                    "137": "size",
                    "138": "1-4 Cxs",
                    "139": "5-9 Cxs",
                    "140": "size",
                    "141": "1-4 Cxs",
                    "142": "5-9 Cxs",
                    "143": "size",
                    "144": "1-4 Cxs",
                    "145": "size",
                    "146": "1-4 Cxs",
                    "147": "5-9 Cxs",
                    "148": "size",
                    "149": "1-4 Cxs",
                    "150": "5-9 Cxs",
                    "151": "size",
                    "152": "1-4 Cxs",
                    "153": "size",
                    "154": "1-4 Cxs",
                    "155": "5-9 Cxs",
                    "156": "size",
                    "157": "1-4 Cxs",
                    "158": "5-9 Cxs",
                    "159": "size",
                    "160": "1-4 Cxs",
                    "161": "size",
                    "162": "1-4 Cxs",
                    "163": "5-9 Cxs",
                    "164": "size",
                    "165": "1-4 Cxs",
                    "166": "5-9 Cxs",
                    "167": "size",
                    "168": "1-4 Cxs",
                    "169": "size",
                    "170": "1-4 Cxs",
                    "171": "5-9 Cxs",
                    "172": "size",
                    "173": "1-4 Cxs",
                    "174": "5-9 Cxs",
                    "175": "size",
                    "176": "1-4 Cxs",
                    "177": "size",
                    "178": "1-4 Cxs",
                    "179": "5-9 Cxs",
                    "180": "size",
                    "181": "1-4 Cxs",
                    "182": "5-9 Cxs",
                    "183": "size",
                    "184": "1-4 Cxs",
                    "185": "size",
                    "186": "1-4 Cxs",
                    "187": "5-9 Cxs",
                    "188": "size",
                    "189": "1-4 Cxs",
                    "190": "5-9 Cxs",
                    "191": "size",
                    "192": "1-4 Cxs",
                    "193": "size",
                    "194": "1-4 Cxs",
                    "195": "5-9 Cxs",
                    "196": "size",
                    "197": "1-4 Cxs",
                    "198": "5-9 Cxs",
                    "199": "size",
                    "200": "1-4 Cxs",
                    "201": "size",
                    "202": "1-4 Cxs",
                    "203": "5-9 Cxs",
                    "204": "size",
                    "205": "1-4 Cxs",
                    "206": "5-9 Cxs",
                    "207": "size",
                    "208": "1-4 Cxs",
                    "209": "size",
                    "210": "1-4 Cxs",
                    "211": "5-9 Cxs",
                    "212": "size",
                    "213": "1-4 Cxs",
                    "214": "5-9 Cxs",
                    "215": "size",
                    "216": "1-4 Cxs",
                    "217": "size",
                    "218": "1-4 Cxs",
                    "219": "5-9 Cxs",
                    "220": "size",
                    "221": "1-4 Cxs",
                    "222": "5-9 Cxs",
                    "223": "size",
                    "224": "1-4 Cxs",
                    "225": "size",
                    "226": "1-4 Cxs",
                    "227": "5-9 Cxs",
                    "228": "size",
                    "229": "1-4 Cxs",
                    "230": "5-9 Cxs",
                    "231": "size",
                    "232": "1-4 Cxs",
                    "233": "size",
                    "234": "1-4 Cxs",
                    "235": "5-9 Cxs",
                    "236": "size",
                    "237": "1-4 Cxs",
                    "238": "5-9 Cxs",
                    "239": "size",
                    "240": "1-4 Cxs",
                    "241": "size",
                    "242": "1-4 Cxs",
                    "243": "5-9 Cxs",
                    "244": "size",
                    "245": "1-4 Cxs",
                    "246": "size",
                    "247": "1-4 Cxs",
                    "248": "size",
                    "249": "1-4 Cxs",
                    "250": "5-9 Cxs",
                    "251": "size",
                    "252": "1-4 Cxs",
                    "253": "size",
                    "254": "1-4 Cxs",
                    "255": "size",
                    "256": "1-4 Cxs",
                    "257": "5-9 Cxs",
                    "258": "size",
                    "259": "1-4 Cxs",
                    "260": "size",
                    "261": "1-4 Cxs",
                    "262": "size",
                    "263": "1-4 Cxs",
                    "264": "5-9 Cxs",
                    "265": "size",
                    "266": "1-4 Cxs",
                    "267": "size",
                    "268": "1-4 Cxs",
                    "269": "size",
                    "270": "1-4 Cxs",
                    "271": "5-9 Cxs",
                    "272": "size",
                    "273": "1-4 Cxs",
                    "274": "size",
                    "275": "1-4 Cxs",
                    "276": "size",
                    "277": "1-4 Cxs",
                    "278": "5-9 Cxs",
                    "279": "size",
                    "280": "1-4 Cxs",
                    "281": "size",
                    "282": "1-4 Cxs",
                    "283": "size",
                    "284": "1-4 Cxs",
                    "285": "5-9 Cxs",
                    "286": "size",
                    "287": "1-4 Cxs",
                    "288": "size",
                    "289": "1-4 Cxs",
                    "290": "size",
                    "291": "1-4 Cxs",
                    "292": "5-9 Cxs",
                    "293": "size",
                    "294": "1-4 Cxs",
                    "295": "size",
                    "296": "1-4 Cxs",
                    "297": "size",
                    "298": "1-4 Cxs",
                    "299": "5-9 Cxs",
                    "300": "size",
                    "301": "1-4 Cxs",
                    "302": "size",
                    "303": "1-4 Cxs",
                    "304": "size",
                    "305": "1-4 Cxs",
                    "306": "5-9 Cxs",
                    "307": "size",
                    "308": "1-4 Cxs",
                    "309": "size",
                    "310": "1-4 Cxs",
                    "311": "size",
                    "312": "1-4 Cxs",
                    "313": "5-9 Cxs",
                    "314": "size",
                    "315": "1-4 Cxs",
                    "316": "size",
                    "317": "1-4 Cxs",
                    "318": "size",
                    "319": "1-4 Cxs",
                    "320": "5-9 Cxs",
                    "321": "size",
                    "322": "1-4 Cxs",
                    "323": "size",
                    "324": "1-4 Cxs",
                    "325": "size",
                    "326": "1-4 Cxs",
                    "327": "5-9 Cxs",
                    "328": "size",
                    "329": "1-4 Cxs",
                    "330": "size",
                    "331": "1-4 Cxs",
                    "332": "size",
                    "333": "1-4 Cxs",
                    "334": "5-9 Cxs",
                    "335": "size",
                    "336": "1-4 Cxs",
                    "337": "size",
                    "338": "1-4 Cxs",
                    "339": "size",
                    "340": "1-4 Cxs",
                    "341": "5-9 Cxs",
                    "342": "size",
                    "343": "1-4 Cxs",
                    "344": "size"
                },
                "Trend Atual": {
                    "0": 0.21283967225015882,
                    "1": 0.2128102377102472,
                    "2": 0.26350640999710917,
                    "3": 0.2701563460756816,
                    "4": 0.22588527407111897,
                    "5": 0.2979715256599122,
                    "6": 0.296849608616763,
                    "7": 0.2926490822720907,
                    "8": 0.13600270693449876,
                    "9": 0.13521692479197586,
                    "10": 0.33447091978831756,
                    "11": 0.39250306608780444,
                    "12": 0.3251552769160695,
                    "13": 0.20687921042625912,
                    "14": 0.31666733546462356,
                    "15": 0.2187614804119686,
                    "16": 0.2999822086004345,
                    "17": 0.30636703951121447,
                    "18": 0.4832017933711329,
                    "19": 0.4602619258882162,
                    "20": 0.45939000587914935,
                    "21": 0.5827554459908473,
                    "22": 0.5640905527933173,
                    "23": 0.582161485010254,
                    "24": 0.3304096970702537,
                    "25": 0.32780667178010314,
                    "26": 0.17695352507034307,
                    "27": 0.430405528253816,
                    "28": 0.19943637505957598,
                    "29": 0.2548660389647522,
                    "30": 0.546970027251221,
                    "31": 0.2548284742538801,
                    "32": 0.31407047958683737,
                    "33": 0.3140549349806476,
                    "34": 0.07609412156661616,
                    "35": 0.14028401527246745,
                    "36": 0.0665643572658971,
                    "37": 0.19780923756984028,
                    "38": 0.42097992355400066,
                    "39": 0.22308234350187156,
                    "40": 0.11918973651706513,
                    "41": 0.1192300348455539,
                    "42": 0.04489105057001433,
                    "43": 0.14968636827726348,
                    "44": 0.03849959232946084,
                    "45": 0.08145391950554819,
                    "46": 0.2222462708875105,
                    "47": 0.07343436410616018,
                    "48": 0.5413940961864712,
                    "49": 0.5415632623771585,
                    "50": 0.19984364453924355,
                    "51": 0.3363372255726181,
                    "52": 0.20479331335082285,
                    "53": 0.13152818764744278,
                    "54": 0.5522955170122629,
                    "55": 0.09576285833690361,
                    "56": 0.17906514929173128,
                    "57": 0.18167220195075673,
                    "58": 0.17682944715299526,
                    "59": 0.20332816538864693,
                    "60": 0.21961456160269832,
                    "61": 0.20337091039301572,
                    "62": 0.49688400660822307,
                    "63": 0.18895939997932792,
                    "64": 0.33003404914241447,
                    "65": 0.33035477166893185,
                    "66": 0.4396037881215084,
                    "67": 0.5240032846193501,
                    "68": 0.4305474985859405,
                    "69": 0.3780672763296619,
                    "70": 0.719478273483059,
                    "71": 0.37988107726339443,
                    "72": 0.22970437608101058,
                    "73": 0.2341194220167455,
                    "74": 0.2575929379059132,
                    "75": 0.1425976250386788,
                    "76": 0.20926315434892287,
                    "77": 0.08663078151021289,
                    "78": 0.424524741095391,
                    "79": 0.07757117221253809,
                    "80": 0.24815212298265682,
                    "81": 0.24776771194610708,
                    "82": 0.20510876405214667,
                    "83": 0.21026356677478342,
                    "84": 0.20738918941033993,
                    "85": 0.20071044218361966,
                    "86": 0.42994187609594287,
                    "87": 0.1611579362897704,
                    "88": 0.2882015145466819,
                    "89": 0.27825410979864756,
                    "90": 0.17279645765354013,
                    "91": 0.2542494185940949,
                    "92": 0.35012388255988025,
                    "93": 0.23346488997480971,
                    "94": 0.4830484047121416,
                    "95": 0.20533106861561748,
                    "96": 0.25554878367921197,
                    "97": 0.2534613055067756,
                    "98": 0.22552379356114763,
                    "99": 0.18287288535616916,
                    "100": 0.21760827663875193,
                    "101": 0.24408664518561102,
                    "102": 0.4756714089024071,
                    "103": 0.2309906697313801,
                    "104": 0.29066369933829045,
                    "105": 0.29923059657328926,
                    "106": 0.12806130710074773,
                    "107": 0.24803316984425153,
                    "108": 0.11943419145566032,
                    "109": 0.04457988259825843,
                    "110": 0.27830066920435764,
                    "111": 0,
                    "112": 0.1719501345061776,
                    "113": 0.171562927474356,
                    "114": 0.21592397608869673,
                    "115": 0.3229012312711734,
                    "116": 0.22461551265521865,
                    "117": 0.3540133072737043,
                    "118": 0.3167789242686465,
                    "119": 0.3150307563244753,
                    "120": 0.20175459502164594,
                    "121": 0.20697478532832894,
                    "122": 0.2495010267837813,
                    "123": 0.26438793396875687,
                    "124": 0.2430178103694681,
                    "125": 0.26829082468547416,
                    "126": 0.296849608616763,
                    "127": 0.22361879201108267,
                    "128": 0.1681220198588986,
                    "129": 0.16783012475988815,
                    "130": 0.3848137963381494,
                    "131": 0.4120929834402589,
                    "132": 0.36640403914709485,
                    "133": 0.36307594704333346,
                    "134": 0.31666733546462356,
                    "135": 0.4522542685764868,
                    "136": 0.3684418323196522,
                    "137": 0.36602989104780714,
                    "138": 0.31813906930422065,
                    "139": 0.49503913377773495,
                    "140": 0.33884119766018234,
                    "141": 0.5092509802772571,
                    "142": 0.5640905527933173,
                    "143": 0.45387151380902874,
                    "144": 0.3763253404124172,
                    "145": 0.37805388750121866,
                    "146": 0.27937488509883474,
                    "147": 0.30393008148445044,
                    "148": 0.2757102488884806,
                    "149": 0.2525906417493689,
                    "150": 0.546970027251221,
                    "151": 0.25617055631886615,
                    "152": 0.32680114555923684,
                    "153": 0.3331017924968156,
                    "154": 0.1250562341022588,
                    "155": 0.15096540999886568,
                    "156": 0.12717321299062603,
                    "157": 0.16575745033224734,
                    "158": 0.42097992355400066,
                    "159": 0.14713646604861075,
                    "160": 0.17534144432898935,
                    "161": 0.17418157737433684,
                    "162": 0.04607231416629753,
                    "163": 0.1437765034423028,
                    "164": 0.07341460794060584,
                    "165": 0.056039600030374635,
                    "166": 0.2222462708875105,
                    "167": 0.03074550106806631,
                    "168": 0.4570997430927968,
                    "169": 0.4567599980313648,
                    "170": 0.2851319553792471,
                    "171": 0.3381604905298988,
                    "172": 0.2835469621470778,
                    "173": 0.44040425601534094,
                    "174": 0.5522955170122629,
                    "175": 0.3382665078369224,
                    "176": 0.18232900012451478,
                    "177": 0.18334966006905468,
                    "178": 0.24913875248820913,
                    "179": 0.17464668654545457,
                    "180": 0.20928928876733013,
                    "181": 0.22073968438261662,
                    "182": 0.49688400660822307,
                    "183": 0.21385104699395338,
                    "184": 0.268176143151152,
                    "185": 0.2744596680666146,
                    "186": 0.433867259687314,
                    "187": 0.540103578169033,
                    "188": 0.4417680446519972,
                    "189": 0.3654224390519498,
                    "190": 0.719478273483059,
                    "191": 0.3508126395947101,
                    "192": 0.209133216172427,
                    "193": 0.2022720282776591,
                    "194": 0.11517957481976562,
                    "195": 0.12912821477780753,
                    "196": 0.10214296201598065,
                    "197": 0.19190302438464024,
                    "198": 0.424524741095391,
                    "199": 0.18776267011184108,
                    "200": 0.2830655096410057,
                    "201": 0.2829922673411062,
                    "202": 0.2118967522972051,
                    "203": 0.21099673741047867,
                    "204": 0.19399957310892427,
                    "205": 0.22539294321441117,
                    "206": 0.42994187609594287,
                    "207": 0.20749671138836948,
                    "208": 0.213096123446751,
                    "209": 0.21291615085197796,
                    "210": 0.2506586745768572,
                    "211": 0.2094446977043573,
                    "212": 0.22647927699607753,
                    "213": 0.25707928985420364,
                    "214": 0.4830484047121416,
                    "215": 0.20470787308785884,
                    "216": 0.26505132101380435,
                    "217": 0.2647174490740657,
                    "218": 0.19516493658896855,
                    "219": 0.1656287482677661,
                    "220": 0.1737821065986529,
                    "221": 0.26000108991645626,
                    "222": 0.4756714089024071,
                    "223": 0.2930872592650788,
                    "224": 0.34610969430739597,
                    "225": 0.34783162767334,
                    "226": 0.28182283801653396,
                    "227": 0.2625898147819719,
                    "228": 0.2710438255312445,
                    "229": 0.26515027119161394,
                    "230": 0.27830066920435764,
                    "231": 0.2889728008865883,
                    "232": 0.2321160151776738,
                    "233": 0.2316419886303517,
                    "234": 0.19075882422582932,
                    "235": 0.30529643312175814,
                    "236": 0.19326523077205718,
                    "237": 0.3643386446469434,
                    "238": 0.3167789242686465,
                    "239": 0.25901234149286695,
                    "240": 0.2166578803818997,
                    "241": 0.2175805160111307,
                    "242": 0.23436018466715794,
                    "243": 0.05429166505535478,
                    "244": 0.21853019713802097,
                    "245": 0.24697552856803795,
                    "246": 0.23759180983120462,
                    "247": 0.2314300680710237,
                    "248": 0.23142891616647754,
                    "249": 0.39362180000643254,
                    "250": 0.4184364702671385,
                    "251": 0.3662571188228474,
                    "252": 0.4749137696298486,
                    "253": 0.4003844679345431,
                    "254": 0.3250532444059925,
                    "255": 0.3261505087113196,
                    "256": 0.2750050698134695,
                    "257": 0.39318627560518304,
                    "258": 0.3048355124505147,
                    "259": 0.43786922130556893,
                    "260": 0.4143682699990997,
                    "261": 0.35069762870594245,
                    "262": 0.3481167423646757,
                    "263": 0.2707054810605497,
                    "264": 0.24812821213059633,
                    "265": 0.27438489589105614,
                    "266": 0.23693799579039093,
                    "267": 0.2593953647869301,
                    "268": 0.34009945105926026,
                    "269": 0.33985031499703294,
                    "270": 0.13238529357879378,
                    "271": 0.14283949833341125,
                    "272": 0.11261623304884241,
                    "273": 0.12345274193394071,
                    "274": 0.09864903306224435,
                    "275": 0.17723530946553695,
                    "276": 0.17689578455544555,
                    "277": 0.06260957345998687,
                    "278": 0.06347721339576583,
                    "279": 0.08375453286475816,
                    "280": 0.11044875193529269,
                    "281": 0.09300193753935851,
                    "282": 0.41793396437890773,
                    "283": 0.4188125126369136,
                    "284": 0.29485195574333206,
                    "285": 0.30575427858951476,
                    "286": 0.28073411492182476,
                    "287": 0.48039353056600254,
                    "288": 0.44304853867460586,
                    "289": 0.23128354890509661,
                    "290": 0.23143572258114958,
                    "291": 0.26166031743077106,
                    "292": 0.15476273138784558,
                    "293": 0.24320109439383367,
                    "294": 0.15115079831914663,
                    "295": 0.19968275887085132,
                    "296": 0.29175007965123173,
                    "297": 0.29226664927207163,
                    "298": 0.4382981860390268,
                    "299": 0.39222220399932756,
                    "300": 0.4151533025340636,
                    "301": 0.3740831888745702,
                    "302": 0.3578251953727126,
                    "303": 0.25249766401032553,
                    "304": 0.2529088081647477,
                    "305": 0.14035957098914886,
                    "306": 0.1003031296327048,
                    "307": 0.1198765195013511,
                    "308": 0.17038860964996005,
                    "309": 0.12540578592244112,
                    "310": 0.23821877950514736,
                    "311": 0.23967710360697997,
                    "312": 0.19683851930969085,
                    "313": 0.1904758625393218,
                    "314": 0.19795786702250262,
                    "315": 0.27050527599986535,
                    "316": 0.2578024242540844,
                    "317": 0.22812869686160775,
                    "318": 0.23835946194417526,
                    "319": 0.2872176816162929,
                    "320": 0.11941426171120213,
                    "321": 0.23245210001409156,
                    "322": 0.18297004815952808,
                    "323": 0.25177544345712954,
                    "324": 0.3814022039332815,
                    "325": 0.3816324778094203,
                    "326": 0.22648469849465414,
                    "327": 0.09787497246166062,
                    "328": 0.20433131014234365,
                    "329": 0.2626890828686319,
                    "330": 0.24051000430486594,
                    "331": 0.2926520852922473,
                    "332": 0.29030566787733303,
                    "333": 0.315521059328352,
                    "334": 0.3280772547585458,
                    "335": 0.29095261073545897,
                    "336": 0.2593081974738734,
                    "337": 0.22723552878836176,
                    "338": 0.234089362875312,
                    "339": 0.23351176423945352,
                    "340": 0.3787006636081367,
                    "341": 0.41959317762153797,
                    "342": 0.36648549643363326,
                    "343": 0.6604727357740372,
                    "344": 0.5627083319723457
                },
                "% Target Positivação": {
                    "0": 0.24121569460007783,
                    "1": 0.2410395475256433,
                    "2": 0.2873652838677233,
                    "3": 0.6834086742880926,
                    "4": 0.2982846191347278,
                    "5": 0.40312200496933387,
                    "6": 0.6034492335503208,
                    "7": 0.37657587479113785,
                    "8": 0.1541420319336395,
                    "9": 0.15425028254547174,
                    "10": 0.40816037004386413,
                    "11": 0.43537286561283717,
                    "12": 0.3695122720380119,
                    "13": 0.5001876012436258,
                    "14": 0.619257064563938,
                    "15": 0.5315122313262813,
                    "16": 0.2999822086004345,
                    "17": 0.3085019919702773,
                    "18": 0.3964248572282161,
                    "19": 0.5397196455885527,
                    "20": 0.3595977882575496,
                    "21": 0.6073610428526613,
                    "22": 0.6603801863299041,
                    "23": 0.6071210697271965,
                    "24": 0.35293041076242837,
                    "25": 0.33676105873232515,
                    "26": 0.3074587275944513,
                    "27": 0.430405528253816,
                    "28": 0.30554991867480263,
                    "29": 0.5035275140754416,
                    "30": 0.6258248839254658,
                    "31": 0.5005921342606237,
                    "32": 0.4282248097362146,
                    "33": 0.4285439961948455,
                    "34": 0.17065671647077224,
                    "35": 0.296914492015669,
                    "36": 0.1655156619481438,
                    "37": 0.23401021253056054,
                    "38": 0.4296473329484008,
                    "39": 0.23377782812906678,
                    "40": 0.14007586843051356,
                    "41": 0.13983594182501394,
                    "42": 0.06042578500323643,
                    "43": 0.18973206420582153,
                    "44": 0.057396966021127156,
                    "45": 0.14646934208317094,
                    "46": 0.42044917784098074,
                    "47": 0.14176885016948906,
                    "48": 0.5413940961864712,
                    "49": 0.5415632623771585,
                    "50": 0.38187294132568134,
                    "51": 0.5195358325704842,
                    "52": 0.3651040635822726,
                    "53": 0.6140550398855394,
                    "54": 0.6259879818578182,
                    "55": 0.6029035720131478,
                    "56": 0.19199343696201127,
                    "57": 0.1916734649826006,
                    "58": 0.25081391681656173,
                    "59": 0.23931030004620388,
                    "60": 0.24080676009768426,
                    "61": 0.2630025369620815,
                    "62": 0.49688400660822307,
                    "63": 0.2536113256540201,
                    "64": 0.503445601101825,
                    "65": 0.5034177581248646,
                    "66": 0.4353348901128276,
                    "67": 0.5488023863318318,
                    "68": 0.4142998515361242,
                    "69": 0.3618737003280747,
                    "70": 0.6352029218379573,
                    "71": 0.34334212198146563,
                    "72": 0.22970437608101058,
                    "73": 0.2341194220167455,
                    "74": 0.23709486029780802,
                    "75": 0.30586073795986846,
                    "76": 0.24445533801595237,
                    "77": 0.2730707584672399,
                    "78": 0.49965361895593197,
                    "79": 0.26400910623056506,
                    "80": 0.32186237556998,
                    "81": 0.3308354651643145,
                    "82": 0.24175490157803298,
                    "83": 0.26674758168692114,
                    "84": 0.22925826664547133,
                    "85": 0.23968799736472451,
                    "86": 0.5068950667093006,
                    "87": 0.22202379413462098,
                    "88": 0.21083571875337398,
                    "89": 0.22818955416147696,
                    "90": 0.24008869426873103,
                    "91": 0.34634953981223987,
                    "92": 0.26932023741291816,
                    "93": 0.3248006077991298,
                    "94": 0.4824055318529098,
                    "95": 0.3078347945502027,
                    "96": 0.31786103579474784,
                    "97": 0.31745698261460553,
                    "98": 0.23290472625062505,
                    "99": 0.18750776421647167,
                    "100": 0.21017174154114934,
                    "101": 0.2515154953115813,
                    "102": 0.5106800749241995,
                    "103": 0.31020130827396414,
                    "104": 0.2460261628543306,
                    "105": 0.23388594336538712,
                    "106": 0.31343890300445093,
                    "107": 0.4744593983894059,
                    "108": 0.3814848225520794,
                    "109": 0.32319189318628777,
                    "110": 0.49731799911829483,
                    "111": 0.34086007309409994,
                    "112": 0.1719501345061776,
                    "113": 0.17064271673045736,
                    "114": 0.27558913300211285,
                    "115": 0.3720249888908496,
                    "116": 0.2606957676786722,
                    "117": 0.3540133072737043,
                    "118": 0.5196688334702608,
                    "119": 0.3284242297086908,
                    "120": 0.2304676214612032,
                    "121": 0.2299845715638623,
                    "122": 0.306438090257981,
                    "123": 0.5226090252032816,
                    "124": 0.29759649184651593,
                    "125": 0.3656621859692439,
                    "126": 0.6034492335503208,
                    "127": 0.33837784222307654,
                    "128": 0.1888334324355557,
                    "129": 0.18795634950676893,
                    "130": 0.4159694989704609,
                    "131": 0.42406297682580507,
                    "132": 0.3915268265628783,
                    "133": 0.4838022238981832,
                    "134": 0.619257064563938,
                    "135": 0.4522542685764868,
                    "136": 0.2978900665753283,
                    "137": 0.3020517253234977,
                    "138": 0.3689410619917076,
                    "139": 0.5275360906831063,
                    "140": 0.36679870049525054,
                    "141": 0.4135456422843813,
                    "142": 0.6603801863299041,
                    "143": 0.41276687012542085,
                    "144": 0.34514686178341164,
                    "145": 0.34359738702597487,
                    "146": 0.27937488509883474,
                    "147": 0.35957036598092107,
                    "148": 0.2749929803227029,
                    "149": 0.2802220770055583,
                    "150": 0.6258248839254658,
                    "151": 0.2636812059377493,
                    "152": 0.39309059349441144,
                    "153": 0.4060795569552629,
                    "154": 0.1250562341022588,
                    "155": 0.24907583067546177,
                    "156": 0.12174120908797903,
                    "157": 0.16575745033224734,
                    "158": 0.4296473329484008,
                    "159": 0.14713646604861075,
                    "160": 0.1526726367834974,
                    "161": 0.15072227105471203,
                    "162": 0.06704132035759852,
                    "163": 0.1761719259639478,
                    "164": 0.06754183881450311,
                    "165": 0.10482900568665905,
                    "166": 0.42044917784098074,
                    "167": 0.09865637656116547,
                    "168": 0.44030087369812315,
                    "169": 0.4399352252778456,
                    "170": 0.3668762336539977,
                    "171": 0.4086963422787613,
                    "172": 0.3510505142652546,
                    "173": 0.43810480296094867,
                    "174": 0.6259879818578182,
                    "175": 0.422964275881174,
                    "176": 0.20885652635340032,
                    "177": 0.20882652085720416,
                    "178": 0.2555430169881371,
                    "179": 0.24210367780464737,
                    "180": 0.24839188506230495,
                    "181": 0.2756398435006981,
                    "182": 0.49688400660822307,
                    "183": 0.2607050759368876,
                    "184": 0.35596982310043646,
                    "185": 0.36233497208823184,
                    "186": 0.46475198797088413,
                    "187": 0.5485032827595079,
                    "188": 0.4101533201698609,
                    "189": 0.3546827052365166,
                    "190": 0.6352029218379573,
                    "191": 0.3548271161622965,
                    "192": 0.25747910704706245,
                    "193": 0.25431006028003617,
                    "194": 0.20669257307165226,
                    "195": 0.2825010073385714,
                    "196": 0.1825327921743755,
                    "197": 0.20440773540462526,
                    "198": 0.49965361895593197,
                    "199": 0.18776267011184108,
                    "200": 0.30832953087875087,
                    "201": 0.30789822222741564,
                    "202": 0.2534901351248939,
                    "203": 0.2663254134130955,
                    "204": 0.24085469633503168,
                    "205": 0.2438340357103671,
                    "206": 0.5068950667093006,
                    "207": 0.23519377820035706,
                    "208": 0.19801152901144026,
                    "209": 0.20210800890601807,
                    "210": 0.25740427485759215,
                    "211": 0.34131996953556454,
                    "212": 0.2392785433420683,
                    "213": 0.25707928985420364,
                    "214": 0.4824055318529098,
                    "215": 0.2944712168699706,
                    "216": 0.3311880896996357,
                    "217": 0.33122389137173636,
                    "218": 0.22872423896891653,
                    "219": 0.1883091726443404,
                    "220": 0.2035567499726284,
                    "221": 0.22100500056300457,
                    "222": 0.5106800749241995,
                    "223": 0.25670653425365897,
                    "224": 0.28716856373765615,
                    "225": 0.29043456499510123,
                    "226": 0.2936599792483601,
                    "227": 0.3153964828044519,
                    "228": 0.28319732737138836,
                    "229": 0.3058927288842313,
                    "230": 0.49731799911829483,
                    "231": 0.2955308675597436,
                    "232": 0.18060067837690436,
                    "233": 0.1812633415562584,
                    "234": 0.2803614326439366,
                    "235": 0.3442039748613572,
                    "236": 0.2720385444713963,
                    "237": 0.3931520213942957,
                    "238": 0.5196688334702608,
                    "239": 0.38359741320543517,
                    "240": 0.24701039982706474,
                    "241": 0.24523364286187466,
                    "242": 0.3124223135673626,
                    "243": 0.2780054387753628,
                    "244": 0.26132156188486405,
                    "245": 0.3142264917495915,
                    "246": 0.3019173754909061,
                    "247": 0.36680925768576017,
                    "248": 0.3658473167871803,
                    "249": 0.4452857682917552,
                    "250": 0.360703580210814,
                    "251": 0.41823116931880283,
                    "252": 0.4749137696298486,
                    "253": 0.4814244247888441,
                    "254": 0.29146999265040063,
                    "255": 0.2948638031954366,
                    "256": 0.3482220144868559,
                    "257": 0.43843047012302494,
                    "258": 0.3558871168259564,
                    "259": 0.43786922130556893,
                    "260": 0.45692264891937356,
                    "261": 0.35069762870594245,
                    "262": 0.3481167423646757,
                    "263": 0.27789350927974926,
                    "264": 0.24812821213059633,
                    "265": 0.27438489589105614,
                    "266": 0.2847855965989612,
                    "267": 0.2593953647869301,
                    "268": 0.4064297956958157,
                    "269": 0.40535184781948197,
                    "270": 0.13238529357879378,
                    "271": 0.13460714626293788,
                    "272": 0.11261623304884241,
                    "273": 0.16240530406759768,
                    "274": 0.1513863733878283,
                    "275": 0.2235039800484648,
                    "276": 0.22320353387971645,
                    "277": 0.07130459320741245,
                    "278": 0.11514214352044574,
                    "279": 0.06217295736400658,
                    "280": 0.11044875193529269,
                    "281": 0.11097567435123208,
                    "282": 0.45550295941852587,
                    "283": 0.4574480596046775,
                    "284": 0.3667491628206408,
                    "285": 0.30575427858951476,
                    "286": 0.327600265882592,
                    "287": 0.43697811360377414,
                    "288": 0.41210895437444933,
                    "289": 0.34565546009980025,
                    "290": 0.34492940682258566,
                    "291": 0.29060288204109624,
                    "292": 0.2092795405454485,
                    "293": 0.27367836452488553,
                    "294": 0.2714548876196249,
                    "295": 0.2781398120430223,
                    "296": 0.3749553138094118,
                    "297": 0.3752911240349073,
                    "298": 0.5322978248715884,
                    "299": 0.4991515276537709,
                    "300": 0.4971214358134649,
                    "301": 0.5907570436427616,
                    "302": 0.594263703006119,
                    "303": 0.27714620884430535,
                    "304": 0.27500252671254155,
                    "305": 0.20047085320543026,
                    "306": 0.3192207591905116,
                    "307": 0.1842069164136853,
                    "308": 0.20049088455480818,
                    "309": 0.20652168288601258,
                    "310": 0.3424196207261294,
                    "311": 0.3419807882837219,
                    "312": 0.24410735444481052,
                    "313": 0.13688095923774077,
                    "314": 0.2084899227385989,
                    "315": 0.32068152514552317,
                    "316": 0.33261082962548805,
                    "317": 0.21822766486453613,
                    "318": 0.21663798054289485,
                    "319": 0.2772926196752578,
                    "320": 0.32548091067628265,
                    "321": 0.2441573832711373,
                    "322": 0.3238402969666319,
                    "323": 0.26708669888754744,
                    "324": 0.35242842185907397,
                    "325": 0.35229121267717334,
                    "326": 0.22648469849465414,
                    "327": 0.09819125342986658,
                    "328": 0.20433131014234365,
                    "329": 0.2523320687532098,
                    "330": 0.22718344051150482,
                    "331": 0.3013100766975016,
                    "332": 0.32688816807190557,
                    "333": 0.315521059328352,
                    "334": 0.31149365046690747,
                    "335": 0.29095261073545897,
                    "336": 0.39755025437557934,
                    "337": 0.33710960957742453,
                    "338": 0.23545919401800988,
                    "339": 0.23802480743107074,
                    "340": 0.3663347700114191,
                    "341": 0.3395549876256303,
                    "342": 0.36648549643363326,
                    "343": 0.5742939111505023,
                    "344": 0.5627083319723457
                },
                "Delta Top": {
                    "0": -0.02837602234991901,
                    "1": -0.028229309815396092,
                    "2": -0.02385887387061414,
                    "3": -0.413252328212411,
                    "4": -0.07239934506360884,
                    "5": -0.10515047930942167,
                    "6": -0.30659962493355775,
                    "7": -0.08392679251904717,
                    "8": -0.018139324999140755,
                    "9": -0.019033357753495878,
                    "10": -0.07368945025554657,
                    "11": -0.042869799525032726,
                    "12": -0.04435699512194241,
                    "13": -0.2933083908173667,
                    "14": -0.3025897290993145,
                    "15": -0.3127507509143127,
                    "16": 0,
                    "17": -0.002134952459062811,
                    "18": 0,
                    "19": -0.0794577197003365,
                    "20": 0,
                    "21": -0.024605596861814072,
                    "22": -0.09628963353658682,
                    "23": -0.024959584716942484,
                    "24": -0.022520713692174676,
                    "25": -0.008954386952222004,
                    "26": -0.13050520252410824,
                    "27": 0,
                    "28": -0.10611354361522665,
                    "29": -0.24866147511068942,
                    "30": -0.07885485667424474,
                    "31": -0.24576366000674355,
                    "32": -0.1141543301493772,
                    "33": -0.11448906121419788,
                    "34": -0.09456259490415608,
                    "35": -0.15663047674320155,
                    "36": -0.0989513046822467,
                    "37": -0.03620097496072025,
                    "38": -0.008667409394400138,
                    "39": -0.010695484627195218,
                    "40": -0.02088613191344843,
                    "41": -0.020605906979460034,
                    "42": -0.015534734433222099,
                    "43": -0.040045695928558056,
                    "44": -0.018897373691666317,
                    "45": -0.06501542257762276,
                    "46": -0.19820290695347023,
                    "47": -0.06833448606332888,
                    "48": 0,
                    "49": 0,
                    "50": -0.18202929678643778,
                    "51": -0.1831986069978661,
                    "52": -0.16031075023144975,
                    "53": -0.48252685223809666,
                    "54": -0.07369246484555536,
                    "55": -0.5071407136762442,
                    "56": -0.012928287670279981,
                    "57": -0.010001263031843871,
                    "58": -0.07398446966356648,
                    "59": -0.03598213465755695,
                    "60": -0.021192198494985937,
                    "61": -0.05963162656906576,
                    "62": 0,
                    "63": -0.06465192567469219,
                    "64": -0.17341155195941055,
                    "65": -0.1730629864559327,
                    "66": 0,
                    "67": -0.024799101712481675,
                    "68": 0,
                    "69": 0,
                    "70": 0,
                    "71": 0,
                    "72": 0,
                    "73": 0,
                    "74": 0,
                    "75": -0.16326311292118967,
                    "76": -0.0351921836670295,
                    "77": -0.186439976957027,
                    "78": -0.07512887786054095,
                    "79": -0.18643793401802697,
                    "80": -0.07371025258732317,
                    "81": -0.08306775321820742,
                    "82": -0.0366461375258863,
                    "83": -0.056484014912137726,
                    "84": -0.0218690772351314,
                    "85": -0.03897755518110485,
                    "86": -0.07695319061335776,
                    "87": -0.060865857844850574,
                    "88": 0,
                    "89": 0,
                    "90": -0.0672922366151909,
                    "91": -0.09210012121814498,
                    "92": 0,
                    "93": -0.09133571782432007,
                    "94": 0,
                    "95": -0.1025037259345852,
                    "96": -0.06231225211553587,
                    "97": -0.06399567710782994,
                    "98": -0.007380932689477426,
                    "99": -0.004634878860302505,
                    "100": 0,
                    "101": -0.007428850125970277,
                    "102": -0.03500866602179242,
                    "103": -0.07921063854258403,
                    "104": 0,
                    "105": 0,
                    "106": -0.1853775959037032,
                    "107": -0.22642622854515435,
                    "108": -0.2620506310964191,
                    "109": -0.2786120105880293,
                    "110": -0.2190173299139372,
                    "111": -0.34086007309409994,
                    "112": 0,
                    "113": 0,
                    "114": -0.05966515691341612,
                    "115": -0.049123757619676245,
                    "116": -0.03608025502345355,
                    "117": 0,
                    "118": -0.20288990920161432,
                    "119": -0.013393473384215482,
                    "120": -0.028713026439557254,
                    "121": -0.023009786235533347,
                    "122": -0.05693706347419972,
                    "123": -0.2582210912345247,
                    "124": -0.054578681477047836,
                    "125": -0.09737136128376972,
                    "126": -0.30659962493355775,
                    "127": -0.11475905021199387,
                    "128": -0.020711412576657096,
                    "129": -0.020126224746880778,
                    "130": -0.031155702632311544,
                    "131": -0.011969993385546163,
                    "132": -0.025122787415783443,
                    "133": -0.12072627685484977,
                    "134": -0.3025897290993145,
                    "135": 0,
                    "136": 0,
                    "137": 0,
                    "138": -0.050801992687486974,
                    "139": -0.0324969569053713,
                    "140": -0.027957502835068204,
                    "141": 0,
                    "142": -0.09628963353658682,
                    "143": 0,
                    "144": 0,
                    "145": 0,
                    "146": 0,
                    "147": -0.05564028449647063,
                    "148": 0,
                    "149": -0.027631435256189385,
                    "150": -0.07885485667424474,
                    "151": -0.007510649618883147,
                    "152": -0.0662894479351746,
                    "153": -0.07297776445844728,
                    "154": 0,
                    "155": -0.09811042067659609,
                    "156": 0,
                    "157": 0,
                    "158": -0.008667409394400138,
                    "159": 0,
                    "160": 0,
                    "161": 0,
                    "162": -0.020969006191300994,
                    "163": -0.032395422521644984,
                    "164": 0,
                    "165": -0.04878940565628442,
                    "166": -0.19820290695347023,
                    "167": -0.06791087549309915,
                    "168": 0,
                    "169": 0,
                    "170": -0.08174427827475056,
                    "171": -0.07053585174886245,
                    "172": -0.06750355211817677,
                    "173": 0,
                    "174": -0.07369246484555536,
                    "175": -0.08469776804425161,
                    "176": -0.026527526228885545,
                    "177": -0.02547686078814948,
                    "178": -0.006404264499927953,
                    "179": -0.0674569912591928,
                    "180": -0.03910259629497481,
                    "181": -0.054900159118081465,
                    "182": 0,
                    "183": -0.046854028942934195,
                    "184": -0.08779367994928444,
                    "185": -0.08787530402161725,
                    "186": -0.03088472828357014,
                    "187": -0.008399704590474899,
                    "188": 0,
                    "189": 0,
                    "190": 0,
                    "191": -0.004014476567586411,
                    "192": -0.04834589087463545,
                    "193": -0.052038032002377055,
                    "194": -0.09151299825188663,
                    "195": -0.1533727925607639,
                    "196": -0.08038983015839486,
                    "197": -0.012504711019985015,
                    "198": -0.07512887786054095,
                    "199": 0,
                    "200": -0.025264021237745182,
                    "201": -0.024905954886309423,
                    "202": -0.04159338282768879,
                    "203": -0.055328676002616844,
                    "204": -0.04685512322610741,
                    "205": -0.018441092495955935,
                    "206": -0.07695319061335776,
                    "207": -0.02769706681198758,
                    "208": 0,
                    "209": 0,
                    "210": -0.0067456002807349535,
                    "211": -0.13187527183120723,
                    "212": -0.012799266345990767,
                    "213": 0,
                    "214": 0,
                    "215": -0.08976334378211173,
                    "216": -0.06613676868583135,
                    "217": -0.06650644229767066,
                    "218": -0.033559302379947975,
                    "219": -0.022680424376574304,
                    "220": -0.029774643373975507,
                    "221": 0,
                    "222": -0.03500866602179242,
                    "223": 0,
                    "224": 0,
                    "225": 0,
                    "226": -0.011837141231826165,
                    "227": -0.05280666802248002,
                    "228": -0.012153501840143854,
                    "229": -0.04074245769261736,
                    "230": -0.2190173299139372,
                    "231": -0.006558066673155338,
                    "232": 0,
                    "233": 0,
                    "234": -0.08960260841810727,
                    "235": -0.03890754173959904,
                    "236": -0.07877331369933913,
                    "237": -0.02881337674735235,
                    "238": -0.20288990920161432,
                    "239": -0.12458507171256822,
                    "240": -0.030352519445165038,
                    "241": -0.027653126850743948,
                    "242": -0.07806212890020467,
                    "243": -0.22371377372000806,
                    "244": -0.042791364746843086,
                    "245": -0.06725096318155355,
                    "246": -0.06432556565970146,
                    "247": -0.13537918961473647,
                    "248": -0.13441840062070276,
                    "249": -0.05166396828532266,
                    "250": 0,
                    "251": -0.05197405049595544,
                    "252": 0,
                    "253": -0.08103995685430104,
                    "254": 0,
                    "255": 0,
                    "256": -0.07321694467338641,
                    "257": -0.045244194517841896,
                    "258": -0.051051604375441706,
                    "259": 0,
                    "260": -0.04255437892027386,
                    "261": 0,
                    "262": 0,
                    "263": -0.00718802821919956,
                    "264": 0,
                    "265": 0,
                    "266": -0.047847600808570256,
                    "267": 0,
                    "268": -0.06633034463655546,
                    "269": -0.06550153282244903,
                    "270": 0,
                    "271": 0,
                    "272": 0,
                    "273": -0.03895256213365697,
                    "274": -0.05273734032558394,
                    "275": -0.04626867058292786,
                    "276": -0.0463077493242709,
                    "277": -0.008695019747425578,
                    "278": -0.05166493012467992,
                    "279": 0,
                    "280": 0,
                    "281": -0.017973736811873572,
                    "282": -0.037568995039618136,
                    "283": -0.03863554696776389,
                    "284": -0.07189720707730873,
                    "285": 0,
                    "286": -0.04686615096076724,
                    "287": 0,
                    "288": 0,
                    "289": -0.11437191119470363,
                    "290": -0.11349368424143608,
                    "291": -0.028942564610325183,
                    "292": -0.05451680915760293,
                    "293": -0.030477270131051865,
                    "294": -0.12030408930047826,
                    "295": -0.07845705317217097,
                    "296": -0.0832052341581801,
                    "297": -0.08302447476283564,
                    "298": -0.09399963883256157,
                    "299": -0.10692932365444335,
                    "300": -0.08196813327940133,
                    "301": -0.2166738547681914,
                    "302": -0.23643850763340635,
                    "303": -0.024648544833979824,
                    "304": -0.02209371854779385,
                    "305": -0.0601112822162814,
                    "306": -0.21891762955780683,
                    "307": -0.0643303969123342,
                    "308": -0.030102274904848125,
                    "309": -0.08111589696357147,
                    "310": -0.10420084122098205,
                    "311": -0.10230368467674195,
                    "312": -0.047268835135119674,
                    "313": 0,
                    "314": -0.010532055716096278,
                    "315": -0.050176249145657825,
                    "316": -0.07480840537140365,
                    "317": 0,
                    "318": 0,
                    "319": 0,
                    "320": -0.20606664896508053,
                    "321": -0.011705283257045757,
                    "322": -0.14087024880710383,
                    "323": -0.015311255430417892,
                    "324": 0,
                    "325": 0,
                    "326": 0,
                    "327": -0.00031628096820596197,
                    "328": 0,
                    "329": 0,
                    "330": 0,
                    "331": -0.008657991405254306,
                    "332": -0.03658250019457254,
                    "333": 0,
                    "334": 0,
                    "335": 0,
                    "336": -0.13824205690170593,
                    "337": -0.10987408078906277,
                    "338": -0.0013698311426978693,
                    "339": -0.0045130431916172165,
                    "340": 0,
                    "341": 0,
                    "342": 0,
                    "343": 0,
                    "344": 0
                },
                "Target_AOV": {
                    "0": 201.16551749108737,
                    "1": 201.16551749108737,
                    "2": 271.6243443233095,
                    "3": 561.1722222222222,
                    "4": 354.25066499893364,
                    "5": 295.0270344396344,
                    "6": 266.8233333333334,
                    "7": 410.5348767507003,
                    "8": 146.6371287878788,
                    "9": 146.6371287878788,
                    "10": 230.00952105102294,
                    "11": 385.2891792929293,
                    "12": 223.28879283424268,
                    "13": 216.6528961038961,
                    "14": 436.28333333333336,
                    "15": 286.1444185463659,
                    "16": 177.22962393162393,
                    "17": 63.34775,
                    "18": 312.4839248511905,
                    "19": 454.3768708775501,
                    "20": 260.36454034391534,
                    "21": 317.66469387755103,
                    "22": 319.81714285714287,
                    "23": 122.9239,
                    "24": 124.2910884920635,
                    "25": 128.14284561011905,
                    "26": 310.9936111111112,
                    "27": 646.8640493827161,
                    "28": 139.32079147727273,
                    "29": 58.57,
                    "30": 451.1848529411765,
                    "31": 76.016,
                    "32": 120.47999999999999,
                    "33": 120.47999999999999,
                    "34": 346.42338541666663,
                    "35": 227.6,
                    "36": 381.83276785714287,
                    "37": 145.44,
                    "38": 123.2625,
                    "39": 145.44,
                    "40": 0,
                    "41": 0,
                    "42": 101.97432098765434,
                    "43": 134.53692307692307,
                    "44": 350.9514138321996,
                    "45": 126.55933333333333,
                    "46": 29.981463414634145,
                    "47": 138.58666666666667,
                    "48": 220.35477813318076,
                    "49": 220.35477813318076,
                    "50": 122.51783054803188,
                    "51": 350.9198174603174,
                    "52": 121.49420833333333,
                    "53": 149.30354166666666,
                    "54": 456.59700000000004,
                    "55": 146.93278846153845,
                    "56": 79.2832855317035,
                    "57": 79.08381382524902,
                    "58": 99.8679965140298,
                    "59": 171.51331111111114,
                    "60": 119.96299673992087,
                    "61": 106.26621212121212,
                    "62": 76.685,
                    "63": 88.63910110524397,
                    "64": 74.90400000000001,
                    "65": 74.90400000000001,
                    "66": 363.5391721363367,
                    "67": 676.2246960886005,
                    "68": 549.834474038359,
                    "69": 269.55793830128204,
                    "70": 148.12675438596492,
                    "71": 214.09501098901097,
                    "72": 100.83536260103607,
                    "73": 104.75783377669006,
                    "74": 154.71188392857144,
                    "75": 147.27633712121212,
                    "76": 220.83108843537414,
                    "77": 50.68933333333333,
                    "78": 110.43176470588234,
                    "79": 50.68933333333333,
                    "80": 88.82469696969697,
                    "81": 86.14032467532466,
                    "82": 89.26083319044824,
                    "83": 118.8256984126984,
                    "84": 112.5573735332821,
                    "85": 67.12172927689593,
                    "86": 150.4125,
                    "87": 86.9135956085956,
                    "88": 86.48970143115211,
                    "89": 87.46624285981252,
                    "90": 147.42654992734668,
                    "91": 442.5353546897547,
                    "92": 223.31138297571994,
                    "93": 224.3953781512605,
                    "94": 184.64875,
                    "95": 207.01767497519842,
                    "96": 166.726081010101,
                    "97": 166.726081010101,
                    "98": 189.36529560019676,
                    "99": 75.85814814814815,
                    "100": 175.93790365448504,
                    "101": 118.07708333333333,
                    "102": 300.7352333333333,
                    "103": 193.66815476190476,
                    "104": 150.88868606990494,
                    "105": 151.79940760022257,
                    "106": 161.43266199422865,
                    "107": 365.8333333333333,
                    "108": 133.5034813213103,
                    "109": 153.31991666666667,
                    "110": 83.9984126984127,
                    "111": 87.46211111111111,
                    "112": 148.53366359877285,
                    "113": 90.00171552347983,
                    "114": 205.47488148479425,
                    "115": 636.2333333333333,
                    "116": 264.5773516600224,
                    "117": 290.5954699657641,
                    "118": 64.89999999999999,
                    "119": 460.2583058608058,
                    "120": 206.54599215262664,
                    "121": 205.49673727113878,
                    "122": 271.96081147196,
                    "123": 804.3293333333334,
                    "124": 395.3513587363074,
                    "125": 292.96228783634666,
                    "126": 266.8233333333334,
                    "127": 388.85647806156265,
                    "128": 216.1373295440508,
                    "129": 216.6082706496253,
                    "130": 259.74105056475264,
                    "131": 685.2257981341576,
                    "132": 308.1573712009565,
                    "133": 244.86249597700436,
                    "134": 436.28333333333336,
                    "135": 400.25702611924197,
                    "136": 155.0259638130137,
                    "137": 154.8797707470646,
                    "138": 283.88354301079414,
                    "139": 635.0797936854813,
                    "140": 329.33409092814946,
                    "141": 210.72824514017728,
                    "142": 319.81714285714287,
                    "143": 360.2052922616599,
                    "144": 212.60847504689755,
                    "145": 213.7208820867404,
                    "146": 196.94950583353466,
                    "147": 689.0776003734827,
                    "148": 334.1612191247049,
                    "149": 132.02641465336134,
                    "150": 451.1848529411765,
                    "151": 143.34216680839003,
                    "152": 585.5067897938077,
                    "153": 591.1947113396387,
                    "154": 317.5917137332555,
                    "155": 395.967,
                    "156": 465.10773269363483,
                    "157": 215.66242640692641,
                    "158": 123.2625,
                    "159": 309.18492424242424,
                    "160": 442.0884851779484,
                    "161": 437.5831384081471,
                    "162": 270.2410890652557,
                    "163": 371.0287839506173,
                    "164": 223.80739612768184,
                    "165": 101.28253002070393,
                    "166": 29.981463414634145,
                    "167": 174.65997354497355,
                    "168": 257.18949320632794,
                    "169": 256.90466802151553,
                    "170": 176.25062534917703,
                    "171": 423.19852272727275,
                    "172": 231.5160004859856,
                    "173": 186.61897403022095,
                    "174": 456.59700000000004,
                    "175": 243.58963708740805,
                    "176": 73.00313478407557,
                    "177": 72.73017319560104,
                    "178": 89.3359727348854,
                    "179": 196.2018735632184,
                    "180": 88.99321866899417,
                    "181": 90.53336253561253,
                    "182": 76.685,
                    "183": 91.08039795521938,
                    "184": 236.00550689594422,
                    "185": 233.2612565725431,
                    "186": 414.21791008810317,
                    "187": 676.2246960886005,
                    "188": 592.3666087540577,
                    "189": 308.2385800865801,
                    "190": 148.12675438596492,
                    "191": 440.29041008991004,
                    "192": 130.51310406541305,
                    "193": 128.9446315447589,
                    "194": 190.8213095402912,
                    "195": 299.00981120731126,
                    "196": 216.77286340876378,
                    "197": 128.0902,
                    "198": 110.43176470588234,
                    "199": 119.55272995522995,
                    "200": 75.60509609048843,
                    "201": 74.3974865098527,
                    "202": 92.62309651453708,
                    "203": 117.62281077694237,
                    "204": 102.39451730554822,
                    "205": 63.63404351395731,
                    "206": 150.4125,
                    "207": 78.6691376573618,
                    "208": 99.92291666644,
                    "209": 99.17305985459893,
                    "210": 130.79605714063533,
                    "211": 340.4947619047619,
                    "212": 172.18321072556543,
                    "213": 117.55017948717949,
                    "214": 184.64875,
                    "215": 117.85570021645022,
                    "216": 151.39395595238096,
                    "217": 151.39395595238096,
                    "218": 183.09883575507067,
                    "219": 95.56589743589744,
                    "220": 191.21592075382057,
                    "221": 152.78211419753086,
                    "222": 300.7352333333333,
                    "223": 365.9526028782725,
                    "224": 123.3720840443606,
                    "225": 126.38176413400282,
                    "226": 134.94160628237915,
                    "227": 217.87470075757577,
                    "228": 165.23733745572062,
                    "229": 85.25645124857222,
                    "230": 83.9984126984127,
                    "231": 125.10254087810337,
                    "232": 125.25741926033957,
                    "233": 124.83802754201076,
                    "234": 194.67662311436632,
                    "235": 744.6470833333333,
                    "236": 266.4119919833452,
                    "237": 296.7022775357975,
                    "238": 64.89999999999999,
                    "239": 598.5526960252633,
                    "240": 176.41749598930483,
                    "241": 185.9089962978198,
                    "242": 391.62461098398165,
                    "243": 554.0933333333334,
                    "244": 416.52103658536583,
                    "245": 231.5291283643892,
                    "246": 270.7911288677447,
                    "247": 435.9500727272727,
                    "248": 435.9500727272727,
                    "249": 328.273437558385,
                    "250": 900.1669317848817,
                    "251": 404.4514065359101,
                    "252": 438.90054124425984,
                    "253": 330.31610103891325,
                    "254": 134.53736312217194,
                    "255": 132.2330274321267,
                    "256": 242.15871414165224,
                    "257": 779.8124761904762,
                    "258": 326.8904833014192,
                    "259": 192.9053691796709,
                    "260": 361.0336546472005,
                    "261": 257.12082621873907,
                    "262": 255.68021331169868,
                    "263": 218.629039994458,
                    "264": 1152.571461038961,
                    "265": 350.05194505470075,
                    "266": 176.28523674242425,
                    "267": 263.9474918068043,
                    "268": 672.9649641658024,
                    "269": 672.9649641658024,
                    "270": 365.8356972245606,
                    "271": 1578.8362727272727,
                    "272": 424.5937053186191,
                    "273": 179.4046643518519,
                    "274": 201.41182344276095,
                    "275": 471.11519047619043,
                    "276": 471.11519047619043,
                    "277": 462.2783680555556,
                    "278": 1137.07,
                    "279": 462.2783680555556,
                    "280": 325.24897959183676,
                    "281": 87.05333333333333,
                    "282": 218.80547350456874,
                    "283": 213.98793537756396,
                    "284": 190.41082369392677,
                    "285": 565.9325425101214,
                    "286": 249.10750132344364,
                    "287": 175.6033258526128,
                    "288": 246.2133916194298,
                    "289": 98.89311711311473,
                    "290": 98.89311711311473,
                    "291": 94.33601615553282,
                    "292": 137.50273333333334,
                    "293": 110.20665387652313,
                    "294": 99.01442735042734,
                    "295": 108.8980293040293,
                    "296": 251.82421376209277,
                    "297": 252.46741681112493,
                    "298": 449.50515250946745,
                    "299": 1554.350315619968,
                    "300": 607.8365261064233,
                    "301": 262.00624999999997,
                    "302": 272.56062499999996,
                    "303": 143.31301884340482,
                    "304": 143.31301884340482,
                    "305": 165.36891218115406,
                    "306": 328.84236666666663,
                    "307": 209.02499169755922,
                    "308": 133.63230368589745,
                    "309": 159.37815075549452,
                    "310": 103.76987472628113,
                    "311": 103.76987472628113,
                    "312": 97.83468270676693,
                    "313": 174.99562500000002,
                    "314": 141.7261495756905,
                    "315": 49.691066666666664,
                    "316": 49.691066666666664,
                    "317": 122.08176097105508,
                    "318": 130.93815272847107,
                    "319": 195.22020030501415,
                    "320": 520.7064285714285,
                    "321": 189.40911348002905,
                    "322": 225.62472727272726,
                    "323": 167.04997474747475,
                    "324": 188.92492637187036,
                    "325": 188.92492637187036,
                    "326": 190.87501016022114,
                    "327": 341.56218750000005,
                    "328": 212.9830934837833,
                    "329": 152.0372883490778,
                    "330": 140.80598484848485,
                    "331": 102.79285001920844,
                    "332": 100.62196738932121,
                    "333": 143.5478407121915,
                    "334": 330.84536079545455,
                    "335": 165.16609518404766,
                    "336": 81.20533333333333,
                    "337": 120.51945766178265,
                    "338": 197.71070838699404,
                    "339": 192.24051181031524,
                    "340": 279.9935924551639,
                    "341": 1977.3076923076924,
                    "342": 578.7943508816765,
                    "343": 406.0327721088435,
                    "344": 768.4245824763473
                }
                }        





        df_targets_final = pd.DataFrame(dict_targets) 
    


        df_targets_final['Key'] =  df_targets_final['Tipo'] + df_targets_final['Categoria'] + df_targets_final['Region'] + df_targets_final['Size']
        df_targets_final = df_targets_final[['Key','% Target Positivação','Target_AOV']]
        df_targets_final = df_targets_final.set_index('Key')

        
        df_categoria_final = df_categoria_final.merge( df_targets_final, how='left', left_index=True, right_index=True)  
        


        df_categoria_final['Delta % Positivação'] = np.where(df_categoria_final['% Target Positivação'] < df_categoria_final['% Positivação Categoria'], 0 , df_categoria_final['% Target Positivação'] - df_categoria_final['% Positivação Categoria']  )
        df_categoria_final['Oportunidade Gmv'] = (df_categoria_final['Delta % Positivação'] * df_categoria_final['Forecast Positivação Geral']  )  * df_categoria_final['Target_AOV']

        df_categoria_final['% Ating Categoria'] =  df_categoria_final['% Positivação Categoria'] /df_categoria_final['% Target Positivação']
        df_categoria_final = df_categoria_final.set_index('DateHour')
        df_categoria_final  = df_categoria_final.sort_index(ascending = False)

        return df_categoria_final


    df_categoria_final = load_df_categoria_Final()
    


     
    df_resumo_categoria = df_categoria_final.copy()
    df_resumo_categoria['Date'] = df_resumo_categoria.index.date
    df_resumo_categoria =  df_resumo_categoria[df_resumo_categoria['Date'] == date_ref]
    df_resumo_categoria = df_resumo_categoria[['Categoria','Regional','Size','Ofertão','Gmv Acum','Oportunidade Gmv','AOV','Target_AOV', 'Positivação Categoria','% Positivação Categoria','% Target Positivação','% Ating Categoria']] 
    df_resumo_categoria = df_resumo_categoria.sort_values('Oportunidade Gmv', ascending = False)



    st.markdown("### Geral") 
    st.markdown("### ") 
 
    df_resumo_categoria

    st.markdown("### ") 

 
    st.markdown("### Filtros") 
    st.markdown("### ") 

    with st.form(key = "my_forms_resumo"):
        
        col = st.columns((2,  2, 8 ), gap='medium')
        with col[0]:
                        
            regional_list =   st.radio('Região', ['RJC', 'RJI','BAC']  )


        with col[1]:

            size_list =   st.radio('Size', ['1-4 Cxs','5-9 Cxs' , 'size']  )


        submit_button_categoria = st.form_submit_button(label = "Submit")



    regional_list = [regional_list]
    size_list = [size_list]


    df_resumo_filtro = df_resumo_categoria.copy()
    df_resumo_filtro = df_resumo_filtro[df_resumo_filtro['Regional'] ==  regional_list[0]]
    df_resumo_filtro = df_resumo_filtro[df_resumo_filtro['Size'] ==   size_list[0]]
    
    df_resumo_filtro = df_resumo_filtro.sort_values('Oportunidade Gmv', ascending = False) 
    df_resumo_filtro

    categoria_list_loop = df_resumo_filtro[['Categoria']].set_index('Categoria').T.columns.to_list() 
    
 
    for i in regional_list:
        
            
        for s in size_list:


            df_plot = df_categoria_final.copy() 
            df_plot = df_plot.reset_index() 
            df_plot = df_plot.set_index('DateHour')  
            df_plot['Weekday'] = df_plot.index.weekday
            weekday = date_ref.weekday()
            
            df_plot = df_plot[df_plot['Regional']== i ]  
            df_plot = df_plot[df_plot['Size']== s]   
            df_plot = df_plot[df_plot['Weekday']== weekday]   
     

            df_plot = df_plot[['Categoria','Regional','Size','Gmv Acum','Oportunidade Gmv','Positivação Categoria','% Positivação Categoria','% Target Positivação','% Ating Categoria']] 
            

            if s == 'size': size_name = 'Geral' 
            else: size_name = s 

            nome_var =  i + ' - ' + size_name  

 

            
            st.markdown("##  " +  nome_var)
            st.markdown("##  "  )


            for c in  categoria_list_loop:


                st.markdown("### " + c) 
                st.markdown("###") 
                df_categoria = df_plot.copy()
                df_categoria = df_categoria[df_plot['Categoria']== c]   
                 

                dados_x =  df_categoria.index
                dados_y1 =  df_categoria['Gmv Acum'] 
                dados_y2 =  df_categoria['Positivação Categoria']
                dados_y3 =  df_categoria['% Positivação Categoria']
                    


                # df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
                # dados_x_forecast =  df_plot_forecast.index
                # #dados_y3 = df_plot_forecast['Forecast Gmv'] 
                
                col = st.columns((2,  2, 2), gap='medium')

                with col[0]:
                    
                    fig=py.line(x=dados_x, y=dados_y1,   title = 'Gmv' ,  labels=dict(y="Gmv" , x="Data", z = 'Hora') , height=300, width= 350, markers = True,    line_shape='spline')

                    fig 

                with col[1]:
                    
                    fig=py.line(x=dados_x, y=dados_y2,   title = 'Positivação' ,  labels=dict(y="Positivação", x="Data" ) , height=300, width= 350, markers = True,    line_shape='spline')

                    fig      
                
                with col[2]:
                    
                    fig=py.line(x=dados_x, y=dados_y3,   title = '% Positivação Categoria' ,  labels=dict(y="% Positivação Categoria" , x="Data") , height=300, width= 350, markers = True,    line_shape='spline')
                    #fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

                    fig        



            #     col = st.columns((2,  8), gap='medium') 

            #     with col[0]:
            #         df_plot
                                
            #         # df_trend_slope = df_trend_slope.rename(columns = {'Slope 1': 'Slope'})
            #         # df_trend_slope

            #         # df_trend_slope = df_trend_slope.rename(columns = {'Slope': 'Slope ' + i + ' - ' + size_name})

            #         # if len(df_trend_slope_final) == 0: 
            #         #     df_trend_slope_final = df_trend_slope.copy()                                
            #         # else:
            #         #     df_trend_slope_final = df_trend_slope_final.merge( df_trend_slope, how='left', left_index=True, right_index=True)  
                
            #     with col[1]:
            #         df_plot
            #         #plot_trends(df_plot,categoria_filter, var, 'Categoria')




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

 

    Rupturas , Trend       = st.tabs(["Rupturas",  "Trend"  ])
    
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

                             
                            df_plot = df_plot[['Categoria','Gmv Acum','Positivação Categoria', 'Positivação Geral', '% Positivação Categoria']]
                            col_lista = ['% Positivação Categoria', 'Gmv Acum','Positivação Categoria', 'Positivação Geral']

                            df_plot =  pd.DataFrame(df_plot.asfreq('d').index).set_index('Data').merge(df_plot, left_index = True, right_index=True,how = "left")
                        

                            df_plot[ col_lista  ] = df_plot[col_lista].fillna(method='ffill')
                            
                            df_plot_guarda = df_plot.copy()

                            df_plot = df_plot[ [var_change_points ]] 
                            df_plot['Categoria'] = c  
                             
               
                            df_plot, df_trend_slope ,var = df_plot_trend(df_plot,[c], var_change_points, 'Categoria')
                            
                            
                            df_plot = df_plot['Trend ' + c]
 
                            data = df_plot.copy()  

                             

                            df_plot_guarda = df_plot_guarda.merge(df_plot, left_index = True, right_index=True,how = "left")
                             

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
                             
 

                            df_plot = df_plot_guarda.copy()
                            df_plot = df_plot.reset_index()  
                            df_plot['Categoria'] = c
                            df_plot['Size'] = s
                            df_plot['Region'] = i  
                            df_plot['Tipo'] = tipo  
                            df_plot = df_plot.rename(columns = {'Trend ' + c : 'Trend'})
                            df_plot = df_plot[['Data','Tipo','Categoria','Region','Size','Trend','% Positivação Categoria','Positivação Categoria', 'Positivação Geral' , 'Gmv Acum'  ]]
                            df_plot['Aov'] = np.where(df_plot['Gmv Acum']>0, df_plot['Gmv Acum']/df_plot['Positivação Categoria'] , 0 )
                            
                            
                            df_plot['index_cp'] = -1
                            
                                         
                            for t in trend_groups['index'].to_list():
                                
                                data1 = trend_groups.copy()
                                data1 = data1[data1['index'] == t] 
                                data1 = data1['Data'].values[0] 
                                data1 = str(data1)[0:10] 
                                data1 = datetime.datetime.strptime( data1 , "%Y-%m-%d") 
                               # data1 = data1.date() 

                                data2 = trend_groups.copy()
                                data2 = data2[data2['index'] == t] 
                                data2 = data2['Data2'].values[0] 
                                data2 = str(data2)[0:10] 
                                data2 = datetime.datetime.strptime( data2 , "%Y-%m-%d") 
                               # data2 = data2.date() 
 
                                 
                                df_plot['index_cp'] = np.where((df_plot['Data'] >= data1) &  (df_plot['Data'] <=data2), t, df_plot['index_cp'])
                                  
                             
                            df_group = df_plot[['index_cp','Aov','Positivação Categoria','Positivação Geral']].groupby('index_cp').mean()
                             
                            
                            trend_groups = trend_groups.reset_index(drop = False)
                            trend_groups = trend_groups.set_index('index')
                            trend_groups = trend_groups.merge( df_group, how='left', left_index=True, right_index=True)
                            trend_groups = trend_groups.reset_index(drop = False)
                            trend_groups = trend_groups.set_index('Change Point Number')
 


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
 

        @st.cache_resource( ttl = 1800) 
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
        
        st.markdown("####  Targets Todos"  )

        df_target = df_categoria_target.copy()
         

    
        df_target['Aov 1'] =  np.where(df_target['index']==0, df_target['Aov'] ,0)
        df_target['Aov 2'] =  np.where(df_target['index']==1, df_target['Aov'] ,0)

        df_target['Positivação Geral 1'] =  np.where(df_target['index']==0, df_target['Positivação Geral'] ,0)
        df_target['Positivação Geral 2'] =  np.where(df_target['index']==1, df_target['Positivação Geral'] ,0)

        df_target = df_target[['Tipo','Categoria','Region','Size','Trend Atual','Trend Top 1','Trend Top 2','Aov 1','Aov 2','Positivação Geral 1','Positivação Geral 2']].groupby(['Tipo','Categoria','Region','Size']).max()

        df_target['Delta Top 1'] =   np.where(  df_target['Trend Atual'] < df_target['Trend Top 1']  ,  df_target['Trend Atual'] -  df_target['Trend Top 1']  ,  0 )
        df_target['Delta Top 2'] =   np.where(  df_target['Trend Atual'] < df_target['Trend Top 2']  ,  df_target['Trend Atual'] -  df_target['Trend Top 2']  ,  0 )
        df_target['Delta Médio'] = (df_target['Delta Top 1'] +  df_target['Delta Top 2'])/2
        

        df_action = df_target.copy()
        df_action = df_action.reset_index(drop = False)
        df_action['Trend Top']  = df_action['Trend Top 2'] 
        df_action['Delta Top']  = df_action['Delta Top 2']  
        df_action = df_action[['Tipo','Categoria','Region','Size','Trend Atual','Trend Top','Delta Top' ,'Aov 2','Positivação Geral 1', 'Positivação Geral 2']]
        
        
        df_action = df_action.sort_values('Delta Top', ascending = True)
        df_action



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


                        fig.update_layout(title=var_tipo, width = 450, height = 300)

                
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

                        fig.update_layout(title=var_tipo, width = 450, height =300)
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

                        fig.update_layout(title= var_tipo, width = 450, height =300)
                        st.plotly_chart(fig)

                        
                    # with col[1]: 

                    #     df_categoria_plot

                    # with col[2]: 

                    #     df_categoria_plot
 



  
        st.markdown("###  Target "  ) 

        
        tipo_list = st.radio("Tipo", ['Geral', 'Bau','Ofertão'] )


        st.markdown("#####  "  )      
        
        col0, col1  = st.columns(2)  
        
         
        
        for r in ['RJC', 'RJI','BAC']:
            
            df_action_region = df_action[df_action['Region']== r] 

            df_action_1_4 = df_action_region[['Size','Tipo','Categoria','Trend Atual','Trend Top','Delta Top','Aov 2']]  
            df_action_1_4 = df_action_1_4[df_action_1_4['Size']=='1-4 Cxs'] 
            df_action_1_4 = df_action_1_4[df_action_1_4['Tipo']== tipo_list] 

            
            df_action_1_4 = df_action_1_4.set_index('Size')
            df_action_1_4 = df_action_1_4.sort_values('Delta Top', ascending = True)
    
            df_action_5_9 = df_action_region[['Size','Tipo','Categoria','Trend Atual','Trend Top','Delta Top','Aov 2']]  
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





 
        # df_target['Target'] = np.where(df_target['Trend'] ==  df_target['Trend Atual'], df_target['Trend Atual']*1.10,   df_target['Trend']  )
        # df_target['Target'] = np.where(df_target['Target'] >=  df_target['Trend Atual']*1.1, df_target['Trend Atual']*1.1,   df_target['Trend']  )
            


        # st.markdown("####  Base"  )
        # df_target = df_target.reset_index(drop = False)
        # df_target
