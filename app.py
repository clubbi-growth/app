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
    
 
tab0   , tab1   = st.tabs(["Performance D0","Categorias" ])
df_count = 0 

button_count = 0   

# %% Categorias 

with tab0:
 
    
 

    # for key, value in cached_data.items():

    #     df_count = df_count + 1 
 
    #     if df_count == 1:

    #         st.markdown('#### ' + cached_data[key]['Categoria'].unique()[0])  

    #         df_plot =  cached_data[key].copy()   
    #         df_plot = df_plot.reset_index(drop = False)
    #         df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #         df_plot = df_plot.set_index('DateHour') 

 

    #         df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #         df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 

    #         if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
    #         if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]

 
    #         dados_x =  df_plot.index
    #         dados_y1 =  df_plot['Positivação Categoria']
    #         dados_y2 =  df_plot['Gmv Acum'] 

    #         df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
    #         dados_x_forecast =  df_plot_forecast.index
    #         dados_y3 = df_plot_forecast['Forecast Gmv'] 
             
    #         col = st.columns((2,  2, 2), gap='medium')

    #         with col[0]:
                
    #             fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação" , x="Data") , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig 

    #         with col[1]:
                
    #             fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum", x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig      
            
    #         with col[2]:
                
    #             fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig        


    #         with st.expander('Detalhes', expanded= False):
    #             st.markdown('#### ' + cached_data["df_RJ_1_4"]['Categoria'].unique()[0])  

            
    #             df_plot =  cached_data['df_RJ_1_4'].copy()   
    #             df_plot = df_plot.reset_index(drop = False)
    #             df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #             df_plot = df_plot.set_index('DateHour') 

    #             df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #             df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 

    #             if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
    #             if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
 
    #             dados_x =  df_plot.index
    #             dados_y1 =  df_plot['Positivação Categoria']
    #             dados_y2 =  df_plot['Gmv Acum'] 

    #             df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
    #             dados_x_forecast =  df_plot_forecast.index
    #             dados_y3 = df_plot_forecast['Forecast Gmv'] 
                
                
    #             col = st.columns((2,  2, 2), gap='medium')

    #             with col[0]:
                    
    #                 fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação", x="Data", ) , height=300, width= 450, markers = True,    line_shape='spline')

    #                 fig 

    #             with col[1]:
                    
    #                 fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum", x="Hora") , height=300, width= 450, markers = True,    line_shape='spline')

    #                 fig      
                
    #             with col[2]:
                    
    #                 fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(x="Data", y="Forecast Gmv") , height=300, width= 450, markers = True,    line_shape='spline')

    #                 fig      

    #             st.markdown('#### ' + cached_data["df_RJ_5_9"]['Categoria'].unique()[0])  

    #             df_plot =  cached_data['df_RJ_5_9'].copy()   
    #             df_plot = df_plot.reset_index(drop = False)
    #             df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #             df_plot = df_plot.set_index('DateHour') 

    #             if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
    #             if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
    #             df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #             df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 


    #             #df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
    #             dados_x =  df_plot.index
    #             dados_y1 =  df_plot['Positivação Categoria']
    #             dados_y2 =  df_plot['Gmv Acum'] 

    #             df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
    #             dados_x_forecast =  df_plot_forecast.index
    #             dados_y3 = df_plot_forecast['Forecast Gmv']  
                
    #             col = st.columns((2,  2, 2), gap='medium')



    #             with col[0]:
                    
    #                 fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação", x="Data") , height=300, width= 450, markers = True,    line_shape='spline')

    #                 fig 

    #             with col[1]:
                    
    #                 fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Hora" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #                 fig      
                
    #             with col[2]:
                    
    #                 fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #                 fig      

  
    #     elif df_count == 4:
    #         st.markdown('#### ' + cached_data[key]['Categoria'].unique()[0])  

    #         df_plot =  cached_data[key].copy()   
    #         df_plot = df_plot.reset_index(drop = False)
    #         df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #         df_plot = df_plot.set_index('DateHour') 

    #         if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
    #         if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
    #         df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #         df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 


    #        # df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
    #         dados_x =  df_plot.index
    #         dados_y1 =  df_plot['Positivação Categoria']
    #         dados_y2 =  df_plot['Gmv Acum'] 

    #         df_plot_forecast = df_plot[df_plot.index >= pd.Timestamp(data_min) +  pd.offsets.Day(30)]
    #         dados_x_forecast =  df_plot_forecast.index
    #         dados_y3 = df_plot_forecast['Forecast Gmv'] 
            
            
    #         col = st.columns((2,  2, 2), gap='medium')

    #         with col[0]:
                
    #             fig=py.line(x=dados_x, y=dados_y1,   title = 'Positivação' ,  labels=dict(y="Positivação Categoria" , x="Data") , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig 

    #         with col[1]:
                
    #             fig=py.line(x=dados_x, y=dados_y2,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Hora") , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig      
            
    #         with col[2]:
                
    #             fig=py.line(x=dados_x_forecast, y=dados_y3,  title = 'Forecast Gmv' ,  labels=dict(y="Forecast Gmv" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig        


         
    #     elif df_count >= 5 : 
            
    #         if df_count==5: st.header("Categoria")   

    #         st.markdown('#### ' + cached_data[key]['Categoria'].unique()[0])  
    #         col = st.columns((2,  2, 2), gap='medium')
    #         with col[0]:
                
                            
    #             df_plot =  cached_data[key].copy()   
    #             df_plot = df_plot.reset_index(drop = False)
    #             df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #             df_plot = df_plot.set_index('DateHour') 

    #             if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]

    #             if hora_list[0] != 'Hora': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]


    #             df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #             df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 
    #            # df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
    #             dados_x =  df_plot.index
 
    #             dados_y =  df_plot['Positivação Categoria']
    #             dados_y2 =  df_plot['% Positivação Categoria'] 
    #             dados_y3 =  df_plot['Gmv Acum'] 
    #             fig=py.line(x=dados_x, y=dados_y,   title = 'Positivação Categoria' ,  labels=dict(y="Positivação", x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig 

    #         with col[1]:
                
    #             fig=py.line(x=dados_x, y=dados_y3,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig     
            
    #         with col[2]:
                
    #             fig=py.line(x=dados_x, y=dados_y2,  title = '% Positivação Categoria' ,  labels=dict(y="% Positivação" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #             fig        
    
  
    #         # with st.expander('Detalhes', expanded= False):

    #         #     var = 10 


    #             # st.markdown('#### Métricas Tráfego Categoria' )   

 
    #             # col = st.columns((2,  2), gap='medium')
    #             # with col[0]:
                    
                                
    #             #     df_plot =  cached_data[key].copy()
    #             #     df_plot = df_plot.reset_index(drop = False)
    #             #     df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #             #     df_plot = df_plot.set_index('DateHour') 
        

    #             #     if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
    #             #     if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]

    #             #     df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #             #     df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 
    #             #     #df_plot = df_plot[df_plot['Hora'] == max_hora_trafego]
            
                    
    #             #     dados_x =  df_plot.index 
    #             #     dados_y =  df_plot['search_products Acum']
    #             #     dados_y2 =  df_plot['% Conversão Search Acum'] 
    #             #     fig=py.line(x=dados_x, y=dados_y,   title = 'Search Products' ,  labels=dict(y="Search Product", x="Data" ) , height=300, width= 500, markers = True,    line_shape='spline')

    #             #     fig 

    #             # with col[1]:
                    
    #             #     fig=py.line(x=dados_x, y=dados_y2,  title = '% Conversão Search Acum' ,  labels=dict(y="% Conversão Search Acum" , x="Hora") , height=300, width= 500, markers = True,    line_shape='spline')

    #             #     fig     

                

    #         categoria_atual = cached_data[key]['Categoria'].unique()[0]     

    #         for key in buttons_dic:

                  

    #             if key == 'Produtos ' + categoria_atual: 
                
                    
    #                 buttons_dic[key] = st.checkbox('Detalhe ' + key)
                    
                    
    #                 if buttons_dic[key]:  

                            
    #                     st.markdown('#### Métricas Top Skus' )  
                        
    #                     st.markdown('#### ' )  


    #                     top_skus_atual = df_orders[df_orders['Categoria'] == categoria_atual ][df_orders['unit_ean_prod'].isin(top_skus)]['unit_ean_prod'].unique().tolist()
                            
    #                     for k in range(0,len(top_skus_atual)):

    #                         produto = df_orders[df_orders['Categoria'] == categoria_atual ][df_orders['unit_ean_prod'] == top_skus_atual[k]]['Produtos'].unique()[0]
    #                         ean_prod = df_orders[df_orders['Categoria'] == categoria_atual ][df_orders['unit_ean_prod'] == top_skus_atual[k]]['unit_ean_prod'].unique()[0]
    #                         st.markdown('##### ' + produto )                            
                                
                                            
    #                         df_prod = cria_df_view_categoria2(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],[top_skus_atual[k]]) 
                                

    #                         df_plot =  df_prod.copy()   
    #                         df_plot = df_plot.reset_index(drop = False)
    #                         df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #                         df_plot = df_plot.set_index('DateHour')  

    #                         col = st.columns((2,  2, 2), gap='medium')

    #                         with col[0]:
                                
                                            
    #                             df_plot =  df_prod.copy()   
    #                             df_plot = df_plot.reset_index(drop = False)
    #                             df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
    #                             df_plot = df_plot.set_index('DateHour') 
                    

    #                             if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
    #                             if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]


    #                             df_plot = df_plot[df_plot.index >= pd.Timestamp(data_min)]
    #                             df_plot = df_plot[df_plot.index <= pd.Timestamp(data_max)] 
    #                             #  df_plot = df_plot[df_plot['Hora'] == max_hora_orders]
                                
    #                             dados_x =  df_plot.index
    #                             dados_y =  df_plot['Positivação Categoria']
    #                             dados_y2 =  df_plot['% Share Positivação Categoria'] 
    #                             dados_y3 =  df_plot['Gmv Acum'] 
    #                             fig=py.line(x=dados_x, y=dados_y,   title = 'Positivação Categoria' ,  labels=dict(y="Positivação", x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #                             fig 

    #                         with col[1]:
                                
    #                             fig=py.line(x=dados_x, y=dados_y3,   title = 'Gmv' ,  labels=dict(y="Gmv Acum" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #                             fig     
                            
    #                         with col[2]:
                                
    #                             fig=py.line(x=dados_x, y=dados_y2,  title = '% Positivação Categoria' ,  labels=dict(y="% Positivação" , x="Data" ) , height=300, width= 450, markers = True,    line_shape='spline')

    #                             fig    


    # st.markdown("###") 



    hora_atual = df_categorias2[df_categorias2['Date'] == datetime.date.today() ]['Hora'].max() -1 


    st.markdown('###### Atualizado em: ' + str(hora_atualizacao) + ' / Hora Filtrada: ' + str(hora_atual))  
    



    @st.cache_resource( ttl = 1000) 
    def load_df_categoria_Final():
        regional_list =  ['RJC', 'RJI','BAC']  
        size_list =     ['1-4 Cxs','5-9 Cxs' , 'size']   


        df_categoria_final = pd.DataFrame()

        for i in regional_list:
            
                
            for s in size_list:


                df_plot = df_categorias2.copy() 
                df_plot = df_plot.reset_index() 
                df_plot = df_plot.set_index('DateHour')  
                weekday = datetime.date.today().weekday()
                
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
                    df_categoria = df_categoria.rename(columns = {'% Positivação Categoria_lag_7D':'% Target Positivação'})
                    df_categoria = df_categoria.rename(columns = {'AOV_lag_7D':'Target AOV'})

                    df_categoria['Delta % Positivação'] = np.where(df_categoria['% Target Positivação'] < df_categoria['% Positivação Categoria'], 0 , df_categoria['% Target Positivação'] - df_categoria['% Positivação Categoria']  )
                    df_categoria['Oportunidade Gmv'] = (df_categoria['Delta % Positivação'] * df_categoria['Forecast Positivação Geral']  )  * df_categoria['Target AOV']
                    
                    df_categoria['Size'] = s
                    df_categoria['Regional'] = i
                    df_categoria['Categoria'] = c
    
                
                    if len(df_categoria_final) ==0: 
                        df_categoria_final = df_categoria.copy()
                    else:
                        df_categoria_final = pd.concat([df_categoria_final,  df_categoria.copy()])
                    

        df_categoria_final['% Ating Categoria'] =  df_categoria_final['% Positivação Categoria'] /df_categoria_final['% Target Positivação']

        return df_categoria_final


    df_categoria_final = load_df_categoria_Final()
    df_resumo_categoria = df_categoria_final.copy()
    df_resumo_categoria['Date'] = df_resumo_categoria.index.date
    df_resumo_categoria =  df_resumo_categoria[df_resumo_categoria['Date'] == datetime.date.today()]
    df_resumo_categoria = df_resumo_categoria[['Categoria','Regional','Size','Gmv Acum','Oportunidade Gmv','Positivação Categoria','% Positivação Categoria','% Target Positivação','% Ating Categoria']] 
    df_resumo_categoria = df_resumo_categoria.sort_values('Oportunidade Gmv', ascending = False)

    st.markdown("### Geral") 
    st.markdown("### ") 
 
    df_resumo_categoria

    st.markdown("### ") 

    col = st.columns((2,  2, 8 ), gap='medium')

    with st.expander('Filtros', expanded= False):

        with st.form(key = "my_forms"):
         
            with col[0]:
                            
                regional_list =   st.radio('Região', ['RJC', 'RJI','BAC']  )


            with col[1]:

                size_list =   st.radio('Size', ['1-4 Cxs','5-9 Cxs' , 'size']  )


            submit_button_categoria = st.form_submit_button(label = "Submit")



    regional_list = [regional_list]
    size_list = [size_list]

    st.markdown("### Filtros") 
    st.markdown("### ") 

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
            weekday = datetime.date.today().weekday()
            
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


# with tab1: 
                              
#     df_categorias2 = df_categorias2.rename(columns={'% Share Positivação Categoria':'% Positivação Categoria'})   
#     df_categorias2['Hora'] = df_categorias2.index.hour 
#     df_categorias2['Data'] = df_categorias2.index.date  
#     df_categorias2["week"] = df_categorias2.index.isocalendar().week
#     df_categorias2["day_of_month"] = df_categorias2.index.day
#     df_categorias2["month"] = df_categorias2.index.month

#     categoria_list = df_categorias2.sort_values('Categoria', ascending = True).Categoria.unique().tolist()
    
    

#     def df_plot_trend(df_plot, categoria_list,var, tipo):
        
#         trend_list = []     
#         dict_trends = {}
        

#         for i in categoria_list:
        
#             df_trend = df_plot.copy()
#             df_trend = df_trend[df_trend[tipo] == i]
                
#             df_trend =  pd.DataFrame(df_trend.asfreq('d').index).set_index('Data').merge(df_trend, left_index = True, right_index=True,how = "left")
            
#             df_trend = df_trend[[var]]
            
            

#             res = STL(  df_trend,  robust=True,  ).fit()


#             df_trend["residual"] = res.resid
#             df_trend["Trend " + i] = res.trend 
#             df_trend["seasonal"] = res.seasonal 
#             df_trend = df_trend.reset_index('Data')

#             slopeList = df_trend["Trend " + i].tolist()
#             slope1 = (slopeList[-1] - slopeList[0]) / (len(slopeList) - 1) 

#             # X = np.arange(len(slopeList)).reshape(-1, 1)  # Independent variable (time)
#             # y = slopeList
#             # model = LinearRegression()
#             # model.fit(X, y)
#             # slope2 = model.coef_[0] 
#             dict_trends = {tipo: i, 'Slope 1': slope1}
#             trend_list.append(dict_trends) 
            
#             df_trend = df_trend.set_index('Data')
            
#             df_plot = df_plot.merge(  df_trend[['Trend ' + i]], how='left', left_index=True, right_index=True)  
            
#             df_trend_slope = pd.DataFrame(trend_list)[[tipo,'Slope 1']].sort_values('Slope 1', ascending = True)
            
#             df_trend_slope  = df_trend_slope.set_index(tipo)[['Slope 1']]


#         return df_plot, df_trend_slope, var


#     def plot_trends(df_plot,categoria_filter, var, tipo ):

#         if len(categoria_filter)== 0: categoria_filter=[tipo]

#         df_plot_trend = df_plot.copy()

#         if categoria_filter[0] != tipo: df_plot_trend = df_plot_trend[df_plot_trend[tipo].isin(categoria_filter)]

#         lista_trend = df_plot.filter(regex='Trend').columns.to_list()  
#         lista_trend_final = []

#         for i in lista_trend: 
            
#             for k in categoria_filter:            
#                 if i[6:].find( k ,0)==0: lista_trend_final.append(i)


#         if len(lista_trend_final) !=0 : lista_trend = lista_trend_final


#         df_plot_trend = df_plot_trend[lista_trend].reset_index()

#         df_plot_trend = df_plot_trend.groupby(df_plot_trend['Data']).max()

#         df_plot_trend = df_plot_trend.reset_index()


    
#         fig = px.line(df_plot_trend, x="Data", y=df_plot_trend.columns , hover_data={"Data": "|%B %d, %Y"}, width=1200) #, title='Trend Categorias')
#         fig.update_xaxes(dtick="M1", tickformat="%b\n%Y")
#         st.plotly_chart(fig, theme="streamlit")

 

#     Trend   = st.tabs([ "Trend"   ])
    
# # %% Trend 
    
#     with Trend: 
        
#         with st.expander('Filtros', expanded= False):

#             with st.form(key = "my_form"):
                
#                 col = st.columns((4, 6), gap='medium')
                
#                 with col[0]:
#                     col = st.columns((2, 2), gap='medium')

#                     with col[0]:

#                         var_trends = ['% Positivação Categoria', 'Positivação Categoria', 'Gmv Acum' ]
#                         var = st.radio("Métrica", var_trends ) 

#                     with col[1]:
#                         categoria_filter = st.multiselect("Categoria", categoria_list,)

#                 submit_button = st.form_submit_button(label = "Submit")
         

#         regional_list =  ['RJC', 'RJI','BAC']  
#         size_list = ['size','1-4 Cxs','5-9 Cxs']   
#         df_trend_slope_final = pd.DataFrame()

#         for i in regional_list:
            
              
#             for s in size_list:


#                 df_plot = df_categorias2.copy()
#                 df_plot = df_plot.reset_index() 
#                 df_plot = df_plot.set_index('Data') 
#                 df_plot = df_plot[df_plot['Hora']==23]  

#                 df_plot = df_plot[df_plot['Region']== i ]  
#                 df_plot = df_plot[df_plot['Size']== s]   
        

#                 if s == 'size': size_name = 'Geral' 
#                 else: size_name = s 

#                 nome_var =  i + ' - ' + size_name + ' - Trend ' + var  
 

#                 df_plot, df_trend_slope ,var = df_plot_trend(df_plot,categoria_list, var, 'Categoria')

 
#                 if   i + ' - ' + size_name != 'BAC - 5-9 Cxs':


                    
#                     st.markdown("##  " +  nome_var)
#                     st.markdown("##  "  )

#                     col = st.columns((2,  8), gap='medium') 

#                     with col[0]:
                                    
#                         df_trend_slope = df_trend_slope.rename(columns = {'Slope 1': 'Slope'})
#                         df_trend_slope

#                         df_trend_slope = df_trend_slope.rename(columns = {'Slope': 'Slope ' + i + ' - ' + size_name})

#                         if len(df_trend_slope_final) == 0: 
#                             df_trend_slope_final = df_trend_slope.copy()                                
#                         else:
#                             df_trend_slope_final = df_trend_slope_final.merge( df_trend_slope, how='left', left_index=True, right_index=True)  
                    
#                     with col[1]:
            
#                         plot_trends(df_plot,categoria_filter, var, 'Categoria')


#         st.markdown("##  Resumo Slopes")
#         st.markdown("##  "  )
        
#         df_trend_slope_final


 
