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

#%% Df Order Previsão


query_order_previsao  = "select \
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
and DATE(ord.order_datetime) >= '2022-01-01' \
and DATE(ord.order_datetime) <  CURDATE()  \
;"\


query_produtos_previsao  = "select convert(prod.ean,char) as ean ,prod.description,prod.category_id, prod.unit_ean, prod.only_sell_package, cat.category as Categoria, cat.section  from clubbi.product prod left join clubbi.category cat on cat.id = prod.category_id ;"
 
 
@st.cache_resource( ttl = 45000) 
def load_produtos_previsao():
    mydb = load_my_sql()  
    query_produtos  = pd.read_sql(query_produtos_previsao,mydb)  
    query_produtos['ean'] = query_produtos['ean'].astype(np.int64).astype(str)
    return query_produtos

df_produtos_previsao = load_produtos_previsao()
 
df_produtos_previsao = df_produtos_previsao.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']]  

@st.cache_resource( ttl = 45000) 
def load_orders_previsao():
    mydb = load_my_sql() 
    query_orders = pd.read_sql(query_order_previsao,mydb)  
        
    df_produtos = load_produtos_previsao()
 
    df_inicial = query_orders.copy()

    df_inicial['Quantity'] = df_inicial['Quantity'].replace(np.nan,0)
    df_inicial = df_inicial[df_inicial['Quantity'] > 0]

    df_inicial['ean'] = df_inicial['ean'].astype(np.int64).astype(str) 
    df_inicial['unit_ean'] = df_inicial['unit_ean'].astype(np.int64).astype(str) 
    

    df_inicial = df_inicial.drop(columns = ['Categoria'])

    
    df_produtos = df_produtos.copy()
    df_produtos['ean'] = df_produtos['ean'].astype(np.int64).astype(str)
    df_produtos = df_produtos.rename(columns={'ean':'unit_ean_prod','description':'Unit_Description'})[['unit_ean_prod','Unit_Description','Categoria']]  



    df_inicial = df_inicial.merge(df_produtos  ,how ='left', left_on='unit_ean', right_on='unit_ean_prod', suffixes=(False, False))
    df_inicial['Categoria'] =   np.where((df_inicial['Categoria'] == 'Óleos, Azeites e Vinagres') ,  'Óleos, Azeites E Vinagres'  , df_inicial['Categoria'] )
    df_inicial['price_managers'] = df_inicial['price_managers'].replace(np.nan, 0 )
    df_inicial['offer_id'] = df_inicial['offer_id'].replace(np.nan, 0 ).astype(np.int64).astype(float)
 
    return df_inicial

df_order_previsao = load_orders_previsao()

# %% Trafego Previsão 

 
query_trafego_prev =  '''select * from public.trafego_site_hours '''
  
@st.cache_resource( ttl = 45000) 
def load_trafego_previsao():
    cursor = load_redshift()
    cursor.execute(query_trafego_prev)
    query_trafego_previsao: pd.DataFrame = cursor.fetch_dataframe()  

      
    df_trafego = query_trafego_previsao.copy()
    df_trafego.columns=[ str(df_trafego.columns[k-1]).title()  for k in range(1, df_trafego.shape[1] + 1)]
    df_trafego = df_trafego[['Datas', 'Datetimes', 'Hora', 'User_Id', 'Chave_Cliente_Dia', 'Chave_Final','Acessos' , 'Trafego', 'Search_Products', 'Add_To_Cart','Checkout']]
    
    

    df_trafego = df_trafego.rename(columns = {'User_Id':'User','Datas':'Data', 'Search_Products':'Trafego_Search_Products', 'Add_To_Cart':'Trafego_Add_To_Cart', 'Checkout':'Trafego_Checkout' })
    
    df_trafego = df_trafego.drop(columns = ['Datetimes','Hora','Chave_Final','Chave_Cliente_Dia'])
    df_trafego['Data'] = pd.to_datetime(df_trafego['Data'])
    df_trafego = df_trafego.groupby(['Data','User']).sum().reset_index(drop = False) 
    

    df_trafego['key'] = df_trafego['Data'].astype(str) + df_trafego['User'].astype(str)
    
    df_trafego['Trafego'] = np.where((df_trafego['Trafego'] > 0) ,  1  , 0 )
    df_trafego['Trafego_Search_Products'] = np.where((df_trafego['Trafego_Search_Products'] > 0) ,  1  , 0 )
    df_trafego['Trafego_Add_To_Cart'] = np.where((df_trafego['Trafego_Add_To_Cart'] > 0) ,  1  , 0 )
    df_trafego['Trafego_Checkout'] = np.where((df_trafego['Trafego_Checkout'] > 0) ,  1  , 0 )
    df_trafego = df_trafego.drop(columns = ['Acessos'])
     
    return df_trafego
 
 
df_trafego_previsao = load_trafego_previsao()


 
 
 
