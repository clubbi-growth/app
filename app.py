# %%  Imports

 

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
  

# %% Data Atual
# Data Atual 
print("Dataa Atual")
data_atual = datetime.date.today()
hora_atual = datetime.datetime.now()
 
data_formatada = data_atual.strftime('%d/%m/%Y')
hora_formatada = hora_atual.strftime('%H:%M:%S')
print(f"Data: {data_formatada}")
print(f"Hora: {hora_formatada}")
 

# %% 



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



# %% Query Produtos 
# Query Produtos
print("Query Produtos - df_produtos")

query_produtos  = "select convert(prod.ean,char) as ean ,prod.description,prod.category_id, prod.unit_ean, prod.only_sell_package, cat.category as Categoria, cat.section  from clubbi.product prod left join clubbi.category cat on cat.id = prod.category_id ;"


@st.cache_data()
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

 
@st.cache_data()
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

 
@st.cache_data()
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


@st.cache_resource()
def load_trafego_d_1():
    cursor = load_redshift()
    cursor.execute(query_trafego)
    data: pd.DataFrame = cursor.fetch_dataframe() 
    return data


@st.cache_data( ttl = 600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
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


@st.cache_resource()
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



@st.cache_data( ttl = 600) # ttl = 60 segundos   
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
    if ean_lista[0] != 'ean': df_trafego = df_trafego[df_trafego['unit_ean_prod'] == ean_lista[0]] 

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
 
   
 

# top_skus_atual = df_orders[df_orders['Categoria'] == 'Leite' ][df_orders['unit_ean_prod'].isin(top_skus)]['unit_ean_prod'].unique().tolist()
 
# #df_prod = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],[top_skus_atual[k]]) 
# df_prod = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],['7898403782387']) 
 

# df_prod[['Gmv','Gmv Acum']].tail(20) 
# df_RJ = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Chocolates'],['ean'])  
# df_RJ = df_RJ.reset_index(drop = False)
# hora_teste =  [22]
# df_RJ['Hora'] = df_RJ['DateHour'].dt.hour
# df_RJ[df_RJ['Hora'].isin(hora_teste)]
 
 

# %%  Load View Geral
print('Load View Geral')

@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos  
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
 
min_date = df_view['Date'].min()
max_date = df_view['Date'].max() 

# %% Load Region
print('Load Region')
@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
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

@st.cache_resource( ttl = 1600) # ttl = 30 Minutos = 60 segundos x 30 = 1800 segundos   
def df_categorias():
# Load DataFrame 1
    df_RJ = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],['ean'])  
    df_RJ_1_4 = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['1-4 Cxs'],['categoria'],['ean'])  
    df_RJ_5_9 = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],['5-9 Cxs'],['categoria'],['ean'])  
    df_BAC = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['BAC'],size_list,['categoria'],['ean'])  
    df_Leite = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Leite'],['ean']) 
    df_Acucar = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Açúcar e Adoçante'],['ean']) 
    df_Biscoitos = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Biscoitos'],['ean']) 
    df_Arroz_Feijao = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Arroz e Feijão'],['ean']) 
    df_Derivados_Leite = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Derivados de Leite'],['ean']) 
    df_Oleos = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Óleos, Azeites e Vinagres'],['ean']) 
    df_Cafe = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Cafés, Chás e Achocolatados'],['ean']) 
    df_Massas = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Massas Secas'],['ean']) 
    df_Cervejas = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Cervejas'],['ean']) 
    df_Sucos = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Sucos E Refrescos'],['ean']) 
    df_Refrigerantes = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Refrigerantes'],['ean']) 
    df_Limpeza = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Limpeza de Roupa'],['ean']) 
    df_Chocolates = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Chocolates'],['ean']) 
    df_Graos = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['Grãos e Farináceos'],['ean']) 
 
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
 
 
# df_user_teste = df_users.copy() #[['client_site_code','region_id','region name','size_final']]
# df_orders_teste = df_orders.drop(columns = ['region_id','Região']) 
# df_teste =  df_orders_teste.merge( df_user_teste ,how ='left', left_on='customer_id', right_on='client_site_code', suffixes=(False, False))
  
 

# df_Leite = cached_data["df_Leite"]
# df_Acucar = cached_data["df_Acucar"]
# df_Biscoitos = cached_data["df_Biscoitos"]
# df_Arroz_Feijao = cached_data["df_Arroz_Feijao"]
# df_Derivados_Leite = cached_data["df_Derivados_Leite"]
# df_Oleos = cached_data["df_Oleos"]
# df_Cafe = cached_data["df_Cafe"]
# df_Massas = cached_data["df_Massas"]
# df_Cervejas = cached_data["df_Cervejas"]
# df_Sucos = cached_data["df_Sucos"]
# df_Refrigerantes = cached_data["df_Refrigerantes"]
# df_Limpeza = cached_data["df_Limpeza"]
# df_Chocolates = cached_data["df_Chocolates"]
# df_Graos = cached_data["df_Graos"]  
 
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
    if len(weekday_list)== 0: weekday_list=['Weekday']
    if len(hora_list)== 0: hora_list=['Hora']
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
        

data_min = date_range[0] 
data_max = date_range[1] + pd.offsets.Day(1)
            


 #   df_view = df_view.reset_index(drop = False)
 ##   df_view = df_view[df_view['Date'] >=   date_range[0]]
   # df_view = df_view[df_view['Date'] <=  date_range[1] ]
   # df_view =  df_view.set_index('DateHour')
 
 
 
tab0 , tab1, tab2, tab3, tab4 = st.tabs(["Categoria","Trafego", "Forecast Gmv D0", "Forecast Peso D0", "Forecast Gmv D+1"])
df_count = 0 

button_count = 0   
        
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
                                
                                            
                            df_prod = cria_df_view_categoria(df_datetime,df_users, df_trafego_produtos, df_orders,pd.Timestamp('2024-01-01'),pd.Timestamp(date.today()),weekday_list,hora_list,region_list,  ['RJC'],size_list,['categoria'],[top_skus_atual[k]]) 
                                

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
    df_categoria_historico = df_categorias_final[['Categoria','Gmv Acum','Gmv Acum Lag Mean 7/14/21/28','% Var Gmv Acum','Positivação Categoria','Positivação Geral','search_products Acum','add_to_cart Acum']]
    df_categoria_historico = df_categoria_historico.sort_values( by = ['DateHour','Gmv Acum'] , ascending=[False,False])
    df_categoria_historico




 
with tab1:

    st.markdown('###### Atualizado em: ' + str(hora_atualizacao_trafego) + ' / Hora Filtrada: ' + str(max_hora_trafego))  
    

    st.header("Variação Trafego")
#    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

    with st.container(): 

       # col = st.columns((9.5,  2), gap='small')

        # with col[0]:
            
                    
        #   st.markdown('#### Variação Tráfego')

        df_view = df_view.sort_values('DateHour',ascending= False)
                    
        df_view = df_view[['Date','Hora','trafego Acum', 'Orders Acum','% Conversão Acum', '% Var trafego Acum', '% Var % Conversão Acum' ,'Gmv Acum','Forecast Gmv', '% Var Gmv Acum' , 'Peso Acum',  'Forecast Peso', '% Var Peso Acum']]
        
        

            
        delta_columns = ['% Var trafego Acum','% Var % Conversão Acum'] #,'% Var Gmv Acum' ,   '% Var Peso Acum']


    
    #   for i in delta_columns:

#             df_view[i] = df_view[i].apply(lambda x: f"{x:.2%}")
    #      df_view[i] = df_view[i].apply(lambda x: '{:.2%}'.format(x))
        #  df_view[i] = df_view[i].apply(lambda x: float(x.strip('%'))  ) 
        #  df_view[i] = df_view[i].apply(lambda x: x[4:]  ) 
        
#           styled_df = df_view[df_view['Date'] == data_max].style.apply(highlight_negative, subset=delta_columns)
#           styled_df

        df_style = df_view.copy()
        df_style = df_style[['Gmv Acum','trafego Acum','Orders Acum','% Conversão Acum','% Var trafego Acum', '% Var % Conversão Acum']]


#            df_style  = df_style[df_style['Date'] == data_max] 
        

        def try_convert(value):
    
            try:
                value = value.rstrip('%')
                value = float(value) / 100
                return value
            except ValueError:
                return None

        def highlight_negative(s):
            return ['color: #CD4C46 ' if try_convert(v) < 0 else 'color: #7900F1' for v in s]
                
            # Remove the trailing '%' sign (if present)
#               percent_str = percent_str

            # Convert the string to a float and divide by 100
#                return float(percent_str) / 100
#                return 


        df_styled = df_style.style\
                    .apply(highlight_negative, subset=delta_columns)

        df_styled







        # with col[1]:

                
        #     st.markdown('#### Takeaways')
        # #     df_trafego_query
        #     with st.expander('About', expanded=True):
        #         st.write('''
        #             - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
        #             - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
        #             - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
        #             ''')

    col = st.columns((2,  2), gap='medium')

    with col[0]:
            
                    
        st.markdown('#### Trafego Acum') 

        df_plot =  df_view.copy()

        df_plot = df_plot.reset_index(drop = False)
        df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
        df_plot = df_plot.set_index('DateHour') 
        

        if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
        if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]
 
 
        dados_x =  df_plot.index
        dados_y =  df_plot['trafego Acum']
        dados_y2 =  df_plot['% Conversão Acum'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Data", y="Trafego") , height=300, width= 600, markers = True,    line_shape='spline')

        fig 

    with col[1]:


                    
        st.markdown('#### Conversão') 
        fig=py.line(x=dados_x, y=dados_y2, height=300, width= 600, markers = True,    line_shape='spline')
        fig
        #  labels=dict(x="Data", y="% Conversão") ,


    col = st.columns((2,  2), gap='medium')

    with col[0]:
            
                    
        st.markdown('#### Gmv Acum') 

        df_plot =  df_view.copy()

        df_plot = df_plot.reset_index(drop = False)
        df_plot['weekday'] = df_plot['DateHour'].dt.weekday 
        df_plot = df_plot.set_index('DateHour') 
        

        if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['weekday'].isin(weekday_list)]
        if weekday_list[0] != 'Weekday': df_plot = df_plot[df_plot['Hora'].isin(hora_list)]


        dados_x =  df_plot.index
        dados_y =  df_plot['Gmv Acum']
        dados_y2 =  df_plot['Orders Acum'] 
        fig=py.line(x=dados_x, y=dados_y,  labels=dict(x="Data", y="Gmv") , height=300, width= 600, markers = True,    line_shape='spline')

        fig 

    with col[1]:


                    
        st.markdown('#### Pedidos Acum') 
        fig=py.line(x=dados_x, y=dados_y2,  labels=dict(x="Data", y="Pedidos") , height=300, width= 600, markers = True,    line_shape='spline')
        fig

  
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
