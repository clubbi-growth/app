# %%  Imports

# import json
# from math import sqrt 
# import pandas as pd
# import numpy as np  
# from sklearn.linear_model import Lasso  
# from feature_engine.creation import CyclicalFeatures 
# from feature_engine.datetime import DatetimeFeatures
# from feature_engine.imputation import DropMissingData
# from feature_engine.selection import DropFeatures  
# from feature_engine.timeseries.forecasting import (LagFeatures,WindowFeatures,)
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
# from statsmodels.tools.eval_measures import rmse
# import prophet
# from prophet.plot import plot_plotly, plot_components_plotly 
# import matplotlib.pyplot as plt 
# import seaborn as sns
# from statsmodels.tsa.seasonal import STL
# from scipy.interpolate import interp1d

# Imports Streamlit 

import streamlit as st
import altair as alt
import pandas as pd

import mysql.connector as connection

#import plotly.express as px 

# import redshift_connector

# conn = redshift_connector.connect(
#     host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
#     database='dev',
#     user='pbi_user',
#     password='4cL6z0E7wiBpAjNRlqKkFiLW'
# )
# cursor: redshift_connector.Cursor = conn.cursor()
# query =  '''select * from public.trafego_site_hours where datas>='2024-03-15' '''
# cursor.execute(query)
# df_trafego_query: pd.DataFrame = cursor.fetch_dataframe() 

query_order  = "select \
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
and DATE(ord.order_datetime) >= '2024-03-01' \
and DATE(ord.order_datetime) < '2025-03-01'  \
;"\


query_produtos  = "select convert(prod.ean,char) as ean ,prod.description,prod.category_id, prod.unit_ean, prod.only_sell_package, cat.category as Categoria, cat.section  from clubbi.product prod left join clubbi.category cat on cat.id = prod.category_id ;"


mydb =  connection.connect(
    host="aurora-mysql-db.cluster-ro-cjcocankcwqi.us-east-1.rds.amazonaws.com",
    user="ops-excellence-ro",
    password="L5!jj@Jm#9J+9K"
)

query_produtos = pd.read_sql(query_produtos,mydb) 
query_orders = pd.read_sql(query_order,mydb) 

mydb.close() #close the connection

# %%

st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
 
#df = pd.read_excel('C:/Users/leona/Área de Trabalho/Prophet/Clientes/Clientes.xlsx')
#df['Ano'] = df['Data'].dt.year 
 

#df['Data'] = df['DataHour'].dt.date     

# %%


with st.sidebar:
    st.title('🏂 US Population Dashboard')
    
    year_list = [2021,2022,2023]
    #list(df.Ano.unique())[::-1]
    
    selected_year = st.selectbox('Select a year', year_list)
  #  df_selected_year = df[df.Ano == selected_year]
  #  df_selected_year_sorted = df_selected_year.sort_values(by="Ano", ascending=False)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


with st.container():
    
   # query_produtos
    query_orders 

# col = st.columns((1.5, 4.5, 2), gap='medium')
# with col[0]:
#     st.markdown('#### Gains/Losses')
 
#     st.metric(label='last_state_name', value='last_state_population', delta='last_state_delta')

    
#     st.markdown('#### States Migration') 

#     migrations_col = st.columns((0.2, 1, 0.2))
#     with migrations_col[1]:
#         st.write('Inbound') 
#         st.write('Outbound') 

# with col[1]:
#     st.markdown('#### Total Population')
#     df_trafego_query

# with col[2]:
#     st.markdown('#### Top States')
 
    
#     with st.expander('About', expanded=True):
#         st.write('''
#             - Data: [U.S. Census Bureau](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html).
#             - :orange[**Gains/Losses**]: states with high inbound/ outbound migration for selected year
#             - :orange[**States Migration**]: percentage of states with annual inbound/ outbound migration > 50,000
#             ''')
