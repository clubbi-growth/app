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
#import plotly.express as px 

import redshift_connector

conn = redshift_connector.connect(
    host='redshift-analytics-cluster-1.c8ccslr41yjs.us-east-1.redshift.amazonaws.com',
    database='dev',
    user='pbi_user',
    password='4cL6z0E7wiBpAjNRlqKkFiLW'
)
cursor: redshift_connector.Cursor = conn.cursor()
query =  '''select * from public.trafego_site_hours where datas>='2024-03-15' '''
cursor.execute(query)
df_trafego_query: pd.DataFrame = cursor.fetch_dataframe() 



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
    df_trafego_query.head(100)

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
