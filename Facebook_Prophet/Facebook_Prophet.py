#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot


# In[2]:


df_115_1 = pd.read_csv('result/cam_115_preset_1.csv')
df_115_1 = df_115_1[['timestamp', 'avg_speed']]
df_115_1['timestamp'] = pd.to_datetime(df_115_1['timestamp'])
df_115_1.rename(columns = {'timestamp':'ds', 'avg_speed':'y'}, inplace = True)
df_115_1.head()


# In[3]:


m_115_1 = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
m_115_1.fit(df_115_1)


# In[4]:


future_115_1 = m_115_1.make_future_dataframe(periods=120, freq='5min')

future_115_1.tail()


# In[5]:


forecast_115_1 = m_115_1.predict(future_115_1)

forecast_115_1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[6]:


fig_115_1_1 = m_115_1.plot(forecast_115_1)


# In[7]:


fig_115_1_2 = m_115_1.plot_components(forecast_115_1)


# In[8]:


plot_plotly(m_115_1, forecast_115_1)


# In[9]:


plot_components_plotly(m_115_1, forecast_115_1)


# In[10]:


a_115_1 = add_changepoints_to_plot(fig_115_1_1.gca(), m_115_1, forecast_115_1)
fig_115_1_1


# In[11]:


df_65_2 = pd.read_csv('result/cam_65_preset_2.csv')
df_65_2 = df_65_2[['timestamp', 'avg_speed']]
df_65_2['timestamp'] = pd.to_datetime(df_65_2['timestamp'])
df_65_2.rename(columns = {'timestamp':'ds', 'avg_speed':'y'}, inplace = True)
df_65_2.head()


# In[12]:


m_65_2 = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
m_65_2.fit(df_65_2)


# In[13]:


future_65_2 = m_65_2.make_future_dataframe(periods=120, freq='5min')

future_65_2.tail()


# In[14]:


forecast_65_2 = m_65_2.predict(future_65_2)

forecast_65_2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[15]:


fig_65_2_1 = m_65_2.plot(forecast_65_2)


# In[16]:


fig_65_2_2 = m_65_2.plot_components(forecast_65_2)


# In[17]:


plot_plotly(m_65_2, forecast_65_2)


# In[18]:


plot_components_plotly(m_65_2, forecast_65_2)


# In[19]:


a_65_2 = add_changepoints_to_plot(fig_65_2_1.gca(), m_65_2, forecast_65_2)
fig_65_2_1

