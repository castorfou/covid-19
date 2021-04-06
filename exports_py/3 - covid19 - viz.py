#!/usr/bin/env python
# coding: utf-8

# https://github.com/UncleGedd/COVID19-EDA/blob/master/COVID-19%20Viz.ipynb

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#hide
import requests
import io
import os

os.environ['NO_PROXY'] = 'raw.githubusercontent.com'

def load_timeseries(name, 
                    base_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series'):
    # Thanks to kasparthommen for the suggestion to directly downloadCSSEGISandData
    url = f'{base_url}/time_series_19-covid-{name}.csv'
    print(url)
    csv = requests.get(url).text
    df = pd.read_csv(io.StringIO(csv))
    return df


# In[3]:


df_confirmed = load_timeseries('Confirmed')
df_deaths = load_timeseries('Deaths')
df_recovered = load_timeseries('Recovered')
df_confirmed.head()


# In[4]:


sorted(df_confirmed['Country/Region'].unique())[:5]


# In[5]:


euro = [
'Austria',
'Belarus',
'Belgium',
'Bosnia and Herzegovina',
'Bulgaria',
'Croatia',
'Cyprus',
'Czechia',
'Denmark',
'Estonia',
'Finland',
'France',
'Germany',
'Greece',
'Hungary',
'Ireland',
'Italy',
'Latvia',
'Liechtenstein',
'Lithuania',
'Luxembourg',
'Malta',
'Monaco',
'Moldova',
'Netherlands',
'North Macedonia',
'Norway',
'Poland',
'Portugal',
'Romania',
'Serbia',
'Slovakia',
'Slovenia',
'Spain',
'Sweden',
'Switzerland',
'Ukraine',
'United Kingdom'
]


# In[6]:


# sum daily data for a particular list of counties
def preprocess_sum(df, countries):
    df_country = df[df['Country/Region'].isin(countries)].drop(df.columns[0:4], axis=1)
    df_country = df_country.apply(lambda c: np.sum(c), axis=0) # returns a series
    dates, ts_data = list(map(lambda i: i[:-3], df_country.index)), list(df_country)
    return dates, ts_data


# In[7]:


# get daily counts of new events (infections, recoveries, deaths)
def preprocess_daily_new(df, countries):
    df_country = df[df['Country/Region'].isin(countries)].drop(df.columns[0:4], axis=1)
    df_country = df_country.apply(lambda c: np.sum(c), axis=0).diff() # returns a series
    dates, ts_data = list(map(lambda i: i[:-3], df_country.index[1:])), list(df_country)[1:]
    return dates, ts_data


# # China

# Time series of infections in China

# In[8]:



dates, ts_infections_china = preprocess_sum(df_confirmed, ['China'])
_, ts_deaths_china = preprocess_sum(df_deaths, ['China'])
_, ts_recovered_china = preprocess_sum(df_recovered, ['China'])


# In[9]:


plt.figure(figsize=(10,5))
plt.plot(dates, ts_infections_china, label="Infections")
plt.plot(dates, ts_deaths_china, label="Deaths")
plt.plot(dates, ts_recovered_china, label="Recovered")
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 China 2020')
plt.show()


# New infections, deaths, recovered
# 
# 

# In[10]:


dates, daily_infections_china = preprocess_daily_new(df_confirmed, ['China'])
_, daily_deaths_china = preprocess_daily_new(df_deaths, ['China'])
_, daily_recovered_china = preprocess_daily_new(df_recovered, ['China'])


# In[11]:


df_new_events_china =  pd.DataFrame([daily_infections_china, daily_deaths_china, daily_recovered_china], 
                                    columns=dates, index=['Infections', 'Deaths', 'Recovered'])
df_new_events_china.head()


# In[12]:


plt.figure(figsize=(10,5))
plt.bar(dates, daily_infections_china, label="Infections", 
        bottom=np.array(daily_deaths_china+np.array(daily_recovered_china)))
plt.bar(dates, daily_deaths_china, label="Deaths", bottom=daily_recovered_china)
plt.bar(dates, daily_recovered_china, label="Recovered")
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 China 2020 - Daily New Events')
plt.show()


# In[13]:


plt.figure(figsize=(10,5))
plt.bar(dates, daily_infections_china, label="Infections", 
        bottom=np.array(daily_deaths_china+np.array(daily_recovered_china)), log=True)
plt.bar(dates, daily_deaths_china, label="Deaths", bottom=daily_recovered_china, log=True)
plt.bar(dates, daily_recovered_china, label="Recovered", log=True)
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 China 2020 - Daily New Events (Log Scale)')
plt.show()


# # Europe

# Time series of infections in Europe
# 
# 

# In[14]:


df_euro = df_confirmed[df_confirmed['Country/Region'].isin(euro)].drop(df_confirmed.columns[0:4], axis=1)
df_euro = df_euro.apply(lambda c: np.sum(c), axis=0) # returns a series
ts_infections_euro = list(df_euro)


# In[15]:


dates, ts_infections_euro = preprocess_sum(df_confirmed, euro)
dates, ts_deaths_euro = preprocess_sum(df_deaths, euro)
dates, ts_recovered_euro = preprocess_sum(df_recovered, euro)


# In[16]:


plt.figure(figsize=(10,5))
plt.plot(dates, ts_infections_euro, label="Infections")
plt.plot(dates, ts_deaths_euro, label="Deaths")
plt.plot(dates, ts_recovered_euro, label="Recovered")
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 Europe 2020')
plt.show()


# New infections, deaths, recovered
# 
# 

# In[17]:


dates, daily_infections_euro = preprocess_daily_new(df_confirmed, euro)
_, daily_deaths_euro = preprocess_daily_new(df_deaths, euro)
_, daily_recovered_euro = preprocess_daily_new(df_recovered, euro)


# In[18]:


df_new_events_euro =  pd.DataFrame([daily_infections_euro, daily_deaths_euro, daily_recovered_euro], 
                                    columns=dates, index=['Infections', 'Deaths', 'Recovered'])
df_new_events_euro.head()


# In[19]:


plt.figure(figsize=(10,5))
SKIP=20
plt.bar(dates[SKIP:], daily_infections_euro[SKIP:], label="Infections", 
        bottom=np.array(daily_deaths_euro[SKIP:]+np.array(daily_recovered_euro[SKIP:])))
plt.bar(dates[SKIP:], daily_deaths_euro[SKIP:], label="Deaths", bottom=daily_recovered_euro[SKIP:])
plt.bar(dates[SKIP:], daily_recovered_euro[SKIP:], label="Recovered")
plt.xticks(range(0, len(dates[SKIP:]), 4))
plt.legend()
plt.title('COVID-19 Euro 2020 - Daily New Events')
plt.show()


# # France

# Time series of infections in France

# In[20]:



dates, ts_infections_france = preprocess_sum(df_confirmed, ['France'])
_, ts_deaths_france = preprocess_sum(df_deaths, ['France'])
_, ts_recovered_france = preprocess_sum(df_recovered, ['France'])


# In[21]:


plt.figure(figsize=(10,5))
plt.plot(dates, ts_infections_france, label="Infections")
plt.plot(dates, ts_deaths_france, label="Deaths")
plt.plot(dates, ts_recovered_france, label="Recovered")
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 France 2020')
plt.show()


# New infections, deaths, recovered
# 
# 

# In[22]:


dates, daily_infections_france = preprocess_daily_new(df_confirmed, ['France'])
_, daily_deaths_france = preprocess_daily_new(df_deaths, ['France'])
_, daily_recovered_france = preprocess_daily_new(df_recovered, ['France'])


# In[23]:


df_new_events_france =  pd.DataFrame([daily_infections_france, daily_deaths_france, daily_recovered_france], 
                                    columns=dates, index=['Infections', 'Deaths', 'Recovered'])
df_new_events_france.head()


# In[24]:


df_new_events_france.T.sum()


# In[25]:


plt.figure(figsize=(10,5))
plt.bar(dates, daily_infections_france, label="Infections", 
        bottom=np.array(daily_deaths_france+np.array(daily_recovered_france)))
plt.bar(dates, daily_deaths_france, label="Deaths", bottom=daily_recovered_france)
plt.bar(dates, daily_recovered_france, label="Recovered")
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 France 2020 - Daily New Events')
plt.show()


# In[26]:


plt.figure(figsize=(10,5))
plt.bar(dates, daily_infections_france, label="Infections", 
        bottom=np.array(daily_deaths_france+np.array(daily_recovered_france)), log=True)
plt.bar(dates, daily_deaths_france, label="Deaths", bottom=daily_recovered_france, log=True)
plt.bar(dates, daily_recovered_france, label="Recovered", log=True)
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 France 2020 - Daily New Events (Log Scale)')
plt.show()


# # Plotting France vs China

# In[27]:


dates,_ = preprocess_sum(df_confirmed, ['France']) # ensure proper date range

plt.figure(figsize=(10,5))
plt.plot(dates, ts_infections_france, label="France Infections")
plt.plot(dates, ts_deaths_france, label="France Deaths")

plt.plot(dates, ts_infections_china, label="China Infections")
plt.plot(dates, ts_deaths_china, label="China Deaths")
plt.xticks(range(0, len(dates), 4))

plt.legend()
plt.title('COVID-19 Infections and Deaths in France and China 2020')
plt.show()


# Closer look at death rates

# In[28]:



plt.figure(figsize=(10,5))
plt.plot(dates, ts_deaths_france, label="France Deaths")
plt.plot(dates, ts_deaths_china, label="China Deaths")
plt.xticks(range(0, len(dates), 4))
plt.legend()
plt.title('COVID-19 Deaths in France and China 2020')
plt.show()


# In[ ]:




