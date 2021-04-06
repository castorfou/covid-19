#!/usr/bin/env python
# coding: utf-8

# https://www.data.gouv.fr/fr/datasets/chiffres-cles-concernant-lepidemie-de-covid19-en-france/

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # load_timeseries

# In[2]:


#hide
import requests
import io
import os
os.environ['NO_PROXY'] = 'raw.githubusercontent.com'

def load_timeseries( 
                    base_url='https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/'):
    # Thanks to kasparthommen for the suggestion to directly downloadCSSEGISandData
    url = f'{base_url}/chiffres-cles.csv'
    print(url)
    csv = requests.get(url).text
    df = pd.read_csv(io.StringIO(csv))
    return df


# In[3]:


df_france = load_timeseries()


# # Recuperation du fichier brut

# In[4]:


df_france


# # Region

# In[5]:


df_region = df_france[df_france.granularite == 'region']


# ## Nettoyage region

# In[6]:


df_region_clean = df_region.drop(['granularite', 'maille_code', 'source_nom', 'source_url', 'source_type'], axis=1)

df_region_clean['date']=pd.to_datetime(df_region_clean['date'])
df_region_clean.info()
df_region_clean


# In[7]:


df_region_clean=df_region_clean[['maille_nom', 'date', 'cas_confirmes', 'deces', 'reanimation']]


# ## Enlever les - dans les noms de regions pour eviter les doublons (meme region mais 2 noms differents)

# In[8]:


df_region_clean['maille_nom']=df_region_clean.maille_nom.str.replace('-', ' ', regex=False)


# ## Passage en colonne, index sont les dates

# In[9]:


df_region_clean.drop_duplicates(subset=['date', 'maille_nom'], keep='last', inplace=True)
df_region_clean_indexed = df_region_clean.set_index(['date', 'maille_nom'])


# In[10]:


df_region_colonnes = df_region_clean_indexed.unstack()
df_region_colonnes


# In[11]:


df_region_colonnes.columns


# In[ ]:




