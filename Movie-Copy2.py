#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import libraries 
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjust the configuration of the plots we will create

#Read in the data 

df = pd.read_csv ('movies.csv')


# In[3]:


df.head ()


# In[4]:


# Let's see if there is any missing data
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[22]:


# Data types for our columns

df.dtypes


# In[23]:


#change data type of columns

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')


# In[24]:


df


# In[25]:


#create a new correct year column

df['yearcorrect'] = df ['released'].astype(str).str[8:]
df


# In[3]:


df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[17]:


pd.set_option('display.max_rows', None)


# In[16]:


#drop any duplicates
df.drop_duplicates()


# In[35]:


df


# In[36]:


# Budget high correlation 
# Company high correlation


# In[2]:


#build a scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel ('Gross Earnings')
plt.ylabel ('Budget for Film')
plt.show()


# In[40]:


df.head()


# In[43]:


# Plot budget vs gross using seaborn
sns.regplot (x='budget', y='gross', data = df, scatter_kws={"color": "red"}, line_kws={"color":"blue"})


# In[46]:


df.corr(method="pearson")


# In[47]:


#High correlation between budget and gross
#I was right


# In[26]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot= True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel ('Movie Features')
plt.ylabel ('Movie Features')
plt.show()


# In[27]:


#Look at company 
df.head()


# In[6]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype =='object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
df_numerized        


# In[7]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot = True)
plt.title ('Correlation Matrix for Numeric Features')
plt.xlabel ('Movie Features')
plt.ylabel ('Movie Features')
plt.show()


# In[9]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs


# In[11]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[12]:


high_corr = sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:


#Votes and budget have the highest correlation to gross earnings
#

