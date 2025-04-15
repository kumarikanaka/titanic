#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import pandas as pd

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display the first few rows
print(df.head())


# In[8]:


# Check structure and missing values
df.info()




# In[10]:


# Statistical summary of numerical columns
df.describe()


# In[12]:


sns.countplot(x='survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# In[14]:


sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()


# In[16]:


sns.histplot(df['age'].dropna(), bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[18]:


sns.boxplot(x='pclass', y='age', data=df)
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()


# In[20]:


# Select numerical columns
numeric_df = df.select_dtypes(include='number')

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[22]:


# We'll create a new DataFrame with selected columns
pairplot_data = df[['age', 'fare', 'pclass', 'survived', 'sex']]


# In[24]:


# Convert 'sex' to numeric (optional, but helps with coloring)
pairplot_data['sex'] = pairplot_data['sex'].map({'male': 0, 'female': 1})


# In[30]:


sns.pairplot(pairplot_data, hue='sex', palette='coolwarm', diag_kind='kde')
plt.suptitle("Pairplot of Titanic Features", y=1.02)
plt.show()


# In[28]:


sns.pairplot(pairplot_data, hue='survived', palette='Set1', diag_kind='kde', corner=True)


# In[ ]:




