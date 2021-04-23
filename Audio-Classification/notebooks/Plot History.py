#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


log_csvs = sorted(os.listdir('../logs'))
print(log_csvs)


# In[3]:


labels = ['Conv 1D', 'Conv 2D', 'LSTM']
colors = ['r', 'm', 'c']


# In[4]:


fig, ax = plt.subplots(1, 3, sharey=True, figsize=(16,5))

for i, (fn, label, c) in enumerate(zip(log_csvs, labels, colors)):
    csv_path = os.path.join('..', 'logs', fn)
    df = pd.read_csv(csv_path)
    ax[i].set_title(label, size=16)
    ax[i].plot(df.accuracy, color=c, label='train')
    ax[i].plot(df.val_accuracy, ls='--', color=c, label='test')
    ax[i].legend(loc='upper left')
    ax[i].tick_params(axis='both', which='major', labelsize=12)
    ax[i].set_ylim([0,1.0])

fig.text(0.5, 0.02, 'Epochs', ha='center', size=14)
fig.text(0.08, 0.5, 'Accuracy', va='center', rotation='vertical', size=14)
plt.show()


# In[ ]:




