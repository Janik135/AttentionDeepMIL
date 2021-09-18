import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns

path = "/Users/janik/Downloads/"

with open(os.path.join(path, 'cnn_bad_split_ok.pkl'), 'rb') as f:
    list_ok = pickle.load(f)

with open(os.path.join(path, 'cnn_bad_split_inoc.pkl'), 'rb') as f:
    list_inoc = pickle.load(f)

attr_df_ok = pd.DataFrame(list_ok)
attr_df_inoc = pd.DataFrame(list_inoc)

waves = wavelength[20:-1]

sns.set_style('dark')
sns.set_theme()

# Line plots
attr_df_ok_mean = attr_df_ok.mean()
sns.lineplot(x = waves, y=attr_df_ok_mean)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Mean attribution')
#plt.savefig('cnn_attr_bad_split_ok.pdf')
plt.show()

attr_df_inoc_mean = attr_df_inoc.mean()
sns.lineplot(x=waves, y=attr_df_inoc_mean)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Mean attribution')
#plt.savefig('cnn_attr_bad_split_inoc.pdf')
plt.show()

# Bar plots
fig, axes = plt.subplots(1, 2, figsize=(10,5))
attr_top_inoc = attr_df_inoc_mean.sort_values()[::-1][:10]
labels = [waves[i] for i in attr_top_inoc.index]
sns.barplot(x=labels, y=attr_top_inoc, color='seagreen', ax=axes[0])# order=labels)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
axes[0].set_xlabel('Wavelength [nm]')
axes[0].set_ylabel('Mean attribution')
axes[0].set_title('inoculated')
#plt.tight_layout()
#plt.savefig('cnn_attr_bad_split_ok_bar.pdf')
#plt.show()

attr_top_ok = attr_df_ok_mean.sort_values()[::-1][:10]
labels = [waves[i] for i in attr_top_ok.index]
sns.barplot(x=labels, y=attr_top_ok, color='seagreen', ax=axes[1])# order=labels)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
axes[1].set_xlabel('Wavelength [nm]')
axes[1].set_ylabel('Mean attribution')
axes[1].set_title('control')
plt.tight_layout()
plt.savefig('cnn_attr_bad_split_bar.pdf')
plt.show()

## MIL Attention weights

all = np.load(os.path.join(path, 'cnn_mil_attention_weights.npy'), allow_pickle=True)
