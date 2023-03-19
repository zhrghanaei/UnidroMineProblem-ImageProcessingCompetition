#@title Report: Q1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ft_df = pd.read_excel('from-to-rqd.xlsx')
output_df = pd.read_csv('output.csv')
ft_df['Prediction'] = output_df['Prediction']

img_list = ['M3-BH3299', 'M3-BH3300', 'M3-BH3301']
max_depth = ft_df['to'].max()

fig, ax = plt.subplots(3, figsize = (17, 15))

for i, img_name in enumerate(img_list):
  img_df = ft_df[ft_df['RunId'].str.startswith(img_name)].sort_values(by=['from'])
  img_df.reset_index(inplace = True)
  y = np.zeros(int(max_depth*100))
  yp = np.zeros(int(max_depth*100))
  for j in range(len(img_df)):
    y[int(img_df.loc[j]['from']*100): int(img_df.loc[j]['to']*100)] = img_df.loc[j]['Prediction']

  final_point = int(img_df.loc[j]['to']*100)
  ax[i].scatter([k for k in range(len(y[:final_point]))], y[:final_point], c = y[:final_point], cmap = plt.cm.plasma)

  ax[i].text(0.5, 0.9, img_name, horizontalalignment = 'center', transform = ax[i].transAxes, fontsize = 15)
  ax[i].set_xlim([0, max_depth*100])
  ax[i].set_xticks([i*100 for i in range(0,int(max_depth+5),5)])
  ax[i].set_xticklabels([i for i in range(0,int(max_depth+5),5)])
  ax[i].set_xlabel('Depth(m)')
  ax[i].set_ylim([0, 6])
  ax[i].set_yticks([1,2,3,4,5])
  ax[i].set_ylabel('RQD')
  ax[i].yaxis.grid(True)

  markers = [plt.Line2D([0,0],[0,0],color=color, marker='s', linestyle='') for color in reversed([plt.cm.plasma(k / 5) for k in range(1,6)])]
  ax[i].legend(markers, reversed([k for k in range(1,6)]), numpoints=3, loc ='upper left')

plt.show()

#@title Report: Q2
from scipy.stats import pearsonr

ft_df = pd.read_excel('from-to-rqd.xlsx')
output_df = pd.read_csv('output.csv')
ft_df['Prediction'] = output_df['Prediction']
ft_df['Percent'] = output_df['Percent']

ft_df['average depth'] = ft_df.apply(lambda row : (row['to'] + row['from'])/2, axis = 1)
max_depth = ft_df['to'].max()

prediction = ft_df['Percent'].to_numpy()
prediction[prediction > 100] = 100
prediction = prediction / np.max(prediction)

avg_depth = ft_df['average depth'].to_numpy()
avg_depth = avg_depth / np.max(avg_depth)

corr, p_value = pearsonr(avg_depth, prediction)
print('correlation:', corr)
print('p-value:', p_value)

fig, ax = plt.subplots(1, figsize = (15, 4))
ax.scatter(avg_depth, prediction)

ax.set_xlabel('Average Depth')
ax.set_ylabel('RQD %')
ax.set_xticks([i/max_depth for i in range(0,int(max_depth), 10)])
ax.set_xticklabels([i for i in range(0,int(max_depth), 10)])
ax.set_yticks([i/100 for i in range(0,int(np.max(prediction))*100+1, 10)])
ax.set_yticklabels([i for i in range(0,int(np.max(prediction))*100+1, 10)])

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
