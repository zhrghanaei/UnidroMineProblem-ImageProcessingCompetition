#@title New Job

import os
import pandas as pd

file_list = os.listdir('train')

### Q1 ###
img_num = len(file_list)

### Q2 ###
boreholes = set()
for name in file_list:
  boreholes.add(name.split('-')[1])
boreholes_num = len(boreholes)

### Q3 ###
from_to = pd.read_excel('from-to-rqd.xlsx')
from_to['length'] = from_to.apply(lambda row : row['to'] - row['from'], axis = 1)
max_length = int(from_to['length'].max())

### Q4 ###
max_to = '-'.join(from_to.loc[from_to['to'].idxmax()]['RunId'].split('-')[:2])

with open('output.txt', 'w') as f:
  f.write(str(img_num) + '\n')
  f.write(str(boreholes_num) + '\n')
  f.write(str(max_length) + '\n')
  f.write(max_to)
