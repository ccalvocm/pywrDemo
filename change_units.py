# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:04:04 2022

@author: Carlos
"""

import pandas as pd
import os

path = r'D:\GitHub\pywrDemo\data'

ts = pd.read_csv(os.path.join(path,'Nodo4.csv'),index_col = 0, parse_dates = True)
ts['Flow'] = ts['Flow'].divide(ts.index.daysinmonth)*1e3/86400
ts.to_csv(os.path.join(path,'Nodo4_m3s.csv'))