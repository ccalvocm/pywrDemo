# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:40:55 2023

@author: ccalvo
"""

import pandas as pd
import os

def fixDf(df,col):
    idx=pd.date_range('1994-04-01','2019-03-01',freq='MS')
    df.index=idx
    return pd.DataFrame(df[col])

def completeDf(df):
    dates=pd.date_range('1994-04-01','2022-03-01',freq='MS')
    dfOut=pd.DataFrame(index=dates,columns=list(df.columns))
    dfOut[:]=0
    dfOut.loc[df.index,df.columns]=df.values
    dfOut.columns=['Q(m3/s)']
    dfOut.index.name='Date'
    return dfOut
        
def main():
    # carpetas de trabajo
    root=r'G:\pywrDemo\Huasco'
    os.chdir(root)
    
    dict_file={'QuebCamInflow.csv':['QuebCam','QEntradaQuemCam.csv'],
               'DdaCanalZR4.csv':['Transmission Link from Withdrawal Node 17 to CanalZR4_1',
                              'QSalidaCanalZR4.csv']}
    
    # proceso
    for file in list(dict_file.keys()):
        df=pd.read_csv(os.path.join('.','data',file),index_col=0,
                       header='infer')
        df=fixDf(df,col=dict_file[file][0])
        dfFull=completeDf(df)
        dfFull.to_csv(os.path.join('data',dict_file[file][1]))

if __name__=='__main__':
    main()