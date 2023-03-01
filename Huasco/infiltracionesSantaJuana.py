# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:15:12 2023

@author: ccalvo
"""

import pandas as pd
import os
import numpy as np

def toFloat(df):
    return df.applymap(lambda x: x.replace(',',
'.')).apply(lambda x: x.str.strip()).replace('',np.nan).replace(',','.').astype(float)

def ravel(df):
    df=df[df.index.notnull()]
    idx=pd.date_range('1997-09-01','2013-09-01',freq='MS')
    dfOut=pd.DataFrame(index=idx,columns=['value'])
    
    dictMon={'Ene':1,'Feb':2,'Mar':3,'Abr':4,'May':5,'Jun':6,'Jul':7,'Ago':8,
             'Sep':9,'Oct':10,'Nov':11,'Dic':12}
    
    for idx in df.index:
        for col in df.columns:
            
            mon=str(dictMon[col.strip()])
            yr=str(int(idx))
            date=pd.to_datetime(yr+'-'+mon+'-01')
            dfOut.loc[date,'value']=df.loc[idx,col]
    
    dfOut.dropna(inplace=True)
    
    return dfOut

def linearReg(dfX,dfY):
    
    idxCommon=dfX.index.intersection(dfY.index)
    dfX=dfX.loc[idxCommon]
    dfY=dfY.loc[idxCommon]

    # import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    
    # create data points
    x = dfX['value'].astype(float).values
    y = dfY['value'].astype(float).values
    
    # calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # calculate the variance for x and y
    x_var = np.var(x, ddof=1) 
    y_var = np.var(y, ddof=1)
    
    # calculate the co-variance for x and y
    xy_cov = np.cov(x,y,ddof=1)[0][1] 
    
    # calculate the m coefficient (slope) 
    m = xy_cov / x_var
    
    # calculate the regression line
    reg_line = [(m*x)+0 for x in x]
    
    # plot the results
    plt.scatter(x,y, color='gray')
    plt.plot(x, reg_line)
    plt.show()
    return m

def main():
    root=r'G:\pywrDemo\Huasco\data'
    infSJ=pd.read_csv(os.path.join(root,'filtracionesSantaJuana.csv'),
                      index_col=0)
    infSJ=infSJ.astype(str).applymap(lambda x: x.strip()).replace('',np.nan)
    volSJ=pd.read_csv(os.path.join(root,'volumenSantaJuana.csv'),
                      index_col=0,encoding='latin1')
    volSJ=toFloat(volSJ)

    dfInf=ravel(infSJ)
    dfVol=ravel(volSJ)
    
    m=linearReg(dfVol*1e6,dfInf*86400)

