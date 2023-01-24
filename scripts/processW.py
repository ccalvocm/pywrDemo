# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:40:55 2023

@author: ccalvo
"""

import pandas as pd
import os


def fixDf(df):
    idx = pd.date_range('1994-04-01', '2019-03-01', freq='MS')
    df.index = idx
    return pd.DataFrame(df)

def completeDf(df):
    dates = pd.date_range('1994-04-01', '2022-03-01', freq='MS')
    dfOut = pd.DataFrame(index=dates, columns=list(df.columns))
    dfOut[:] = 0
    dfOut.loc[df.index, df.columns] = df.values
    dfOut.index.name = 'Date'
    return dfOut

def afluentesStaJuana():
    """
    calcula los afluentes al Santa Juana de Rio Huasco en Algodones 03820001-1

    Returns
    -------
    None.

    """
    path=os.path.join('.','data','q_relleno_1987-2022_monthly.xlsx')
    df=pd.read_excel(path,sheet_name='Data',index_col=0,parse_dates=True)
    df.index.name='Date'
    df.to_csv(os.path.join('.','data','q_relleno_1987-2022_monthly.csv'))
    
def main():
    # carpetas de trabajo
    root = r'G:\pywrDemo\Huasco'
    os.chdir(root)

    dict_file = {'QEntradas.csv':'QEntradas.csv',
                 'Qsalidas.csv': 'QSalidasCanales.csv'}

    # proceso
    for file in list(dict_file.keys()):
        df = pd.read_csv(os.path.join('.', 'inputs', file), index_col=0,
                         header='infer')
        df = fixDf(df)
        dfFull = completeDf(df)
        dfFull.to_csv(os.path.join('data', dict_file[file]))


if __name__ == '__main__':
    main()
