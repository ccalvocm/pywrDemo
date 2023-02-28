# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 12:38:15 2022

@author: ccalvo
"""

import os
import flopy
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#%%
class model(object):
    def __init__(self,pathNam,name,startDate):
        self.path=pathNam
        self.name=name
        self.model=None
        self.deltaX=280000.0
        self.deltaY=6820000.0
        self.startDate=startDate
    def load(self):
        self.model=flopy.modflow.Modflow.load(self.path,version="mfnwt",
 exe_name="MODFLOW-NWT.exe")
    def check(self):
        print(self.model.check())
    def run(self):
        # self.model.run()
        self.model.write_input()
        success, mfoutput = self.model.run_model('MODFLOW-NWT.exe')
        print(success)
    def copy(self):
        import copy
        return copy.deepcopy(self)

def makeOC(mf):
    """
    

    Parameters
    ----------
    mf : modflow model
        DESCRIPTION.

    Returns
    -------
    None.

    """
    sps=mf.dis.nper
    spd={(i,0) : ['SAVE HEAD'] for i in range(0,sps)}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd,
                             save_every=True, compact=True,unit_number=39)
    return None

def getDeltas():
    pathLinkage=os.path.join('..','geodata','linkageHuasco.shp')
    pathDIS=os.path.join('..','geodata','DISHuasco.shp')
    
    gdfLink=gpd.read_file(pathLinkage)
    gdfDIS=gpd.read_file(pathDIS).dissolve()
    
    deltaX=gdfLink.bounds['minx']-gdfDIS.bounds['minx']
    deltaY=gdfLink.bounds['miny']-gdfDIS.bounds['miny']

def makeDIS(modelo):
    """
    
    '1994-04-01','2019-03-01'
    Parameters
    ----------
    mf : modflow model
        DESCRIPTION.

    Returns
    -------
    None.

    """

    mf=modelo.model
    dis=mf.dis
    nlay=dis.nlay
    nrow=dis.nrow
    ncol=dis.ncol
    delr=dis.delr
    delc=dis.delc
    top=dis.top
    botm=dis.botm
    perlen=list(dis.perlen.array)+list(pd.date_range('2019-04-01','2022-03-01',
                                                     freq='MS').days_in_month)
    nper=len(perlen)
    nstp=list(12*np.ones(len(perlen)).astype(int))
    # steady=[False if ind>0 else True for ind,x in enumerate(nstp)]
    steady=[False for ind,x in enumerate(nstp)]
    mf.start_datetime=modelo.startDate
    dis3 = flopy.modflow.ModflowDis(
    mf, nlay, nrow, ncol, delr=delr, delc=delc, top=top, botm=botm,
    nper=nper,perlen=perlen,nstp=nstp,steady=steady,unitnumber=12,
    tsmult=1.2)
    return None
    
def NWT(mf):
    return flopy.modflow.ModflowNwt(mf,headtol=0.001,fluxtol=600,
maxiterout=600,thickfact=1e-05,linmeth=2,iprnwt=1,ibotav=1,options='COMPLEX')
    
def BAS(mf):
    bas=gpd.read_file(os.path.join('..','geodata','bas6Huasco.shp'))
    bas[bas['ibound_1']==1].to_file(os.path.join('..','geodata',
                                              'bas6HuascoActive.shp'))
    
def parseDates(df):
    colDate=df.columns[df.columns.str.contains('Fecha')][0]
    df[colDate]=df[colDate].apply(lambda x: pd.to_datetime(x))
    return df,colDate

def getDate(modelo):
    """
    
    El primer stress period es permanente
    Parameters
    ----------
    modelo : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return pd.date_range(modelo.startDate,'2022-03-01',freq='MS')
    
def wellsMFP(modelo):
    pathWells=os.path.join('.','geodata','CaudalTotal_CAS123.shp')
    wellsMFP=gpd.read_file(pathWells)
    wellsMFP.drop_duplicates(inplace=True)

def years(strYear):
    yrRet='19'+strYear
    if int(yrRet)<1910:
        return str(int(yrRet)+100)
    else:
        return yrRet

def fixDate(gdf):
    gdf.loc[gdf[gdf['Fecha De R']=='-'].index,'Fecha De R']='01/01/1980'
    idx=gdf[gdf['Fecha De R'].str.contains('abr')].index
    gdf.loc[idx,'Fecha De R']=gdf.loc[idx,'Fecha De R'].apply(lambda x: '01/04/'+years(x.split('-')[-1]))
    gdf['Fecha De R']=gdf['Fecha De R'].str.replace('-','/')
    return gdf

def loadDis():
    pathDis=r'G:\pywrDemo\Huasco\geodata\DISHuasco.shp'
    gdf=gpd.read_file(pathDis)
    gdf.set_crs(epsg='32719',inplace=True)
    return gdf
    
def makeWEL(modelo):
    import geopandas as gpd
    import shapely
    # DAA subterraneos
    pathDAA=r'G:\OneDrive - ciren.cl\2022_Ficha_Atacama\03_Entregas\ICTF_agosto\DAA_Atacama_shacs_val_revH.shp'
    daa=gpd.read_file(pathDAA)
    daaSubt=gpd.GeoDataFrame(daa[daa['Naturaleza']=='Subterranea'])
    daaSubCons=daaSubt[daaSubt['Tipo Derec']!='No Consuntivo']
    daaSubConsCont=daaSubCons[((daaSubCons['Ejercicio'].str.contains('Continuo',
na=False)) | (daaSubCons['Ejercicio'].isnull()))]
       
    # trasladar el modelo
    daaSubConsCont.geometry=daaSubConsCont.geometry.apply(lambda x: shapely.affinity.translate(x, 
                                    xoff=-modelo.deltaX, yoff=-modelo.deltaY))
    # celdas activas
    modelLimit=gpd.read_file(os.path.join('..','geodata','bas6Huasco.shp'))
    limit=modelLimit[modelLimit['ibound_1']>0]
    
    # overlay con las celdas activas
    daaSubOverlay=gpd.overlay(daaSubConsCont,limit)
    
    # arreglar las fechas de resolucion
    daaSubOverlay=fixDate(daaSubOverlay)
    
    # convertir a unidades de l/s a m/d
    daaSubOverlay['Caudal Anu']=daaSubOverlay['Caudal Anu'].str.replace(',',
'.').astype(float)
    idx=daaSubOverlay[daaSubOverlay['Unidad D_1']=='Lt/min'].index
    daaSubOverlay.loc[idx,
                      'Caudal Anu']=daaSubOverlay.loc[idx,
                                                      'Caudal Anu'].values/60
    daaSubOverlay['Caudal Anu']=-86400*1e-3*daaSubOverlay['Caudal Anu']
    
    gdfDIS=loadDis()
    daaSubOverlay=gpd.sjoin(daaSubOverlay,gdfDIS)
    daaSubOverlay['COLROW']=daaSubOverlay['column_right'].astype(str)+','+daaSubOverlay['row_right'].astype(str)
    # daaSubOverlay['COLROW']=daaSubOverlay.geometry.apply(lambda u: str(int(u.x/200))+','+str(138-int(u.y/200)))
    daaSubOverlay,colDate=parseDates(daaSubOverlay)
    
    # actualizar el paquete WEL
    # crear diccionario del paquete WEL
    wel_spd=dict.fromkeys(range(modelo.model.dis.nper))
    
    useFactor=1
    # El factor de uso se obtiene al comparar los bombeos de diciembre de 2017
    # y enero de 2018. Viene del hecho que los pozos no bombean el 100% del derecho

    
    # lista años
    listDates=getDate(modelo)
    for stp in wel_spd.keys():
        date=listDates[stp]
        listSpd=[]
        # sumar los DAA antes del año del stp
        daaSubSum=daaSubOverlay.copy()
        # filtrar los daa otorgados a la fecha
        daaSubSum=daaSubSum[daaSubSum[colDate].apply(lambda x: x)<=date]
        
        daaSubSum=daaSubSum.groupby(['COLROW']).agg('sum')['Caudal Anu']*useFactor
        
        for col in range(modelo.model.dis.ncol-1):
            for row in range(modelo.model.dis.nrow-1):
                try:
                    flux=daaSubSum.loc[str(col)+','+str(row)]
                    listSpd.append([0,row,col,flux]) 
                except:
                    continue
        wel_spd[stp]=[x for x in listSpd if x[-1]<=0]
    wel = flopy.modflow.ModflowWel(modelo.model,stress_period_data=wel_spd,
                                   unitnumber=20)

def processBudget():
    import matplotlib.pyplot as plt
    import flopy
    # zone_file = os.path.join('.', "gv6.zones")
    # zon = read_zbarray(zone_file)
    # nlay, nrow, ncol = zon.shape    
    # zb = ZoneBudget('gv6nwt.cbc', zon)
    # dfZB=zb.get_dataframes()
    # names=list(zb.get_record_names())
    # names=[ 'TOTAL_IN','TOTAL_OUT']
    # names=['TOTAL_IN']
    
    
    # dateidx1 = dfZB.index[0][0]
    # dateidx2 = dfZB.index[-1][0]
    # zones = ['ZONE_1']
    # dfParsed=dfZB.reset_index()
    # dfZB=dfParsed.pivot_table(index='totim',columns='name',values='ZONE_1',aggfunc='last')
    # # cols=[x for x in dfZB (if 'TOTAL' not in x) | ('ZONE' not in x) | ]
    # dfZB[list(dfZB.columns[dfZB.columns.str.contains('TO_')])]=-dfZB[list(dfZB.columns[dfZB.columns.str.contains('TO_')])]
    # dfZB[cols].plot()
    
    ruta_lst=os.path.join('.','mfnwt.lst')
    mf_list =  flopy.utils.MfListBudget(ruta_lst)
    df_incremental, df_cumulative=mf_list.get_dataframes(start_datetime="1993-01-01")
    dfError=df_incremental[['IN-OUT','PERCENT_DISCREPANCY']]/86400
    dfError.columns=['Entradas-salidas','Discrepancia del balance (%)']
    fig,ax=plt.subplots(1)
    dfError.plot(ax=ax)
    ax.set_ylim([-1e-3,1e-3])
    ax.set_ylabel('Entradas-salidas ($m^3/s$)',fontsize=14)
    plt.grid()
    plt.savefig(os.path.join('.','out','cierreBalanceCopiapo.svg'),
                bbox_inches='tight')  

    cols=[x for x in df_incremental.columns if ('TOTAL_' not in x) & ('IN-OUT' not in x) & ('PERCENT' not in x)]
    df_incremental[[x for x in cols if '_OUT' in x]]=-df_incremental[[x for x in cols if '_OUT' in x]]
    df_incremental=df_incremental/86400
    df_incremental[cols].plot()
    plt.ylabel('Balance ($m^3/s$)',fontsize=14)
    plt.savefig(os.path.join('.','out','balanceCopiapo.svg'),
                bbox_inches='tight')    
    df_incremental.to_excel(os.path.join('.','out','balanceCopiapo.xlsx'))
    # incremental, cumulative = mf_list.get_budget()
    
    #Leer el balance del primer timestep y primer stress period
    data = mf_list.get_data()
    plt.figure()
    plt.bar(data['index'], data['value'])
    plt.xticks(data['index'], data['name'], rotation=45, size=6)
    plt.show()
    plt.ylabel('Balance volumétrico ($m^3$)')
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join('.','out','balancePromedioCopiapo.svg'),
                bbox_inches='tight')    
    
def processHeads(modelo):

    # import the HeadFile reader and read in the head file
    from flopy.utils import HeadFile
    from flopy.export import vtk
    import matplotlib.pyplot as plt
    import flopy.utils.binaryfile as bf
    
    mf=modelo.model
    name=modelo.path.split('\\')[-1].replace('.nam','.hds')
    head_file = os.path.join('.', name)
    hds = HeadFile(head_file)
        
    hdobj = bf.HeadFile(head_file, precision='single')
    hdobj.list_records()
    rec = hdobj.get_data(kstpkper=(0, 0))
    rec[0][rec[0]==999]=np.nan
    plt.figure()
    plt.imshow(rec[0],vmin=0,interpolation='nearest')
    
    # create the vtk object and export heads
    vtkobj = vtk.Vtk(mf)
    otfolder=os.path.join('.','out')
    vtk.export_heads(mf, hdsfile=head_file, 
                     otfolder=otfolder,kstpkper=(0,0),
                     point_scalars=True)  
    vtkobj.add_heads(hds)
    vtkobj.write(os.path.join('.','out', name.replace('.nam','.vtu')))
    
def makeRCH(modelo):

    # cargar precipitaciones
    mf=modelo.model
    pp=pd.read_csv(os.path.join('.','GWPp.csv'),index_col=0,parse_dates=True)
    dfR=pp.copy()
    dfR.columns=['R']
    dfR.loc[dfR.index,'R']=0

    # actualizar el paquete RCH
    # actualizar la tasa de recarga por celda
    # match con los srtress periods
 
    # crear diccionario del paquete RCH
    rch_spd=dict.fromkeys(range(mf.dis.nper))
    
    rchAll=mf.rch.rech.array
    
    for t in range(mf.dis.nper):
        dfR.loc[dfR.index[t],'R']=np.sum(rchAll[t][0])
        
    dfR.loc[dfR.index>'2019-03-01','R']=np.nan
    dfRPp=pd.concat([pp,dfR],axis=1)
    imp=IterativeImputer(imputation_order='ascending',random_state=0,
max_iter=50,min_value=0,max_value=dfR.loc[dfR.index.year>=2000].max().values[0],
sample_posterior=True)
    Y=imp.fit_transform(dfRPp)
    res=pd.DataFrame(Y,columns=dfRPp.columns,index=dfRPp.index)
    
    for stp in range(list(rch_spd.keys())[-1]+1):  
        fRech=1.
        rechStp=rchAll[stp][0]        
        if stp>300:
            fRech=res.loc[res.index[stp],'R']/np.sum(rechStp)
        print(stp,fRech)
        rch_spd[stp]=float(fRech)*rechStp.astype(np.float32)[:]
            
    rch=flopy.modflow.ModflowRch(mf,nrchop=3,rech=rch_spd,irch=9,)
    return rch

def main():

    # correr modelo de embalse
    os.chdir(os.path.join('..','scripts'))
   
    # correr modelo hidrológico

    pathNam=os.path.join('..','modflow','mfnwt.nam')
    os.chdir(os.path.dirname(pathNam))
    modelo=model(pathNam,'Copiapo','1994-04-01')
    modelo.load()
    
    makeDIS(modelo)
    NWT(modelo.model)
    makeOC(modelo.model)
    # makeWEL(modelo)
    makeRCH(modelo)
    
    # incoporar la recarga del modelo superficial
    
    # escribir los paquetes
    # modelo.model.write_input(['WEL','OC','DIS','NWT','RIV','BAS6',
    #                           'UPW','CHD','RCH'])
    modelo.model.write_input(['OC','DIS','NWT','RIV','BAS6',
                              'UPW','CHD','RCH'])
    # correr modelo de aguas subterráneas
    modelo.model.run_model(silent=False)
    
    # processHeads(modelo)
    # processBudget()

if __name__=='__main__':
    main()