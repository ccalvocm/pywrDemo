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

# %%


class model(object):
    def __init__(self, pathNam, name, startDate):
        self.path = pathNam
        self.name = name
        self.model = None
        self.deltaX = 280000.0
        self.deltaY = 6820000.0
        self.startDate = startDate

    def load(self):
        self.model = flopy.modflow.Modflow.load(self.path, version="mfnwt",
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
    sps = mf.dis.nper
    spd = {(i, j): ['SAVE HEAD', 'SAVE BUDGET'] for i in range(0,
                                                               sps) for j in range(mf.dis.nstp.array[0])}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd,
                                 save_every=True, compact=True, unit_number=14)
    return oc


def getDeltas():
    pathLinkage = os.path.join('..', 'geodata', 'linkageHuasco.shp')
    pathDIS = os.path.join('..', 'geodata', 'DISHuasco.shp')

    gdfLink = gpd.read_file(pathLinkage)
    gdfDIS = gpd.read_file(pathDIS).dissolve()

    deltaX = gdfLink.bounds['minx']-gdfDIS.bounds['minx']
    deltaY = gdfLink.bounds['miny']-gdfDIS.bounds['miny']


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

    mf = modelo.model
    dis = mf.dis
    nlay = dis.nlay
    nrow = dis.nrow
    ncol = dis.ncol
    delr = dis.delr
    delc = dis.delc
    top = dis.top
    botm = dis.botm
    perlen = list(dis.perlen.array)+list(pd.date_range('2019-04-01', '2022-03-01',
                                                       freq='MS').days_in_month)
    nper = len(perlen)
    nstp = list(12*np.ones(len(perlen)).astype(int))
    # steady=[False if ind>0 else True for ind,x in enumerate(nstp)]
    steady = [False for ind, x in enumerate(nstp)]
    mf.start_datetime = modelo.startDate
    dis3 = flopy.modflow.ModflowDis(
        mf, nlay, nrow, ncol, delr=delr, delc=delc, top=top, botm=botm,
        nper=nper, perlen=perlen, nstp=nstp, steady=steady, unitnumber=12,
        tsmult=1.2)
    return None


def NWT(mf):
    return flopy.modflow.ModflowNwt(mf, headtol=0.001, fluxtol=600,
                                    maxiterout=600, thickfact=1e-05, linmeth=2, iprnwt=1, ibotav=1, options='COMPLEX')


def BAS(mf):
    bas = gpd.read_file(os.path.join('..', 'geodata', 'bas6Huasco.shp'))
    bas[bas['ibound_1'] == 1].to_file(os.path.join('..', 'geodata',
                                                   'bas6HuascoActive.shp'))


def parseDates(df):
    colDate = df.columns[df.columns.str.contains('Fecha')][0]
    df[colDate] = df[colDate].apply(lambda x: pd.to_datetime(x))
    return df, colDate


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
    return pd.date_range(modelo.startDate, '2022-03-01', freq='MS')


def wellsMFP(modelo):
    pathWells = os.path.join('.', 'geodata', 'CaudalTotal_CAS123.shp')
    wellsMFP = gpd.read_file(pathWells)
    wellsMFP.drop_duplicates(inplace=True)


def years(strYear):
    yrRet = '19'+strYear
    if int(yrRet) < 1910:
        return str(int(yrRet)+100)
    else:
        return yrRet


def fixDate(gdf):
    gdf.loc[gdf[gdf['Fecha De R'] == '-'].index, 'Fecha De R'] = '01/01/1980'
    idx = gdf[gdf['Fecha De R'].str.contains('abr')].index
    gdf.loc[idx, 'Fecha De R'] = gdf.loc[idx, 'Fecha De R'].apply(
        lambda x: '01/04/'+years(x.split('-')[-1]))
    gdf['Fecha De R'] = gdf['Fecha De R'].str.replace('-', '/')
    return gdf


def loadDis():
    pathDis = r'G:\pywrDemo\Huasco\geodata\DISHuasco.shp'
    gdf = gpd.read_file(pathDis)
    gdf.set_crs(epsg='32719', inplace=True)
    return gdf


def makeWEL(modelo):
    import geopandas as gpd
    import shapely
    # DAA subterraneos
    pathDAA = r'G:\OneDrive - ciren.cl\2022_Ficha_Atacama\03_Entregas\ICTF_agosto\DAA_Atacama_shacs_val_revH.shp'
    daa = gpd.read_file(pathDAA)
    daaSubt = gpd.GeoDataFrame(daa[daa['Naturaleza'] == 'Subterranea'])
    daaSubCons = daaSubt[daaSubt['Tipo Derec'] != 'No Consuntivo']
    daaSubConsCont = daaSubCons[((daaSubCons['Ejercicio'].str.contains('Continuo',
                                                                       na=False)) | (daaSubCons['Ejercicio'].isnull()))]

    # trasladar el modelo
    daaSubConsCont.geometry = daaSubConsCont.geometry.apply(lambda x: shapely.affinity.translate(x,
                                                                                                 xoff=-modelo.deltaX, yoff=-modelo.deltaY))
    # celdas activas
    modelLimit = gpd.read_file(os.path.join('..', 'geodata', 'bas6Huasco.shp'))
    limit = modelLimit[modelLimit['ibound_1'] > 0]

    # overlay con las celdas activas
    daaSubOverlay = gpd.overlay(daaSubConsCont, limit)

    # arreglar las fechas de resolucion
    daaSubOverlay = fixDate(daaSubOverlay)

    # convertir a unidades de l/s a m/d
    daaSubOverlay['Caudal Anu'] = daaSubOverlay['Caudal Anu'].str.replace(',',
                                                                          '.').astype(float)
    idx = daaSubOverlay[daaSubOverlay['Unidad D_1'] == 'Lt/min'].index
    daaSubOverlay.loc[idx,
                      'Caudal Anu'] = daaSubOverlay.loc[idx,
                                                        'Caudal Anu'].values/60
    daaSubOverlay['Caudal Anu'] = -86400*1e-3*daaSubOverlay['Caudal Anu']

    gdfDIS = loadDis()
    daaSubOverlay = gpd.sjoin(daaSubOverlay, gdfDIS)
    daaSubOverlay['COLROW'] = daaSubOverlay['column_right'].astype(
        str)+','+daaSubOverlay['row_right'].astype(str)
    # daaSubOverlay['COLROW']=daaSubOverlay.geometry.apply(lambda u: str(int(u.x/200))+','+str(138-int(u.y/200)))
    daaSubOverlay, colDate = parseDates(daaSubOverlay)

    # actualizar el paquete WEL
    # crear diccionario del paquete WEL
    wel_spd = dict.fromkeys(range(modelo.model.dis.nper))

    useFactor = 1
    # El factor de uso se obtiene al comparar los bombeos de diciembre de 2017
    # y enero de 2018. Viene del hecho que los pozos no bombean el 100% del derecho

    # lista años
    listDates = getDate(modelo)
    for stp in wel_spd.keys():
        date = listDates[stp]
        listSpd = []
        # sumar los DAA antes del año del stp
        daaSubSum = daaSubOverlay.copy()
        # filtrar los daa otorgados a la fecha
        daaSubSum = daaSubSum[daaSubSum[colDate].apply(lambda x: x) <= date]

        daaSubSum = daaSubSum.groupby(['COLROW']).agg('sum')[
            'Caudal Anu']*useFactor

        for col in range(modelo.model.dis.ncol-1):
            for row in range(modelo.model.dis.nrow-1):
                try:
                    flux = daaSubSum.loc[str(col)+','+str(row)]
                    listSpd.append([0, row, col, flux])
                except:
                    continue
        wel_spd[stp] = [x for x in listSpd if x[-1] <= 0]
    wel = flopy.modflow.ModflowWel(modelo.model, stress_period_data=wel_spd,
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

    ruta_lst = os.path.join('..','modflow','mfnwt.lst')
    mf_list = flopy.utils.MfListBudget(ruta_lst)
    df_incremental, df_cumulative = mf_list.get_dataframes(
        start_datetime="1994-03-01")
    dfError = df_incremental[['IN-OUT', 'PERCENT_DISCREPANCY']]/86400
    dfError.columns = ['Entradas-salidas', 'Discrepancia del balance (%)']
    fig, ax = plt.subplots(1)
    dfError.plot(ax=ax)
    ax.legend(list(dfError.columns),fontsize=16)
    ax.set_ylim([-1e-4, 1e-4])
    ax.set_ylabel('Entradas-salidas ($m^3/s$)', fontsize=16)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(os.path.join('..','modflow','out', 'cierreBalanceHuasco.svg'),
                bbox_inches='tight')

    cols = [x for x in df_incremental.columns if (
        'TOTAL_' not in x) & ('IN-OUT' not in x) & ('PERCENT' not in x)]
    df_incremental[[x for x in cols if '_OUT' in x]] = - \
        df_incremental[[x for x in cols if '_OUT' in x]]
    df_incremental = df_incremental/86400
    df_incremental[cols].plot()
    plt.ylabel('Balance ($m^3/s$)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid()
    plt.savefig(os.path.join('..','modflow','out', 'balanceHuasco.svg'),
                bbox_inches='tight')
    df_incremental.to_excel(os.path.join('..','modflow','out', 'balanceHuasco.xlsx'))
    # incremental, cumulative = mf_list.get_budget()

    # Leer el balance del primer timestep y primer stress period
    data = mf_list.get_data()
    plt.figure()
    plt.bar(data['index'], data['value'])
    plt.xticks(data['index'], data['name'], rotation=45, size=6)
    plt.show()
    plt.ylabel('Balance volumétrico ($m^3$)',fontsize=14)
    plt.tight_layout()
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(os.path.join('..','modflow','out', 'balancePromedioHuasco.svg'),
                bbox_inches='tight')


def replace(df, oldNum, newNum):
    """
    equivalencias:
        2 es 03070002
        1 es 03070001
        3 es 03070000 quebrada Honda
        4 es NULL Huasco grande

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    oldNum : TYPE
        DESCRIPTION.
    newNum : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df[:]
    df['index_right'][df['index_right'] == oldNum] = newNum
    return df


def getRiverLeakage(modelo):
    import matplotlib.pyplot as plt
    from flopy.utils import ZoneBudget

    pathSubcuencas = r'G:\WEAP\WEAP_areas\Huasco_PEGRH_5.3\SIG\Subsubcuencas_Huasco_DARH.shp'
    gdfSubcuencas = gpd.read_file(pathSubcuencas)

    # translate
    gdfSubcuencasT = gdfSubcuencas.translate(-280000., -6820000.)
    gdfTranslate = gpd.GeoDataFrame(gdfSubcuencasT.index,
                                    geometry=gdfSubcuencasT.geometry)
    gdfDIS = loadDis()

    # qaqc
    fig, ax = plt.subplots(1)
    gdfTranslate.plot(ax=ax)
    gdfDIS.plot(ax=ax, color='c')

    DISscuen = gpd.sjoin(gdfDIS, gdfTranslate, how='left')
    DISscuenDR = DISscuen.drop_duplicates('node')

    DISscuenDR = replace(DISscuenDR, 1, 2)
    DISscuenDR = replace(DISscuenDR, 0, 1)
    DISscuenDR = replace(DISscuenDR, 11, 3)
    DISscuenDR = replace(DISscuenDR, 13, 4)
    DISscuenDR['index_right'][DISscuenDR['index_right'].isnull()] = 5

    x = np.random.randint(100, 200, size=(2, modelo.model.dis.nrow,
                                          modelo.model.dis.ncol))
    for i in range(1, modelo.model.dis.nrow+1):
        for j in range(1, modelo.model.dis.ncol+1):
            x[0, i-1, j-1] = DISscuenDR[(DISscuenDR['row'] == i) &
                                        (DISscuenDR['column'] == j)]['index_right'].values[0]

    x[1, :, :] = x[0, :, :]
    pathZone = os.path.join('.', 'Huasco.zone')
    ZoneBudget.write_zone_file(pathZone, x, fmtin=35, iprn=2)

    zon = ZoneBudget.read_zone_file(pathZone)
    # Create a ZoneBudget object and get the budget record array
    cbc_f = os.path.join('.', "mfnwt.cbc")
    zones = ['ZONE_1', 'ZONE_2', 'ZONE_3', 'ZONE_4']

    nPer = modelo.model.dis.nper
    dfPercola = pd.DataFrame([], index=pd.date_range('1994-04-01',
                                                     '2022-03-01', freq='MS'),
                             columns=zones)
    dfAflora = pd.DataFrame([], index=pd.date_range('1994-04-01',
                                                    '2022-03-01', freq='MS'),
                            columns=zones)
    names = ['TO_RIVER_LEAKAGE', 'FROM_RIVER_LEAKAGE']
    for sp in range(modelo.model.dis.nper):
        zb = flopy.utils.ZoneBudget(cbc_f, zon, kstpkper=(11, sp))
        df = zb.get_dataframes()
        tim = df.index[0][0]
        dfPercola.loc[dfPercola.index[sp],
                      zones] = df.loc[tim, 'FROM_RIVER_LEAKAGE']
        dfAflora.loc[dfAflora.index[sp],
                     zones] = df.loc[tim, 'TO_RIVER_LEAKAGE']

    dfPercola.multiply(1/86400.).to_csv(os.path.join('..','scripts','percolacion.csv'))
    dfAflora.multiply(1/86400.).to_csv(os.path.join('..','scripts','afloramientos.csv'))
    dfNeto=dfAflora-dfPercola
    dfNeto.to_csv(os.path.join('..','scripts','recuperaciones.csv'))
    
def processHeads(modelo):

    # import the HeadFile reader and read in the head file
    from flopy.utils import HeadFile
    from flopy.export import vtk
    import matplotlib.pyplot as plt
    import flopy.utils.binaryfile as bf

    mf = modelo.model
    name = modelo.path.split('\\')[-1].replace('.nam', '.hds')
    head_file = os.path.join('.', name)
    hds = HeadFile(head_file)

    hdobj = bf.HeadFile(head_file, precision='single')
    hdobj.list_records()
    rec = hdobj.get_data(kstpkper=(0, 0))
    rec[0][rec[0] == 999] = np.nan
    plt.figure()
    plt.imshow(rec[0], vmin=0, interpolation='nearest')

    # create the vtk object and export heads
    vtkobj = vtk.Vtk(mf)
    otfolder = os.path.join('.', 'out')
    vtk.export_heads(mf, hdsfile=head_file,
                     otfolder=otfolder, kstpkper=(0, 0),
                     point_scalars=True)
    vtkobj.add_heads(hds)
    vtkobj.write(os.path.join('.', 'out', name.replace('.nam', '.vtu')))


def makeRCH(modelo):

    # cargar precipitaciones
    mf = modelo.model
    pp = pd.read_csv(os.path.join('.', 'GWPp.csv'),
                     index_col=0, parse_dates=True)
    dfR = pp.copy()
    dfR.columns = ['R']
    dfR.loc[dfR.index, 'R'] = 0

    # actualizar el paquete RCH
    # actualizar la tasa de recarga por celda
    # match con los srtress periods

    # crear diccionario del paquete RCH
    rch_spd = dict.fromkeys(range(mf.dis.nper))

    rchAll = mf.rch.rech.array

    for t in range(mf.dis.nper):
        dfR.loc[dfR.index[t], 'R'] = np.sum(rchAll[t][0])

    dfR.loc[dfR.index > '2019-03-01', 'R'] = np.nan
    dfRPp = pd.concat([pp, dfR], axis=1)
    imp = IterativeImputer(imputation_order='ascending', random_state=0,
                           max_iter=50, min_value=0, max_value=dfR.loc[dfR.index.year >= 2000].max().values[0],
                           sample_posterior=True)
    Y = imp.fit_transform(dfRPp)
    res = pd.DataFrame(Y, columns=dfRPp.columns, index=dfRPp.index)

    for stp in range(list(rch_spd.keys())[-1]+1):
        fRech = 1.
        rechStp = rchAll[stp][0]
        if stp > 300:
            fRech = res.loc[res.index[stp], 'R']/np.sum(rechStp)
        print(stp, fRech)
        rch_spd[stp] = float(fRech)*rechStp.astype(np.float32)[:]

    rch = flopy.modflow.ModflowRch(mf, nrchop=3, rech=rch_spd, irch=9,)
    return rch


def crop(df):
    df.index.name = ''
    return df.loc[df.index < '2019-04-01']


def SW():
    import matplotlib.pyplot as plt
    """
    obtener las entradas y salidas superficiales

    Returns
    -------
    None.

    """
    pathEntrada = os.path.join('..', 'data', 'QEntradasm3day.csv')
    dfEntrada = pd.read_csv(pathEntrada,
                            index_col=0, parse_dates=True).sum(axis=1)/86400
    pathSalida = os.path.join('..', 'data', 'QSalidasCanalesm3day.csv')
    dfSalida = pd.read_csv(pathSalida, index_col=0,
                           parse_dates=True).sum(axis=1)/86400

    # calcular la demanda de aguas subterraneas
    ruta_lst = os.path.join('..', 'modflow', 'mfnwt.lst')
    mf_list = flopy.utils.MfListBudget(ruta_lst)
    df_incremental, df_cumulative = mf_list.get_dataframes(
        start_datetime="1994-03-01")

    cols = [x for x in df_incremental.columns if (
        'TOTAL_' not in x) & ('IN-OUT' not in x) & ('PERCENT' not in x)]
    df_incremental[[x for x in cols if '_OUT' in x]] = - \
        df_incremental[[x for x in cols if '_OUT' in x]]
    df_incremental = df_incremental/86400
    dfWells = df_incremental['WELLS_OUT']
    dfWells.index = dfSalida.index

    dfRiegoSup = dfSalida-dfWells
    plt.close('all')
    fig, ax = plt.subplots(1)
    crop(dfEntrada).plot(ax=ax)
    crop(dfRiegoSup).plot(ax=ax, color='r')
    plt.ylabel('Caudal ($m^3/s$)')
    ax.grid()
    ax.legend(['Oferta hídrica superficial', 'Demanda hídrica superficial'])
    plt.savefig(os.path.join('balanceSuperficial.svg'),bbox_inches='tight')


def main():

    # correr modelo de embalse
    os.chdir(os.path.join('..', 'scripts'))

    # correr modelo hidrológico

    pathNam = os.path.join('..', 'modflow', 'mfnwt.nam')
    os.chdir(os.path.dirname(pathNam))
    modelo = model(pathNam, 'Copiapo', '1994-04-01')
    modelo.load()

    # makeDIS(modelo)
    # NWT(modelo.model)
    # makeOC(modelo.model)
    # makeWEL(modelo)
    # makeRCH(modelo)

    # incoporar la recarga del modelo superficial

    # escribir los paquetes
    # modelo.model.write_input(['WEL','OC','DIS','NWT','RIV','BAS6',
    #                           'UPW','CHD','RCH'])
    # modelo.model.write_input(['OC','DIS','NWT','RIV','BAS6',
    #                           'UPW','CHD','RCH'])
    # correr modelo de aguas subterráneas
    modelo.model.run_model(silent=False)

    processHeads(modelo)
    processBudget()

if __name__=='__main__':
    main()
