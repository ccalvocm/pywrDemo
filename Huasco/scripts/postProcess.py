# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 08:35:05 2022

@author: ccalvo
"""

from flopy.utils import HeadFile
import os
from flopy.export import vtk
import flopy

def main():

    # dirs
    model_ws=os.path.join('..','modflow')
    head_file = os.path.join(model_ws, "mfnwt.hds")
    outFile=os.path.join('.','modflow','run','heads.vtk')
    output_dir=os.path.join('..','modflow','out')
    well_output_dir = os.path.join(output_dir, "WEL")
    dis_output_dir = os.path.join(output_dir, "DIS")
    bas_output_dir = os.path.join(output_dir, "BAS")
    upw_output_dir = os.path.join(output_dir, "UPW")
    lpf_output_dir = os.path.join(output_dir, "LPF")
    riv_output_dir = os.path.join(output_dir, "RIV")


    # create the vtk object and export heads
    nam_file=os.path.join(model_ws, "mfnwt.nam")
    # load model
    ml = flopy.modflow.Modflow.load(nam_file, model_ws='.', check=False)
    
    
    from flopy.export.utils import model_export
    def exportModel():
        model_export('Copiapo.vtk',ml,fmt='vtk',epsgint=24879)


    # export vectors
    # import the CellBudgetFile reader and read the CBC file
    from flopy.utils import CellBudgetFile
    
    cbc_file = os.path.join(model_ws, "mfnwt.cbc")
    cbc = CellBudgetFile(cbc_file)
    
    # get frf, fff, flf from the Cell Budget file (or SPDIS when using MF6)
    from flopy.utils import postprocessing
    
    frf = cbc.get_data(text="FLOW RIGHT FACE", kstpkper=(0, 300), full3D=True)[0]
    fff = cbc.get_data(text="FLOW FRONT FACE", kstpkper=(0, 300), full3D=True)[0]
    flf = cbc.get_data(text="FLOW LOWER FACE", kstpkper=(0, 300), full3D=True)[0]
    
    spdis = postprocessing.get_specific_discharge((fff, frf, flf), ml)
    
    # create the Vtk() object
    vtkobj = vtk.Vtk(ml, vertical_exageration=10)
    
    # add the vector
    vtkobj.add_vector(spdis, name="spdis")
    
    # write to file
    vtkobj.write(os.path.join(output_dir, "spdis_vector.vtu"))


def exportHds(ml,head_file,model_ws):
    
    hds = HeadFile(head_file)   
    # export heads
    vtkobj = vtk.Vtk(model=ml)
    vtk.export_heads(ml,head_file,'.',kstpkper=[(0,2)])
    
def exportWel(ml,well_output_dir):
    ml.wel.export(well_output_dir, fmt="vtk")
    
def exportDis(ml,dis_output_dir):
    ml.dis.export(dis_output_dir, fmt="vtk",vertical_exageration=6)

def exportBas(ml,bas_output_dir):
    ml.bas6.export(bas_output_dir, fmt="vtk")

def exportUpw(ml,upw_output_dir):
    # Make sure spatial reference set to epsg 24879 to create RCH and WEL packages
    xul = 309000
    yul = 6993800
    rot = 0
    model_epsg = 24879
    ml.sr = flopy.utils.reference.SpatialReference(delr=ml.dis.delr.array, delc=ml.dis.delc.array, xul=xul, yul=yul, rotation=rot, epsg=model_epsg)

    ml.upw.export(upw_output_dir, fmt="vtk")

def exportLpf(ml,lpf_output_dir):
    ml.lpf.export(lpf_output_dir, fmt="vtk")
