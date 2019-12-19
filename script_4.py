# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:38:59 2019

@author: marafatto_f
"""
import os, glob
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import h5py


path = r'E:\Dropbox\Post-doc\data\Diamond_jul18\PI_38_w1_ar1\point_xanes\not_normalized'
path_nexus = r'F:\DLS_newdownload\DLS_2019-4-05_14-47-12\dls\i18\data\2018\sp18848-1\Experiment_1\nexus'

folder = glob.glob(os.path.join(path,'*.xmu'))
LCF_energies = ['12625','12630','12658','12671','12677','12686','12726','12755']
refs_ox = r'E:\\Dropbox\\Post-doc\\data\\Diamond_jul18\\PI_38_w1_ar1\\point_xanes\\refs\\Tl_III'
refs_red = r'E:\\Dropbox\\Post-doc\\data\\Diamond_jul18\\PI_38_w1_ar1\\point_xanes\\refs\\Tl_I'

list_nexus = glob.glob(os.path.join(path_nexus,'*.nxs'))
list_nexus = [file for file in list_nexus if any(os.path.basename(file)[0:6] in x for x in folder)]

fullspectrum = '/entry1/qexafsXspress3FFI0/AllElementSum'
energy_nxs = '/entry1/qexafsXspress3FFI0/Energy'
I0 = '/entry1/qexafsXspress3FFI0/I0'
channels_els = [(545,605),(975,1032),(1034,1087)] #@ in order, the channel ranges for Mn, Tl and As

tx_Tl = 0.4061
tx_Mn = 0.0016
tx_As = 0.00033

att_mass_Tl = 204.39*165.033 #g/mol * cm2/g
att_mass_Mn = 54.95*77.955
att_mass_As = 74.92*148.610


att_Tl = tx_Tl * att_mass_Tl
att_Mn = tx_Mn * att_mass_Mn
att_As = tx_As * att_mass_As

refs_red_ls = glob.glob(os.path.join(refs_red,'*.nor'))
refs_ox_ls = glob.glob(os.path.join(refs_ox,'*.nor'))
redox = ['red','ox']
ref_list = refs_red_ls+refs_ox_ls
len_red = range(0,len(refs_red_ls))
len_ox = range(len(refs_red_ls),len(ref_list))
couples = [(red,ox) for red in len_red for ox in len_ox]

ref_head = [os.path.basename(ref)[0:10] for ref in ref_list]
N_ref = len(ref_list)
nor_vals = np.zeros((len(LCF_energies),len(ref_list)))
energy_refs = [float(os.path.basename(energy)) for energy in LCF_energies]
for index,reference in enumerate(ref_list):
	ref = np.loadtxt(reference)
	for en,energy in enumerate(energy_refs):
		value = ref[np.where(np.around(ref[:,0],decimals=0)==energy),1]
		nearest = ref[np.abs(ref[:,0]-energy).argmin(),1]
		if value.size == 1:
			nor_vals[en,index] = value.item()
		elif value.size == 0:
			nor_vals[en,index] = nearest
		else:
			nor_vals[en,index] = np.mean(value)
nor_vals = nor_vals / nor_vals[-1]

def pixelLCF(el_norm,nor_vals,ref_list):
    LCF_1comp_val = np.zeros((2))
    LCF_res, LCF_val = np.empty(len(ref_list)),np.empty((len(ref_list),2))
    for ref in range(len(ref_list)):
        fit_mat = np.vstack((nor_vals[:,ref],np.zeros(nor_vals[:,ref].shape))).transpose() # we add a variable that is all ones to make it work, not sure if it does something bad or not...
        tmp_1 = optimize.lsq_linear(fit_mat,el_norm[:],(0,1)) # for the moment does not work with the general lsq_linear
        LCF_res[ref] = np.sum(np.array([x1**2 for x1 in tmp_1['fun']]))
        LCF_val[ref,:] = np.array([x2 for x2 in tmp_1['x']])
        try:
            lowest = np.amin(np.ma.masked_equal(LCF_res,0))
            index = np.where(LCF_res == lowest)[0][0]
            LCF_1comp = ref_head[index+1] # need to mask the zero values because we are working with masked values, otherwise it will select the first ref always
            LCF_1comp_val[:] = LCF_val[index]
            LCF_1comp_res = LCF_res[index]
        except:
            LCF_1comp = []
            LCF_1comp_val[:] = 0
            LCF_1comp_res = 0
            
    return LCF_1comp, LCF_1comp_res, LCF_1comp_val

def pixel_2compLCF(el_norm,nor_vals,couples):
    LCF_2comp = []
    LCF_2comp_val = np.zeros((2))
    LCF_res_2, LCF_val_2 = np.empty(len(couples)), np.empty((len(couples),2))
    for cpind,couple in enumerate(couples): # the couples are such that the first is the reduced, the second the oxidized
        tmp_2 = optimize.lsq_linear(nor_vals[:,couple],el_norm[:],(0,1))
        LCF_val_2[cpind,:] = np.array([x2 for x2 in tmp_2['x']])
        LCF_res_2[cpind] = np.sum(np.array([x1**2 for x1 in tmp_2['fun']]))
        try:
            lowest2 = np.amin(np.ma.masked_equal(LCF_res_2,0))
            index2 = np.where(LCF_res_2 == lowest2)[0][0]
            LCF_2comp = [ref_head[x3] for x3 in couples[index2]]
            LCF_2comp_val[:] = LCF_val_2[index2]
            LCF_2comp_res = LCF_res_2[index2]
        except:
            LCF_2comp = ['','']
            LCF_2comp_val[:] = 0
            LCF_2comp_res= 0
    return LCF_2comp, LCF_2comp_res, LCF_2comp_val 

def LCF_sel(LCF_2comp, LCF_2comp_val,LCF_2comp_res,LCF_1comp, LCF_1comp_res):
    LCF_sel_oxstate = np.zeros((2))
    if LCF_2comp_res <= 0.8*LCF_1comp_res:
        LCF_select = LCF_2comp
        LCF_sel_oxstate[:] = np.ma.fix_invalid(LCF_2comp_val[:]/np.sum(LCF_2comp_val[:]),fill_value=0) # this will populate with the values of the components, scaled to 1
    else:
        LCF_select = LCF_1comp
        LCF_sel_oxstate[:] = np.nan
        # this part below is to put the 1 component fit as reduced or oxidized
#                if LCF_1comp[x,y] < len(len_red):
#                    LCF_sel_oxstate[0,x,y] = LCF_1comp_val[0,x,y]
#                else:
#                    LCF_sel_oxstate[1,x,y] = LCF_1comp_val[0,x,y]
    return LCF_select, LCF_sel_oxstate

text_lcf_1 = ['LCF_red_comp','LCF_ox_comp','LCF_1comp','LCF_residual_2comp','LCF_residual_1comp','LCF_red_value']
text_lcf_2 = ['LCF_ox_value', 'LCF_normred_value','LCF_normrox_value']
text_lcf_3 = ['2 comp better than 1','Tl cts','Tl/Mn ratio','Tl/As ratio']

text_lcf = text_lcf_1 + text_lcf_2 + text_lcf_3
raw_pxanes = {os.path.basename(filename):np.zeros(len(LCF_energies)) for filename in folder}
LCF = {os.path.basename(filename):{} for filename in folder}

for indx,filename in enumerate(folder):
    tmparray = np.loadtxt(filename)
    index = os.path.basename(filename)
    with h5py.File(list_nexus[indx],'r') as inputf:
        Fspect = inputf[fullspectrum][()]/inputf[I0][()][:,None]
        energy_ind = pd.Series(np.around(inputf[energy_nxs][:],decimals=0))
        
        Mn = np.sum(Fspect[:,channels_els[0][0]:channels_els[0][1]],axis=1)/(channels_els[0][1]-channels_els[0][0])
        As = np.sum(Fspect[:,channels_els[1][0]:channels_els[1][1]],axis=1)/(channels_els[1][1]-channels_els[1][0])
        Tl_x = np.sum(Fspect[:,channels_els[2][0]:channels_els[2][1]],axis=1)/(channels_els[2][1]-channels_els[2][0])
        
        Above = np.average(Tl_x[energy_ind[energy_ind==12710].index[0]:energy_ind[energy_ind==12730].index[0]])
        Below = np.average(Tl_x[energy_ind[energy_ind==12640].index[0]:energy_ind[energy_ind==12650].index[0]])
        Tl_v = (Above-Below) / att_Tl
        Mn_v = np.average(Mn[energy_ind[energy_ind==12710].index[0]:energy_ind[energy_ind==12730].index[0]]) / att_Mn
        As_v = np.average(As[energy_ind[energy_ind==12710].index[0]:energy_ind[energy_ind==12730].index[0]]) / att_As
    
    for en,energy in enumerate(energy_refs):
        value = tmparray[np.where(np.around(tmparray[:,0],decimals=0)==energy),1]
        nearest = tmparray[np.abs(tmparray[:,0]-energy).argmin(),1]
        if value.size == 1:
            raw_pxanes[index][en] = value.item()
        elif value.size == 0:
            raw_pxanes[index][en] = nearest
        else:
            raw_pxanes[index][en] = np.mean(value) 
    raw_pxanes[index] = raw_pxanes[index] / raw_pxanes[index][-1] # to normalize the data like the references
    pix_1_comp, pix_1_res, pix_1_val  = pixelLCF(raw_pxanes[index],nor_vals,ref_list)
    pix_2_comp, pix_2_res, pix_2_val = pixel_2compLCF(raw_pxanes[index],nor_vals,couples)
    sel_lcf, sel_lcf_oxstate = LCF_sel(pix_2_comp,pix_2_val,pix_2_res,pix_1_comp,pix_1_res)
    sel = pix_2_res < 0.8*pix_1_res
    
    out1, out2, out3 = pix_2_comp, pix_2_res, pix_2_val
    if sel:
        out1_5 = [],[]
    else:
        out1_5 = pix_1_comp, pix_1_res
    out4 = out3/np.sum(out3)
    out5 = sel
    
    LCF[index] = [out1[0],out1[1],out1_5[0],out1_5[0], out2, out3[0], out3[1], out4[0],out4[1], out5, Tl_v, Tl_v/Mn_v, Tl_v/As_v]

raw = pd.DataFrame.from_dict(raw_pxanes,orient='index',columns=energy_refs)
lcf_out = pd.DataFrame.from_dict(LCF,orient='index',columns=text_lcf)

writer = pd.ExcelWriter(os.path.join(path,'output.xlsx'),engine='xlsxwriter')
raw.to_excel(writer,sheet_name='raw_data')
lcf_out.to_excel(writer,sheet_name='LCF_out')
writer.save()

