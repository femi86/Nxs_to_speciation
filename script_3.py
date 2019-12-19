# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:05:33 2019

@author: marafatto_f


on PI_38_w1_ar1
	- make small script that identifies the coordinates on a 2D map (mask) for the point XANES that is shown (assign filename at beginning):
"""
import os, glob, re, h5py
import numpy as np
from scipy import optimize as opt
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2

""" USER INPUT """

path = 'F:\\DLS_newdownload\\DLS_2019-4-05_14-47-12\\dls\\i18\\data\\2018\\sp18848-1'
path_eval = 'E:\\Dropbox\\Post-doc\\data\\Diamond_jul18\\PI_38_w1_ar4\\point_xanes'

refs_ox = 'E:\\Dropbox\\Post-doc\\data\\Diamond_jul18\\refs\\Tl_III'
refs_red ='E:\\Dropbox\\Post-doc\\data\\Diamond_jul18\\refs\\Tl_I'

path_norm = os.path.join(path_eval,'normalized')

LCF_path = 'E:\\Dropbox\\Post-doc\\data\\Diamond_jul18\\LCF_out\\LCF_out\\old\\PI_38_w1_ar1_out\\LCF'
chem_map_1 = np.array(Image.open(os.path.join(LCF_path,'PI_38_w1_ar1_rating.tif')))
chem_map_ox = np.array(Image.open(os.path.join(LCF_path,'PI_38_w1_ar1_2comp_normox_val.tif')))
chem_map_red = np.array(Image.open(os.path.join(LCF_path,'PI_38_w1_ar1_2comp_normred_val.tif')))
window = 3 # number of pixels around to average around

athena_norm = 1 # 1 if reading files normalized by athena, 0 if normalization here (not implemented yet)
path2 = os.path.join(path,"Experiment_1","nexus") # either read ascii, faster but requires more code, or read nexus file?
dir_out = os.path.join(path_eval,'point_XANES_output')
reduced_filelist = 0

twoD = '124336' # this is the 2d map related to the point XANES. taken at below the edge because it is the ref for allignment in the chem maps

bad_files = []
red_files = set(range(124518,124567)).difference(bad_files)

tx_num = 0.406 # the transmitted beam for the Tl La photons through 6 abs lengths of Ga filter
tx_den = 0.0160 # same as above but for Mn
att_filt = tx_den/tx_num
att_mass = 204.39/54.95 # molar mass ratio of the two elements

tot_att= att_mass / att_filt # this value is then multiplied by Mn to correct, or the reciprocal is multiplied by the Tl/Mn ratio

# here below the definition of the font
font = ImageFont.load_default()
fontsize = 0.5
fontthick = 1
height = fontsize+fontthick+2
border = 10 # the border to add at the top and bottom so that the labels can be read

""" FUNCTION DEFINITION """

def mask(X,Y,array, text): # find where the array with the mask is.
    box = np.zeros(array.shape)
    limits = cv2.getTextSize(text, font, fontsize, fontthick)
    box[X-1:X+limits[0][0]+1,Y-1:Y+limits[0][1]+1] = 1
    test = box+array
    falling_in = box*array
    return test,falling_in

""" AUTOMATED PART """

twoDfil = h5py.File([file2d for file2d in glob.glob(os.path.join(path,'processed','*.nxs')) if twoD in file2d][0],'r') # check if it works
if reduced_filelist:
    norm_files = [item for item in glob.glob(os.path.join(path_norm,'*.nor')) if [i for i in red_files if re.search('\d{'+str(len(twoD))+'}',item).group(0) == str(i)]]
    filelist = [item for item in glob.glob(os.path.join(path,path2,'*.nxs')) if [i for i in red_files if re.search('\d{'+str(len(twoD))+'}',item).group(0) == str(i)]]
else:
    norm_files = [item for item in glob.glob(os.path.join(path_norm,'*.nor'))]

filelist = [item for item in glob.glob(os.path.join(path2,'*.nxs')) if any(os.path.basename(item)[0:os.path.basename(item).find('.')] in i for i in norm_files)]
#%%
#
refs_red_ls = glob.glob(os.path.join(refs_red,'*.nor'))
refs_ox_ls = glob.glob(os.path.join(refs_ox,'*.nor'))

ref_list = refs_red_ls+refs_ox_ls

redox = ['red','ox']

len_red = range(0,len(refs_red_ls))
len_ox = range(len(refs_red_ls),len(ref_list))

couples = [(red,ox) for red in len_red for ox in len_ox]

dimXAN = "entry1/scan_dimensions"
t3x = "entry1/before_scan/t3x/t3x"
t3y = "entry1/before_scan/t3y/t3y"
energy = "entry1/qexafsXpress3FFIO/Energy"
Fluo = "entry1/qexafsXpress3FFIO/FFI0"

x2d = twoDfil['/entry/result/t3x'][()]
y2d = twoDfil['/entry/result/t3y'][()]
Tl_data = twoDfil['/entry/auxiliary/0-XRF Elemental Maps from ROIs/Tl-La/data'][()]
Mn_data = twoDfil['/entry/auxiliary/0-XRF Elemental Maps from ROIs/Mn-Ka/data'][()]*tot_att


"""SCRIPT"""

#- open the file where the 2D map is taken, read coordinate pairs (index, value)x (index, value)y and initialize a datacube with map
initialfile = h5py.File(filelist[0],'r')
sizeXan = initialfile[dimXAN][()]
# read the Tl-La and Mn-Ka

if athena_norm:
    sizeXan = np.shape(np.loadtxt(norm_files[0]))[0]
    LCF_energies = np.loadtxt(norm_files[0])[:,0]

#%% reference reading
if  len(str(int(LCF_energies[0])))> 4:
    digits = 5
else:
    digits = 4
ref_head = {} # range(len(ref_list))
N_ref = len(ref_list)
nor_vals = np.empty((len(LCF_energies),len(ref_list)))
for en,energy_ref in enumerate(LCF_energies):
    for index,reference in enumerate(ref_list):
        ref = np.loadtxt(reference)
        value = ref[np.where(np.around(ref[:,0],decimals=0)==energy_ref),3]
        nearest = ref[np.abs(ref[:,0]-energy_ref).argmin(),1] # this is to find the nearest value if it is not present in the list
        if value.size == 1:
            nor_vals[en,index] = value.item()
        elif value.size == 0:
            nor_vals[en,index] = nearest
        else:
            nor_vals [en,index] = np.mean(value)
        ref_name = os.path.basename(reference)
        if len(ref_name) < 10:
            ref_head[index] = ref_name[0:ref_name.find('.')] # this could be modified depending ont the names of the references.
        else:
            ref_head[index] = ref_name[0:10]
#####
#lo_spline = initialfile[energy][()][0:np.where(initialfile[energy][()]<)]
#np.where(initialfile[energy][()]<)
#hi_spline = initialfile[energy][()]

# initialize the matrix where to put the xanes

xanes_energies = ['xanes_energy_%s' % int(i) for i in np.loadtxt(norm_files[0])[:,0]]
LCF_names1 = ['LCF_rat_1ref','LCF_res_1','LCF_val1','LCF_rating_2ref1','LCF_rating_2ref2']
LCF_names2 = ['LCF_rating_val1','LCF_rating_val2','LCF_res_ref1','LCF_selection']
LCF_names = LCF_names1 + LCF_names2
filenames = ['Tl-La', 'Mn-Ka','Tl_Mn_ratio'] + ['pXAN_mask'] + xanes_energies + LCF_names

guide = np.zeros((len(filenames),len(y2d),len(x2d)))
guide[0,:,:] = Tl_data
guide[1,:,:] = Mn_data

ratio = Tl_data/np.ma.masked_less(Mn_data,1)
# here make a mask to remove crazy values (histogram and eliminate )
ratio = ratio/np.amax(ratio)
ratio = np.ma.masked_less_equal(ratio,1)
guide[2,:,:] = ratio

qx_norm = []
labels = []
# initialize the LCF
# get all possible combinations of references

text_out = {os.path.basename(xanesfil)[0:os.path.basename(xanesfil).find('.')]:[] for xanesfil in norm_files}
header1 = ['posX','posY','LCF_rating_1ref','val_LCF_1comp','LCF_rating_2ref']
header2 = ['val_LCF red vs ox','Tl(III) %','nssr_1comp','nssr_2comp','2 comp better than 1','LCF map on pixel: red vs ox or 1 comp rating']
header3 = [ 'Tl counts above edge', 'Tl/Mn ratio above edge']
header = header1 + header2 + header3

for ind,Xanes in enumerate(filelist):
    count = 3
    pos_ind = 1
    # read the nexus file to read the coordinates
    qxanes = h5py.File(Xanes,'r')
    try:
        posX, posY = list(x2d).index(np.around(qxanes[t3x][()],decimals=2)), list(y2d).index(np.around(qxanes[t3y][()],decimals=2))
    except: # this means that if it is beyond the map limits, it will put it on an edge.
        nearestX = x2d[np.abs(x2d-qxanes[t3x][()]).argmin()]
        nearestY = y2d[np.abs(y2d-qxanes[t3y][()]).argmin()]
        posX, posY = list(x2d).index(nearestX), list(y2d).index(nearestY)

    # normalize

    # read the position of E0 in the data

    # fit a linear spline for the pre-edge -->

    # fit a quadratic spline for post-edge -->

    # ...

    # read the athena-normalized data
    
    qx_norm = np.loadtxt(norm_files[ind])
    guide[count,posY,posX] = pos_ind
    labels.append([str(ind+1),posX,posY])
    count += 1
    pos_ind += 1
    guide[count:count+len(qx_norm),posY,posX] = qx_norm[:,3]
    count += len(qx_norm)
    LCF_sel_val = []
# LCF "rating" for best ref (1 component)
    LCF_res1, LCF_val1 = [], []
    for refr in range(len(ref_list)):
        fit_mat = np.vstack((nor_vals[:,refr],np.zeros(nor_vals[:,refr].shape))).transpose() # need this for the 1 component because otherwise it does not work...
        #fit_mat = np.vstack((nor_vals[:,ref],nor_vals[:,ref])).transpose() # we add a variable that is all ones to make it work, not sure if it does something bad or not...
        tmp = opt.lsq_linear(fit_mat,qx_norm[:,3],(0,1))
        # LCF_res.append(res)
        LCF_val1.append(np.array([x2 for x2 in tmp['x']]))
        LCF_res1.append(np.sum(np.array([x1**2 for x1 in tmp['fun']])))
    lowest = min(np.ma.masked_equal(LCF_res1,0))
    if lowest is not np.ma.masked:
        index1 = LCF_res1.index(lowest)
        guide[count,posY,posX] = index1+1 # need to mask the zero values because we are working with masked values, otherwise it will select the first ref always
        count +=1
        guide[count,posY,posX] = LCF_val1[index1][0]
        count += 1
        guide[count,posY,posX] = LCF_res1[index1]
        count +=1
    else:
        guide[count:count+3,posY,posX] = 0
        count +=3
    LCF_sel_val.append(lowest)
# LCF "rating" for best 2 component fit --> factorial combinations in loop
    LCF_res2 = []
    LCF_val2 = []
    for couple in couples:
        #fit_mat = np.vstack((nor_vals[:,ref],nor_vals[:,ref])).transpose() # we add a variable that is all ones to make it work, not sure if it does something bad or not...
        tmp = opt.lsq_linear(nor_vals[:,couple],qx_norm[:,3],(0,1))
        # LCF_res.append(res)
        LCF_val2.append(np.array([x2 for x2 in tmp['x']]))
        LCF_res2.append(np.sum(np.array([x1**2 for x1 in tmp['fun']])))
    lowest = min(np.ma.masked_equal(LCF_res2,0))
    if lowest is not np.ma.masked:
        index2 = LCF_res2.index(lowest)
        guide[count:count+2,posY,posX] = [x+1 for x in couples[index2]] # need to mask the zero values because we are working with masked values, otherwise it will select the first ref always
        count += 2
        guide[count:count+len(tmp['x']),posY,posX] = LCF_val2[index2]
        count += len(tmp['x'])
        guide[count,posY,posX] = LCF_res2[index2]
        count += 1
    else:
        guide[count:count+2,posY,posX] = 0
        count += 2
        guide[count:count+len(tmp['x']),posY,posX] = 0
        count += len(tmp['x'])
        guide[count,posY,posX] = 0
        count += 1
    LCF_sel_val.append(lowest)
# evaluate whether 2 component fit to be kept or only 1 (only if res is 20% lower than 1 component)
# this will be only saved in text file?
    if LCF_sel_val[1] <= 0.8*LCF_sel_val[0]:
        guide[count,posY,posX] = 2 # value of 2 means the 2 component is better
        correl = [np.nanmean(chem_map_red[posX-window:posX+window,posY-window:posY+window]),np.nanmean(chem_map_ox[posX-window:posX+window,posY-window:posY+window])]
    else:
        guide[count,posY,posX] = 1 # value of 1 means the 1 component is better
        correl = [np.nanmean(chem_map_red[posX-window:posX+window,posY-window:posY+window]),np.nanmean(chem_map_ox[posX-window:posX+window,posY-window:posY+window])]
    ox_perc = LCF_val2[index2][1]/np.sum(LCF_val2[index2]) # convert the LCF results in % TlIII

    
    text_1_3 = [[posX,qxanes[t3x][()]],[posY,qxanes[t3y][()]],ref_head[index1]]
    text_4_6 = [np.around(LCF_val1[index1][0],decimals=3),[ref_head[ind] for ind in couples[index2]],np.around(LCF_val2[index2],decimals=3)]
    text_7_10 = [ox_perc,np.around(LCF_res1[index1],decimals=3),np.around(LCF_res2[index2], decimals=3),[LCF_sel_val[1] <= 0.8*LCF_sel_val[0]],correl]
    try:
        text_11_13 = [Tl_data[posX,posY],ratio[posX,posY]]
    except:
        text_11_13 = [np.nan, np.nan]
    text_out[os.path.basename(Xanes)[0:os.path.basename(Xanes).find('.')]] = text_1_3 + text_4_6 + text_7_10 + text_11_13

try:
    os.stat(dir_out)
except:
    os.makedirs(dir_out)

# save results in 2d maps
for cnt,immap in enumerate(guide):
    old_img = Image.fromarray(immap)
    img = ImageOps.expand(old_img,border=border,fill='black')
    img_name = os.path.join(dir_out,filenames[cnt]+'.tif')
    img.save(img_name)
    # here below make the labeled image
    if filenames[cnt] =='pXAN_mask':
        fullmap = np.copy(immap)
        img1 = ImageOps.expand(Image.fromarray(fullmap),border=0,fill='black')
        draw1 = ImageDraw.Draw(img1)
        for ind,lab in enumerate(labels):
            newmap = np.copy(immap)
            img = ImageOps.expand(Image.fromarray(newmap),border=border,fill='black')
            xy = [lab[1]+border,lab[2]+border]
            draw = ImageDraw.Draw(img)
            draw.ellipse([(xy[0],xy[1]),(xy[0]+2,xy[1]+2)],fill=1)
            draw1.ellipse([(xy[0],xy[1]),(xy[0],xy[1])],fill=1)
            draw.text((xy[0]+2,xy[1]+2),'p'+lab[0],font=font)
            draw1.text((xy[0]+2,xy[1]+2),'p'+lab[0],font=font)
            del draw
            #cv2.putText(img,'p'+lab[0],(xy[0]+2,xy[1]+2),font,fontsize,(255,255,0),fontthick,cv2.LINE_AA)                
            img.save(os.path.join(dir_out,'pXAN_mask_labeled%i.tif' % (ind+1)))
        del draw1
        img1.save(os.path.join(dir_out,'pXAN_mask_labels_all.tif'))
        

out = pd.DataFrame(data=text_out, index = header)
out = out.transpose()
out.to_csv(os.path.join(path_eval,'results_list.txt'),sep='\t')
# save index values and ref names:
tmp = pd.DataFrame(data={str(index+1):ref for index,ref in enumerate(ref_head)},index = ['reference'])
tmp = tmp.transpose()
tmp.to_csv(os.path.join(path_eval,'ref_list.txt'),sep='\t')
print('DONE!!!')
