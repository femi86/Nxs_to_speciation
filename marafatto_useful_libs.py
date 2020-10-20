# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:56:17 2020


@author: marafatto_f


collection of useful functions and procedures that I use often
import them all as mul, as a shorthand name, preferred compared to star imports
(easier for debugging)

"""

import os,sys,glob
import numpy as np
import cv2
import pandas as pd
import scipy as sp


def safe_mkdir(file_path,a = False):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if a:
        return file_path
  
def progbar(text,curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    #print('\r','\t '+'#'*filled_progbar + '-'*(full_progbar-filled_progbar), '{:>7.2%}'.format(frac), end='')
    text = '\t'+text+': '+'#'*filled_progbar + '-'*(full_progbar-filled_progbar)+ '{:>7.2%}'.format(frac)
    sys.stdout.write('\r'+text)
    sys.stdout.flush()
    
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def mov_avg(array,movavg_shape,mov_bin):
    if movavg_shape == 1:
        result = cv2.GaussianBlur(array,(mov_bin,mov_bin),0)
    elif movavg_shape == 0:
        result = cv2.blur(array,(mov_bin,mov_bin))
    return result

def RGB_fun(arR,arG,arB):
    R = np.array(np.around(arR/np.nanmax(arR)*256),np.uint8)
    G = np.array(np.around(arG/np.nanmax(arG)*256),np.uint8)
    B = np.array(np.around(arB/np.nanmax(arB)*256),np.uint8)
    RGB = cv2.merge((B,G,R))
    return RGB
    
def ROIs(sel_roi,el_norm, num, den, directory,scl): 
    '''
    this function reads in alligned images from a chemical map, allows the user to
    select a series of rois, then saves each selected roi in individual objects
    for further independent processing.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    ROI = {} # first roi is the fullmap
    roi_coord = {}
    roi_masks = {}
    loop = True
    count = 1
    El1 = num
    empty = np.zeros(np.shape(El1))
    canv_incr = 3
    def inc_canv(array): # this is set for rgb images, so considers a 3d array where 3rd dimension is RGB
        row = np.zeros((canv_incr,np.shape(array)[1],3),np.uint8)
        col = np.zeros((np.shape(array)[0]+canv_incr*2,canv_incr,3),np.uint8)
        return np.hstack((col,np.vstack((row,array,row)),col))
    El2 = den
    image1 = RGB_fun(El1/np.nanmax(El1)*1.1,El2/np.nanmax(El2)*1.1, empty)
    image2 = RGB_fun(El1*1.1,El2*1.1,empty)
    image3 = RGB_fun(El1,El2,empty)
    try:
        image4 = cv2.imread(os.path.join(directory,'fullmap','LCF', 'fullmap_redR_nanG_oxB_El1IntScale.tif'))
    except:
        print('fullmap not available, check the directory, or run full lcf first')
        image4 = RGB_fun(empty,empty,empty)    
    outimage1 = inc_canv(image1)
    outimage2 = inc_canv(image2)
    imSh = outimage2.shape
    outimage2 = cv2.resize(outimage2,(int(imSh[0])*scl,int(imSh[0])*scl))
    outimage3 = inc_canv(RGB_fun(empty,empty,empty))
    images = [outimage1,outimage2,outimage3]
    image_names = ['ROIs_REl1_GEl2_debug.tif','ROIs_El1/El2.tif','only_areas.tif']
    if not sel_roi:
        try:
            roi_coord = pd.read_csv(os.path.join(directory,'roi_coordinates.csv'))
        except:
            raise  ValueError('no such file as roi_coordinates.csv in the output folder')
            print(ValueError)
        for r in range(roi_coord.shape[0]):
            X1,Y1,width,height = int(roi_coord['X1'][r]),int(roi_coord['Y1'][r]),int(roi_coord['width'][r]),int(roi_coord['height'][r])
            ROI['roi_n'+str(r+1)] = el_norm[:,Y1:Y1+height, X1:X1+width]
            roi_masks['roi_n'+str(r+1)] = [Y1,Y1+height,X1,X1+width]
            X1_shift, Y1_shift = X1+canv_incr, Y1+canv_incr
            center = (X1_shift,Y1_shift-2)
            for image in images:
                cv2.rectangle(image,(X1_shift,Y1_shift),(X1_shift+width,Y1_shift+height),(255,0,0),2)
                cv2.putText(image,'ROI'+str(r+1),center,font,0.5,(255,0,0),1) 
        [cv2.imwrite(os.path.join(directory,img_n),image) for image in images for img_n in image_names]

    #%%
    elif sel_roi:
        try:
            os.rename(os.path.join(directory,'roi_coordinates.csv'),
                      os.path.join(directory,'roi_coordinates.csv.bkup'))
        except:
            print('there was no previous coordinates file to backup')
        while loop:
            roi_ID = 'ROI{}'.format(str(count))
            cv2.imshow('El1 red vs El2 blue map',outimage2)
            print('select on El1 vs EL2 image Roi n.{}, type "c" to cancel selection,',
                  'then hit \n enter twice to continue, or \n enter + q to stop'
                  .format(roi_ID))
            r = cv2.selectROI('El1 red, EL2 green, scaled by El1 and El2 and contrast',image1,False) # the image here should by El1/El2, with also a plot of El1 max intensity?
            r1 = (int(r[0]/scl),int(r[1]/scl),int(r[2]/scl),int(r[3]/scl))
            roi_coord['roi'+str(count)] = [r1[0],r1[1],r1[2],r1[3]]
            k = cv2.waitKey(0) # this is to read the quit signal
            if k == ord('q'):
                loop=False
                for n,image in enumerate(images):
                    cv2.imwrite(os.path.join(directory,image_names[n]),image)
                print('ok, exiting roi selection')
            else:
                ROI['roi_n'+str(count)] = el_norm[:,r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
                roi_masks['roi_n'+str(count)] = [r1[1],r1[1]+r1[3],r1[0],r1[0]+r1[2]]
                X1_shift, Y1_shift = r1[0]+canv_incr, r1[1]+canv_incr
                center = (X1_shift,Y1_shift-1)
                for image in images:
                    cv2.rectangle(image,(X1_shift,Y1_shift),(X1_shift+r1[2],Y1_shift+r1[3]),(255,0,0),2)
                    cv2.putText(image,roi_ID,center,font,0.5,(255,0,0),1)
                count +=1
            cv2.destroyWindow('El1 red vs El2 blue map')
            cv2.destroyWindow('El1 red, El2 green, scaled by El1 and EL2 and contrast') 
        tmp = pd.DataFrame.from_dict(data=roi_coord,orient='index',columns=['X1','Y1','width','height'])
        tmp.to_csv(os.path.join(directory,'roi_coordinates.csv'))
    return ROI, roi_masks, roi_coord

def pixelLCF(shape,el_norm,nor_vals,ref_list):
    LCF_1comp = np.zeros((shape[0],shape[1]))
    LCF_1comp_val = np.zeros((2,shape[0],shape[1]))
    LCF_1comp_res = np.zeros((shape[0],shape[1]))
    for x in range(shape[0]):
        for y in range(shape[1]):
            LCF_res, LCF_val = np.empty(len(ref_list)),np.empty((len(ref_list),2))
            for ref in range(len(ref_list)):
                fit_mat = np.vstack((nor_vals[:,ref],np.zeros(nor_vals[:,ref].shape))).transpose() # we add a variable that is all zeros to make it work, not sure if it does something bad or not...
                tmp_1 = sp.optimize.lsq_linear(fit_mat,el_norm[:,x,y],(0,1)) # for the moment does not work with the general lsq_linear
                LCF_res[ref] = np.sum(np.array([x1**2 for x1 in tmp_1['fun']]))
                LCF_val[ref,:] = np.array([x2 for x2 in tmp_1['x']])
            try:
                lowest = np.amin(np.ma.masked_equal(LCF_res,0))
                index = np.where(LCF_res == lowest)[0][0]
                LCF_1comp[x,y] = index+1 # need to mask the zero values because we are working with masked values, otherwise it will select the first ref always
                LCF_1comp_val[:,x,y] = LCF_val[index]
                LCF_1comp_res[x,y] = LCF_res[index]
            except:
                LCF_1comp[x,y] = 0
                LCF_1comp_val[:,x,y] = 0
        progbar('pixelLCF',x,shape[0],20)
    return LCF_1comp, LCF_1comp_res, LCF_1comp_val

def pixel_2compLCF(shape,el_norm,nor_vals,couples):
    LCF_2comp = np.zeros((2,shape[0],shape[1]))
    LCF_2comp_val = np.zeros((2,shape[0],shape[1]))
    LCF_2comp_res = np.zeros((shape[0],shape[1]))
    for x in range(shape[0]):
        for y in range(shape[1]):
            LCF_res_2, LCF_val_2 = np.empty(len(couples)), np.empty((len(couples),2))
            for cpind,couple in enumerate(couples): # the couples are such that the first is the reduced, the second the oxidized
                tmp_2 = sp.optimize.lsq_linear(nor_vals[:,couple],el_norm[:,x,y],(0,1))
                LCF_val_2[cpind,:] = np.array([x2 for x2 in tmp_2['x']])
                LCF_res_2[cpind] = np.sum(np.array([x1**2 for x1 in tmp_2['fun']]))
            try:
                lowest2 = np.amin(np.ma.masked_equal(LCF_res_2,0))
                index2 = np.where(LCF_res_2 == lowest2)[0][0]
                LCF_2comp[:,x,y] = [x3+1 for x3 in couples[index2]]
                LCF_2comp_val[:,x,y] = LCF_val_2[index2]
                LCF_2comp_res[x,y] = LCF_res_2[index2]
            except:
                LCF_2comp[:,x,y] = 0
                LCF_2comp_val[:,x,y] = 0
                LCF_2comp_res[x,y] = 0
        progbar('2comp_lcf',x,shape[0],20)
    return LCF_2comp, LCF_2comp_res, LCF_2comp_val

def LCF_sel(shape,len_red,LCF_2comp, LCF_2comp_val,LCF_2comp_res,LCF_1comp, LCF_1comp_res):
    LCF_select = np.full((2,shape[0],shape[1]),np.nan)
    LCF_sel_oxstate = np.full((2,shape[0],shape[1]),np.nan)
    for x in range(shape[0]):
        for y in range(shape[1]):
            if LCF_2comp_res[x,y] <= 0.8*LCF_1comp_res[x,y]:
                LCF_select[:,x,y] = LCF_2comp[:,x,y]
                LCF_sel_oxstate[:,x,y] = LCF_2comp_val[:,x,y]/np.sum(LCF_2comp_val[:,x,y]) # this will populate with the values of the components, scaled to 1
            else:
                if LCF_1comp[x,y] < len_red:
                    LCF_select[0,x,y] = LCF_1comp[x,y]
                    LCF_sel_oxstate[0,x,y] = 1
                else:
                    LCF_select[1,x,y] = LCF_1comp[x,y]
                    LCF_sel_oxstate[1,x,y] = 1
    return LCF_select, LCF_sel_oxstate

def pixelLCF_mono(el_norm,nor_vals,ref_head):
    LCF_1comp_val = np.zeros((2))
    LCF_res, LCF_val = np.empty(len(ref_head)),np.empty((len(ref_head),2))
    for ref in range(len(ref_head)):
        fit_mat = np.vstack((nor_vals[:,ref],np.zeros(nor_vals[:,ref].shape))).transpose() # we add a variable that is all ones to make it work, not sure if it does something bad or not...
        tmp_1 = sp.optimize.lsq_linear(fit_mat,el_norm[:],(0,1)) # for the moment does not work with the general lsq_linear
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

def pixel_2compLCF_mono(el_norm,nor_vals,couples,ref_head):
    LCF_2comp = []
    LCF_2comp_val = np.zeros((2))
    LCF_res_2, LCF_val_2 = np.empty(len(couples)), np.empty((len(couples),2))
    for cpind,couple in enumerate(couples): # the couples are such that the first is the reduced, the second the oxidized
        tmp_2 = sp.optimize.lsq_linear(nor_vals[:,couple],el_norm[:],(0,1))
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

def LCF_sel_mono(LCF_2comp, LCF_2comp_val,LCF_2comp_res,LCF_1comp, LCF_1comp_res):
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