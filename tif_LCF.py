# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 06 10:30:50 2019

@author: marafatto_f
"""
'''
Procedure / algortithm / rationale

This script's purpose is to read the rois in a set of maps (based on energy)

inputs:
dir in = dir_out_root of previous script. the rationale is that each sample
folder has subfolders for each energy with each energy having subfolders for the
full spectrum and for the rois. data is saved as tiffs in each. the full spectrum
tiffs can be read by PyMCA easily.
elements allignment = find the element in the filename
references = read in the references and normalize them (athena normalized files)
files to bundle = same filename + energies. optionally, give the energy list
dirs out = LCF, allignment, masks


requirements:
    - make variable with sample names (read dir) --> outside loop
    - make variable with energy names (read subdirs) --> inside each loop, 
    loop through the energies
    - put references in the main folder path as normalized athena files 
    (individual), save them with a short filename that has the name in max 
    9 characters.


operations to do:
        - read all images in a 3/4D array (energies, roi maps) --> 
        pillow or numpy or skimage (read image stack?)
        - make 2D list with energies read in, roi names/filenames
        - allignment --> find index of image to allign in list, apply 
        allignment to all, save in folder allignment

        - make cutoff masks on 15 keV map (mask_cut):
            - histogram (np.histogram) El1 background vs concretions 
            --> evaluate counts and mask (all below value are 0)
            - histogram (np.histogram) El2 counts background --> histogram, 
            set value for bg, select only values that are 5-10 times bg

        - make masks based on cutoff (mask):
            - apply cutoff mask
            - divide El1/El2 image (15 keV)
            - np.histogram, select ranges (maskN)
        - LCF_all of all images
        - LCF_mask of masked regions:
            - option 1: LCF of all, multiply by maskN, average all values
            - option 2: raw images loop multiply by maskN (increase dimension 
                        array by N), average values within masks, LCF
        - fromLCF_mask:
            - save/plot XANES obtained, save XANES reference only values

the principle of this script is to read into dictionaries all the files, this
 makes it easier to deal with the data and keeps a track of the data along the 
 whole script.also, it does not require initialization of the variables, which 
 can provide useful.However, it may require some thoughts of how to access the
 actual data when dictionaries are nested within eachother. Also, there is 
 quite a use of list comprehensions, that make the code shorter but may be
 more cryptic to understand.

Notes for future edits:

LCF algorithm is now a non negative least squares method, but in the results
we get ratios that are beyond 1. We should be able to constrain the ratios 
from 0 to 1. may be due to normalization?

Also, the normalization may not be optimal now.



Update 2020 07 30

generalization of the script. the script reads in maps at different energies, 
alligns them based on one element that does not change and has enough features
for the algorithm, and then gives the results in terms of LCF (element 1 at 2 
oxidation states), in terms of element 1 / element 2 ratios, and also with a
third element if needed.

it reads in an hdf5 file that is generated from another script, that reads in 
FDA files


'''
import os, glob, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.io import imread
import scipy as sp
import pandas as pd
import seaborn as sns
import cv2
import imageio
import marafatto_useful_libs as mul
import h5py
from difflib import SequenceMatcher

'''
user input
'''
# DIRECTORIES (values):
path: str = r'C:\Dropbox\Post-Doc_II\data\phoenix\2021\data\2021\02\24\2021_0224_225705_I_3_MF_pell_OXS'
# path: str = r'C:\Dropbox\Post-Doc_II\data\phoenix\2021\data\2021\02\25\2021_0225_053706_I_4_MF_pell_OXS'
run_all = 1  # select 1 to run all or 0 to only run one, specify in list below
run = ['XAS']  # name which map should be run, if run_all = 0
# this below should be one subfolder inside called 'ox' and one 'red'
references: str = r'C:\Dropbox\Post-Doc_II\data\phoenix\2021\data\2021\02\refs'
ignore = ['old', 'pymca', 'test', ]  # folders in the input dir to ignore
cus_nm = ''  # a custom name to give to the outputs
# PRE-PROCESSING
det_resum = 1
pymca_set = 0

# PROCESSING:
ROI_opt = 0
# 1 if you want to run on rois, 0 if you want to run the full map as one 
# single roi. will contain also pixel-wise LCF
sel_roi = 1
# 1 if you want to select new rois, 0 if you want to use ones that were predefined before
roi_lcf = 1
# 0 if you just want to just do the averages of the rois, 1 if you 
# want to get the images of the rois with lcf pixel by pixel
crit = -1  # default 0.8, if set to -1 it will allways select the 2 component fit,
# values between 0 and 1 determine how better the 2 component fit residual needs
# to be to choose the 2 component fit over the 1 component fit.
moving_avg = 0  # if to make a moving average on the data or not
movavg_shape = 0
# shape of the moving average, 1 for gaussian blur or 0 for normal blur
m_b = 3  # the bin for the moving average
image_scl = 1

mask = 0  # select if you want to mask the numerator
edge = 0
# edge step filter option, otherwise mask is values in high 
# energy map higher than avg of low energy map
edge_ratio = 3  # edge step threshold to accept (x times the edge step)
El1_masking = 1  # 1 if you want to mask based on an element, 0 if not
El1_threshold = 600
# the value for the hi energy, attenuation corrected 
# El map for which the El background is eliminated

all_en = -1  # the energy to use as the allignment, default 0
alg_el = 'SiKa'  # element to use for the allignment
r_El1 = 'SKa'  # element for the numerator, and energy dataset. this is the one the LCF is run on
r_El2 = 'SiKa'  # element to use as element 2,and energy dataset
r_El3 = 'FeL'  # element to use as element 3, and energy dataset
X_name = 'ScanX_set'
Y_name = 'ScanY_set'
out_chem = 'none'  # energy not part of chemical imaging

# pymca datasets to choose if looking at the deconvoluted data
# to pick the pymca data (in same folder, in images subfolder) 
# for correlations. it should be the highest energy done, within the chemical
# imaging dataset. if another dataset, script needs to be changed

# ATTENUATION (values): multiply by tx, divide by attenuation,
# this is when not using 15 kev map. must be in 
# same order as els_to_use variable
atten_on = 0  # 1 if you want to correct attenuation, 0 if only raw values
num_scal = 1
allign_sel = 0  # 1 if you want to allign the images, otherwise 0
# if no attenuation, enter the value you want to scale the 
# numerator if there is an external calibration for example


''' 
code
'''

## reading part
print('###########')
print("LET'S GO!!!")
print('###########')

refs_red_ls = glob.glob(os.path.join(references, 'red', '*.nor'))
refs_ox_ls = glob.glob(os.path.join(references, 'ox', '*.nor'))
redox = ['red', 'ox']
ref_list = refs_red_ls + refs_ox_ls
red_ls = range(0, len(refs_red_ls))
ox_ls = range(len(refs_red_ls), len(ref_list))
couples = [(red, ox) for red in red_ls for ox in ox_ls]
len_red = len(refs_red_ls)

alldirs = [x[0] for x in os.walk(path) if 'extracted' in x[0]]
filelist = [y for ls in [glob.glob(os.path.join(sub, '*.h5')) \
                         for sub in alldirs] for y in ls]

samples = [os.path.join(path, x) for x in next(os.walk(path))[1]]

# if pymca_set:
#     samples = [s for s in samples if 'pymca' in os.path.basename(s)]

gridtest = [X_name, Y_name]
samp_count = 0

els = {'alg_el': alg_el, 'r_El1': r_El1, 'r_El2': r_El2, 'r_El3': r_El3}
indexes = {'X_name': X_name, 'Y_name': Y_name}

for sample in samples:
    energies = []
    array = {}
    for folder in next(os.walk(sample))[1]:
        fullpath = os.path.join(sample, folder)
        energy = float(folder)
        energies.append(energy)
        for file in glob.glob(fullpath, '*.tif'):
            el = os.path.basename(file).split('.')[0]
            array[energy][el] = imageio.imread(file)

    text = ''
    if det_resum:
        els = {el: ''.join(('resum_', val)) for el, val in els.items()}
    else:
        els = {el: ''.join((val, '_sum_cps')) for el, val in els.items()}
    alg_el_name, El1_name, El2_name, El3_name = els.values()
    LCF_name = El1_name
    dirname_out = os.path.join(os.path.dirname(sample), 'LCF_out')
    samp_name = os.path.basename(sample).split('.')[0] + cus_nm
    print('#*#*#*#*#*#*#*#*#*#*#*#*#')
    print('running sample {}'.format(samp_name))

    # %%
    # this is to check if the dataset is actually a chemical redox dataset,
    # and usually if there is less than 3 energies it is not
    if len(energies) < 3:
        print('not a valid chemical mapping dataset')
        continue

    LCF_energies = [en for en in energies if out_chem not in en]
    en_values = [float(energy) for energy in LCF_energies]
    allign = {nen: array[nen][els['alg_el']] for nen in energies}
    '''
    allignment part
    the idea behind this routine is to allign the images to account for 
    energy-dependent drift by looking at features in a fluorescence map 
    that do not change as a function of incident energy. it requires the
    files to all have the same shape. it only fixes translation in X and Y. 
    there should actually only be changes in the Y direction, but there is 
    also probably changes in the beam spot profile which we make the 
    assumption are negligeable (sub-pixel)
    '''
    overall_mismatch = np.empty(allign[energies[0]].shape)
    shift = {}
    for nen, energy in enumerate(energies):
        reference_al = allign[energies[all_en]]
        shift[energy], error, diffphase = \
            phase_cross_correlation(reference_al, \
                                    allign[energy], \
                                    upsample_factor=100)
        diagnostic = \
            sp.ndimage.shift(allign[energy], shift[energy]) - reference_al
        try:
            overall_mismatch = overall_mismatch + np.abs(diagnostic)
        except:
            if allign[energies[nen]].shape != reference_al.shape:
                print('the images are not the same size')
                print('')
        for key in array[energy]:
            # this algorithm below makes the shift based on the calculated translation.
            if allign_sel:
                array[energy][key] = sp.ndimage.shift(array[energy][key], shift[energy], order=1)
            else:
                text = f'_NoAllignment'
            if moving_avg:
                nu_key = f'{key}mv_avg{str(m_b)}x{str(m_b)}'
                array[energy][nu_key] = array[energy].pop(key)

                array[energy][nu_key] = mul.mov_avg(array[energy][nu_key], movavg_shape, m_b)
                dirname = os.path.join(dir_out_r, samp_name + f'_mvavg_{str(m_b)}x{str(m_b)}_{text}')
                key = nu_key
            else:
                dirname = os.path.join(dir_out_r, samp_name + f'{text}')
            # energy_nm = os.path.basename(energy)
            out_all = mul.safe_mkdir(os.path.join(dirname, 'allign'), True)
            img1 = Image.fromarray(array[energy][key])
            # check if it actually works here
            suff = '_alligned.tif'
            img_name = f'{out_all}/{key}_{energy}_{suff}'
            img1.save(img_name)
    diag = Image.fromarray(overall_mismatch)
    diag_name = os.path.join(out_all, '{}_all_diag.tif'.format(els['alg_el']))
    diag.save(diag_name)
    shape = np.shape(overall_mismatch)
    # reference reading
    print('\t######################')
    print('\t# reading references #')
    print('\t######################')

    ref_head = [os.path.basename(ref)[0:10] for ref in ref_list]
    N_ref = len(ref_list)
    nor_vals = np.zeros((len(LCF_energies), len(ref_list)))
    energy_refs = [float(r_energy) for r_energy in LCF_energies]
    for index, reference in enumerate(ref_list):
        ref = np.loadtxt(reference)
        for en, r_energy in enumerate(energy_refs):
            value = ref[np.where(np.around(ref[:, 0], decimals=0) == r_energy), 1]
            nearest = ref[np.abs(ref[:, 0] - r_energy).argmin(), 1]
            if value.size == 1:
                nor_vals[en, index] = value.item()
            elif value.size == 0:
                nor_vals[en, index] = nearest
            else:
                nor_vals[en, index] = np.mean(value)
    nor_vals = nor_vals / nor_vals[
        -1]  # this is make the refs similar to the data (last energy is 1 in the data)--> "normalizing" the references also in case they were not already
    try:
        nor_vals / 1
    except:
        print('the reference list is empty')
    '''
    normalization part -- CAN STILL BE IMPROVED MAYBE?
    this part is to attempt to make a normalization of the maps. it will 
    take the average for the highest energy map, and the minimum for the 
    lowest energy map to normalize the data. this may be not the best 
    solution, but I did not find something better for the moment
    '''
    # here below divide by the attenuation of El1 by the filter

    hi_El = array[LCF_energies[-1]][[k for k in array[LCF_energies[-1]] \
                                     if LCF_name in k][0]]
    lo_El = array[LCF_energies[0]][[k for k in array[LCF_energies[0]] \
                                    if LCF_name in k][0]]
    El1_map = array[LCF_energies[-1]][[k for k in array[LCF_energies[-1]] \
                                       if els['r_El1'] in k][0]]
    El2_map = array[LCF_energies[0]][[k for k in array[LCF_energies[0]] \
                                      if els['r_El2'] in k][0]]
    El3_map = array[LCF_energies[0]][[k for k in array[LCF_energies[0]] \
                                      if els['r_El3'] in k][0]]  # takes highest energy map


    correct_vals = [El3_map, El1_map, El2_map]
    names = ['El3_atten', 'El1_atten', 'El2_atten']
    dir_atten = mul.safe_mkdir(os.path.join(dirname_out, 'attenuation_corrected'), True)
    for ind_att, mp in enumerate(correct_vals):
        Image.fromarray(mp).save(os.path.join(dir_atten, names[ind_att] + '.tif'))
    if El1_masking:
        El1_mask = np.copy(hi_El)
        El1_mask[El1_mask <= El1_threshold] = 0
        El1_mask[El1_mask > El1_threshold] = 1
    else:
        El1_mask = np.ones(np.shape(hi_El))

    sel_en = LCF_energies[-1]

    El_sel = [k for k in array[sel_en] if LCF_name in k][0]
    Den_sel = [k for k in array[sel_en] if El2_name in k][0]
    Den_raw = np.ma.masked_less(array[sel_en][Den_sel] * El1_mask, 1)
    El_raw = np.ma.masked_less(array[sel_en][El_sel] * El1_mask, 1)
    hi_El_msk = np.ma.masked_array(hi_El, Den_raw.mask)
    lo_El_msk = np.ma.masked_array(lo_El, Den_raw.mask)
    El1_int_scal = np.ma.fix_invalid(hi_El / np.nanmax(hi_El), fill_value=0)
    El2_int_scal = np.ma.fix_invalid(Den_raw / np.nanmax(Den_raw), fill_value=0)

    ### all changes for hdf5 reading are up to here, below everything should be fine###
    # mask for all data
    '''
    here is a mask to select only the pixels with minimum counts higher 
    than background for El1 it takes the mean value of the highest energy map 
    and uses this as a threshold for creating a num_mask
    '''
    if mask:
        if edge:
            mask_Num = np.greater_equal(hi_El, lo_El * edge_ratio)
        else:
            mask_Num = np.greater_equal(hi_El, np.mean(lo_El))
        Image.fromarray(mask_Num.astype(int)).save(os.path.join(dirname_out, \
                                                                'num_mask.tif'))
    else:
        mask_Num = np.ones(np.shape(hi_El))

    # population of the arrays with normalized data for the LCF
    el_norm = np.zeros((len(LCF_energies), shape[0], shape[1]))
    for Eind, energy in enumerate(LCF_energies):
        sel_El = [k for k in array[energy] if LCF_name in k][0]
        data = (((array[energy][sel_El] * El1_mask) - lo_El) / (hi_El - lo_El)) * El1_mask
        el_norm[Eind, :, :] = np.ma.fix_invalid(data, fill_value=0)

    ''' 
    LCF selection/rating based on residuals after testing all references on
    each pixel:
    1. evaluate the residuals for all references for each pixel
    2. for each pixel, select only the value with lowest residual and assign 
    it in the map
    3. save a list with the reference and the index
    4. evaluate the 2 component fit that works the best, with the lowest residual
    5. select in a final map whether the 1 component is better or the 2 component
    '''

    # ROI selection
    if ROI_opt:
        ROI, roi_masks, roi_coord = \
            mul.ROIs(sel_roi, el_norm, hi_El, Den_raw, dirname_out, image_scl)
        outfilename = 'roi.csv'
    else:
        ROI, roi_masks = {}, {}
        ROI['fullmap'] = el_norm
        roi_masks['fullmap'] = [0, np.shape(el_norm)[1], 0, np.shape(el_norm)[2]]
        outfilename = 'full.csv'

        print('\n\t# saving LCF images #')
        print('\t#################')

    # %%
    valid_roi = [roi for roi in ROI if ROI[roi].size]  # this is because the quit does not always work
    LCF_dict = {}
    LCF_avg = {}
    allpix = {}
    spectrum = {}

    for roi_ind, roi in enumerate(valid_roi):
        print('\t')
        print('\t# running roi {} of {} #'.format(roi_ind, len(valid_roi) + 1))
        print('\t#################')
        dir_out = mul.safe_mkdir(os.path.join(dirname_out, roi), True)
        x1, x2, y1, y2 = roi_masks[roi]

        hi_el_roi = hi_El[x1:x2, y1:y2]
        Den_raw_roi = Den_raw[x1:x2, y1:y2]
        El_raw_roi = El_raw[x1:x2, y1:y2]
        mask_Num_roi = mask_Num[x1:x2, y1:y2]
        El1_mask_roi = El1_mask[x1:x2, y1:y2]
        El3_roi = El3_map[x1:x2, y1:y2]
        El1_roi = El1_map[x1:x2, y1:y2]
        El2_roi = El2_map[x1:x2, y1:y2]
        El1_int_scal_roi = El1_int_scal[x1:x2, y1:y2]
        El2_int_scal_roi = El2_int_scal[x1:x2, y1:y2]
        # this part saves all the pixel results as images
        shape_roi = (x2 - x1, y2 - y1)
        G = np.zeros((shape_roi[0], shape_roi[1]))
        flatt_1 = np.ma.masked_less_equal(El1_roi * El1_mask_roi, 0).flatten()
        flatt_2 = np.ma.masked_less_equal(El2_roi * El1_mask_roi, 0).flatten()
        flatt_3 = np.ma.masked_less_equal(El3_roi * El1_mask_roi, 0).flatten()
        allpix[roi] = pd.DataFrame({'{}_roi'.format(El1_name): flatt_1,
                                    '{}_roi'.format(El2_name): flatt_2,
                                    '{}/{}_roi'.format(El1_name, El2_name): flatt_1 / flatt_2,
                                    '{}_roi'.format(El3_name): flatt_3})
        roi_el_norm = el_norm[:, x1:x2, y1:y2]
        roi_spectrum = np.mean(roi_el_norm.reshape(len(LCF_energies), -1), 1)
        spectrum[roi] = pd.Series(roi_spectrum, index=en_values)

        if roi_lcf:
            print('\n\t# running pixel LCF on ROI %s #' % roi)
            print('\t#################')

            shape_roi = (x2 - x1, y2 - y1)  # this could be 1 less than what it should be
            if sum(shape_roi) <= 2:
                continue
            LCF_sel_header = ['2comp_red', '2comp_ox', '1comp']

            print('\n\t# saving ROI %s images #' % roi)
            print('\t#################')

            LCF_1comp, LCF_1comp_res, LCF_1comp_val = \
                mul.pixelLCF(ROI[roi], nor_vals)
            LCF_2comp, LCF_2comp_res, LCF_2comp_val = \
                mul.pixel_2compLCF(ROI[roi], nor_vals, couples)
            LCF_select, LCF_sel_oxstate = \
                mul.LCF_sel(len_red, LCF_2comp, LCF_2comp_val,
                            LCF_2comp_res, LCF_1comp, LCF_1comp_res, crit)

            nm = roi + '1comp_rating.tif'
            Image.fromarray(LCF_1comp).save(os.path.join(dir_out, nm))
            nm = roi + '1comp_NSSR.tif'
            Image.fromarray(LCF_1comp_res).save(os.path.join(dir_out, nm))

            for couple in range(2):
                nm = roi + '_{}_rating.tif'.format(redox[couple])
                Image.fromarray(LCF_2comp[couple]).save(os.path.join(dir_out, nm))
            nm = roi + '_NSSR.tif'
            Image.fromarray(LCF_2comp_res).save(os.path.join(dir_out, nm))
            for num, comp in enumerate(LCF_2comp_val):
                nm = roi + '_2comp_{}_val.tif'.format(redox[num])
                Image.fromarray(LCF_2comp_val[num]).save(os.path.join(dir_out, nm))
                nm = roi + '_2comp_norm_{}_val.tif'.format(redox[num])
                Image.fromarray(LCF_sel_oxstate[num]).save(os.path.join(dir_out, nm))
            for ind, index in enumerate(LCF_select):
                nm = roi + 'LCF_sel_{}_rating.tif'.format(LCF_sel_header[ind])
                Image.fromarray(index).save(os.path.join(dir_out, nm))
            refs_corr = ['no LCF'] + ref_head
            refs = pd.DataFrame([[refs_corr[b] for b in a] \
                                 for a in LCF_2comp.reshape(2, -1).T.astype(int)],
                                columns=['LCF_ref_ox', 'LCF_ref_red'])
            allpix[roi][f'{El1_name}_Ox'] = \
                np.ma.masked_invalid(LCF_sel_oxstate[1]).flatten()
            allpix[roi] = allpix[roi].join(refs)

            # make the correlations of El1 and El2 with oxidation state
            cleaned = allpix[roi].dropna()
            plt.rcParams.update({'font.size': 18})
            points = plt.scatter(cleaned[f'{El2_name}_roi'],
                                 cleaned[f'{El1_name}_roi'],
                                 c=cleaned[f'{El1_name}_Ox'],
                                 vmin=0, vmax=1, s=10, cmap='Spectral_r')
            m, b = np.polyfit(cleaned[f'{El2_name}_roi'],
                              cleaned[f'{El1_name}_roi'], 1)
            cbar = plt.colorbar(points)
            cbar.set_label('{}(Ox) percent'.format(El1_name))
            plt.xlim(cleaned[f'{El2_name}_roi'].min() - 10,
                     cleaned[f'{El2_name}_roi'].max() + 10)
            plt.ylim(cleaned[f'{El1_name}_roi'].min() - 10,
                     cleaned[f'{El1_name}_roi'].max() + 10)
            sb_plt = sns.regplot(f'{El2_name}_roi', f'{El1_name}_roi',
                                 data=cleaned, scatter=False, color='.1',
                                 line_kws={'color': 'red', 'linestyle': 'dashed'})
            sb_plt.set(ylabel=f'{El1_name} raw counts', xlabel= \
                f'{El2_name} raw counts')
            lgd_txt = \
                'regr. vals\n{}/{} = {}\n{} background = {}' \
                    .format(El1_name, El2_name, round(m, 3), El1_name, round(b, 3))
            plt.text(cleaned[f'{El2_name}_roi'].min() - 10,
                     cleaned[f'{El1_name}_roi'].max(),
                     lgd_txt, alpha=.75)
            nm = f'{roi}_{El1_name}_{El2_name}_correlations.svg'
            plt.savefig(os.path.join(dir_out, nm))
            plt.close()

        third_el = [k for k in array[sel_en] if El3_name in k][0]
        third_el2 = array[sel_en][third_el]

        RGB_El2 = mul.RGB_fun(El2_roi, G, El1_roi)
        nm = roi + f'_{El2_name}R_nanG_{El1_name}B_scaleint.tif'
        cv2.imwrite(os.path.join(dir_out, nm), RGB_El2)

    # %%
    # save all the text files with elaborated results
    df_dict = {str(index + 1): ref for index, ref in enumerate(ref_head)}
    tmp = pd.DataFrame(data=df_dict, index=['reference'])
    tmp = tmp.transpose()

    # this is raw if wanting to look at the pseudo xanes
    roi_option = ['fullmap', 'rois']
    im_lcf = ['no_im_lcf', 'im_lcf']
    nm = 'output_{}_{}.xlsx'.format(roi_option[ROI_opt], im_lcf[roi_lcf])
    with pd.ExcelWriter(os.path.join(dirname_out, nm)) as writer:
        tmp.to_excel(writer, sheet_name='ref_list')
        for roi in valid_roi:
            allpix[roi].to_excel(writer, sheet_name=f'{roi}_allpixels')
            spectrum[roi].to_excel(writer, sheet_name=f'{roi}_avg_spectrum')
    samp_count += 1
