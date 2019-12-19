# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 10:56:21 2018

@author: marafatto_f

PROCEDURE / Algorithm

This file will read the h5 files and save as tiff files.

It will create:
- root directory with samplename + energy:
        - subdir 'full' with the full spectra (1 tiff per channel)
        - subdir 'ROIS' with the roi data (1 tiff per image)

prerequisites (not implemented yet but could be):
    multiprocessing module:
        split the tasks in 4 or 8 depending on the cores available:
            - variable to count the cores
            - way to split the tasks of reading the data (run the whole script in parallel depnding on variable count_cores)
                - one option could be to make a function that has the whole script/loop inside?
steps to do:
    1. read hdf5 nexus file:
        - read size (trx and try)
        - read readback values
        - read data
        - read snake option
    2. interpolate/reshape the data if snake
        - griddata should be enough, use the *1000 option
        - shift every second line left and cut last pixel (cut first value every uneven line)
    3. save the data:
        - to save in folder all of one sample we need to figure out that samples are the same. loop over similarity (make a temp variable from filelist)

        - tiff option: folders of sample run, subfolders with energies and sub-subfolders with rois and fullspectrum

To implement:

figure out the interpolation part for images that are not the same size:
    read t3x and t3y values of all images (make a grid with length). take the biggest and resize others to this one if not equal:
	- save t3x and t3y as a text file in root

    the following is to be done in the following script:
   - list shapes images
	- take max shape as reference
	- if not = max shape --> resize to max --> data = interpolate (values with original x, y, grid to max_shape x, y)
	- else, do normal --> data = data

	export of images:
	- rois
	- full (tiff stacks?)
	- txt with original x and y for each energy (original grid), new grid if made (interpolated grid)

"""
import h5py
import os, glob, re
import numpy as np
from PIL import Image
import skimage.transform as tf
import pandas as pd
import itertools

'''
USER INPUT
'''
path = [os.getcwd() , 'F:\\DLS_2017\\Diamond_I18_1709_sp15773_all']
path_choice = 1 # select 1 if you want to specify the path manually, 0 to run in the current dir
dir_out_root = 'N:\\DLS_out_2017' # given that the amount of data is often large, you may want to export out to another disk
rebin = 0 #0 if no, 1 if you want to rebin. select the scale factor below
scale = 0.5
en_len = 5 # if the filename contains the energy uncomment, and if it is keV put 5, otherwise 4
reduced_list = 3 # 0 if all, 1 if only the ones in the diagnostic that have 0, 2 if all those not in diagnostic, 3 if only the ones in "selection"
export_full = 0 # 1 if you want to also export the full spectrum
selection = [range(95579,95583)]#,range(124340,124343)] # remember, python does not include the last value in a range
data_src = 2017

''' FUNCTIONS '''
# this function below will try to identify whether the data is collected in snake mode, and if yes
# whether it is the odd or even lines that need shifting. it will then shift the lines and "clip" 1 column
# of pixels and the last row.
def cipher_find(x):
    out = int(re.search(ciphers,os.path.basename(x)).group(0))
    return out

def snake_fix(array): # array is the array you want to fix, X_r is the real X read from the encoder.
    try:
        snake_chk = np.diff(pandafile[trigger][:,:,0,0]) # this is important for the snake correction, reads the counter
    except:
        print('there was no pandafile, no snake fix performed')
        snake_chk = [1]
    #tmp = [index for index,i in enumerate(snake_chk[:,0]) if i < 0]
    if -1 in snake_chk:
        #array_ch = np.empty(np.shape(array[:,2:]))
        array_ch = np.empty(np.shape(snake_chk))
        for ind,row in enumerate(snake_chk):
            try:
                if 1 in row:
                    array_ch[ind] = array[ind,0:-1]
                elif -1 in row:
                    array_ch[ind] = array[ind,1:]
            except:
                # print 'length issue with file '+orig_name
                continue
    else: # there is no snake
        array_ch = array
    return array_ch

''' END FUNCTIONS'''

'''
CODE PART
'''
files_bad = []
path = path[path_choice]

sumfact = 2/0.5 # this is so the rebinning takes a sum
# for the processed file
root_group = '/entry/auxiliary/0-XRF Elemental Maps from ROIs/'
full = '/entry/result/data'
if data_src == 2018:
    Xval ='/entry/result/t3x'
    Yval = '/entry/result/t3y'
    ciphers = '\d{6}'
elif data_src == 2017:
    Xval ='/entry/result/table_x'
    Yval = '/entry/result/table_y'
    ciphers = '\d{5}'
# for the snake correction, PANDABOX file
trigger = 'entry/NDAttributes/NDArrayUniqueId'

# for the main nxs file without PANDABOX
energy = '/entry/instrument/DCM/energy'
I0_nam = '/entry/I0/data/'
name = '/entry/sample/name'

dirs = ['full','rois']

filelist = glob.glob(os.path.join(path,'processed','*.nxs'))

selection = list(itertools.chain(*selection))
selected_list = [x for x in filelist if cipher_find(x) in selection]
files_all,files_ok,files_done, red_list = [], [], [], []
if reduced_list == 1 or reduced_list == 2:
    files_scanned = pd.read_csv(os.path.join(dir_out_root,'diagnostic.txt'), sep='\t')
    columns = files_scanned.columns.values
    files_ok = [x for y,x in enumerate(files_scanned[columns[0]]) if files_scanned[columns[1]][y] == 0]
    files_done = [os.path.basename(x) for x in filelist if os.path.basename(x) in files_scanned[columns[0]].any()]
elif reduced_list == 3:
    red_list = [os.path.basename(x) for x in filelist if x not in selected_list]

reduced_sel = [files_all,files_ok,files_done,red_list]
    # files_to_scan = [124328]
filelist = [item for item in filelist if os.path.basename(item) not in reduced_sel[reduced_list]]

rawdir = [[os.path.join(root,name1) for name1 in files if '.h5' in name1] for root,dirs1,files in os.walk(path)]
rawdir = [item for sublist in rawdir for item in sublist]
rawentry = glob.glob(os.path.join(path,'*.nxs'))

list_names = []

diagnostic = {}
diagnostic_head = ['rawfile/pandafile absence','I0 missing','missing_rois','missing_full']

#%%

for filenum,filei in enumerate(filelist):
    filein = filelist[filenum]
    count = 1
    filenam = os.path.basename(filein)
    filenam_noext = filenam[0:filenam.find('.')]
    file_id = re.search(ciphers,filenam).group(0)
    print('processing file '+ filenam)
    diagn_file = str(filenum)+' '+filenam
    diagnostic[diagn_file] = []
    try:
        sel_file1 = [item for item in rawentry if file_id in item and 'PANDABOX' not in item][0]
        rawfile = h5py.File(sel_file1,'r')
        sel_file2 = [item for item in rawdir if file_id in item and 'PANDABOX' in item][0]
        pandafile = h5py.File(sel_file2,'r')
    except IOError:
        print("skipping file %s, rawfile and/or pandafile not found/corrupt" % filenam_noext)
        diagnostic[diagn_file].append(1)
        continue
    diagnostic[diagn_file].append(0)
    hdfile = h5py.File(filein,'r') # read the Xpress3a processed file
    orig_name = filenam_noext
    raw_name = rawfile[name][()]
    match = re.search('\d{5}',raw_name)
    if match and pandafile:
        root_name = raw_name.replace(match.group(0),'')
    elif match and not pandafile:
        root_name = raw_name.replace(match.group(0),'nopanda')
    elif not raw_name:
        root_name = orig_name
    else:
        root_name = raw_name
    print('      aka %s with newname %s' % (raw_name,root_name))
    root_energy = np.around(rawfile[energy][()],decimals=1)
    energy_name = str(int(root_energy))
    while os.path.exists(os.path.join(dir_out_root,root_name,energy_name,dirs[0])):
        energy_name = '%s_%d' %(int(root_energy),count)
        count += 1
    print('processing file %s at energy %s ' % (root_name,energy_name))
    grid = (hdfile[Xval][()],hdfile[Yval][()])
    for direc in dirs:
        try:
            os.stat(os.path.join(dir_out_root,root_name,energy_name,direc))
        except:
            os.makedirs(os.path.join(dir_out_root,root_name,energy_name,direc))
    # X,Y = hdfile[Xval].value,hdfile[Yval].value
    try:
        rois = list(hdfile[root_group].keys())
        diagnostic[diagn_file].append(0)
    except:
        print(orig_name+' empty dataset rois')
        diagnostic[diagn_file].append(1)
    #step = abs(np.around(np.median(np.diff(X_r)),decimals=3))
    #file_data = np.empty((len(filelist),len(rois),len(X),len(Y)))
    list_names.append(root_name)
    try:
        I0 = rawfile[I0_nam][:,:,0,0]
        diagnostic[diagn_file].append(0)
    except:
        print('could not read the I0, rawfile missing?')
        diagnostic[diagn_file].append(1)
        I0 = 1
    try:
        Fdata = hdfile[full][()]
        diagnostic[diagn_file].append(0)
    except:
        print(orig_name + ' empty dataset full')
        diagnostic[diagn_file].append(1)
    if hdfile[root_group+rois[0]+'/data'][()].shape != I0.shape:
        print("there's something funny with the array shapes, incomplete map?")
        files_bad.append(filein)
        continue
    for ind,el in enumerate(rois):
        data = hdfile[root_group+el+'/data'][()]
        if rebin == 0:
            data_fix = snake_fix(data)/snake_fix(I0)
        else:
            data_fix = tf.rescale(snake_fix(data/I0),scale)*sumfact
            data = tf.rescale((data/I0),scale)*sumfact
        img = Image.fromarray(data_fix)
        img1 = Image.fromarray(data)
        tmp_dir = os.path.join(dir_out_root,root_name,energy_name,'rois')
        img_name = os.path.join(tmp_dir,root_name+'_%s_%ieV.tif' % (el,root_energy))
        img1_name = os.path.join(tmp_dir,root_name+'_orig_%s_%ieV.tif' % (el,root_energy))
        img.save(img_name)
        img1.save(img1_name)
    if export_full:
        Fdatafix_shp = np.shape(data_fix)+(4096,)
        # Fdata_shp = Fdata_shp + (np.shape(Fdata)[2],)
        data_fix = np.zeros(Fdatafix_shp)
        Fdata_out = []
        tmp_dir = os.path.join(dir_out_root,root_name,energy_name,'full')
        full_out = h5py.File(os.path.join(tmp_dir,root_name+'_%01.deV.h5' % (root_energy)),'w')
        for ch in range(np.shape(Fdata)[2]):
            if ch%1000 == 0:
                status = int(ch/float(len(range(np.shape(Fdata)[2]))))*100
                print("processing full spectrum %.0d percent" % status)
            if rebin == 0:
                data_fix[:,:,ch] = snake_fix(Fdata[:,:,ch])/snake_fix(I0)
                Fdata_out.append(Fdata[:,:,ch])
            else:
                data_fix[:,:,ch] = tf.rescale(snake_fix(Fdata[:,:,ch]/I0),scale)*sumfact
                Fdata_out.append(tf.rescale(Fdata[:,:,ch]/I0,scale)*sumfact)
        full_out.create_dataset('full_spectrum',data=np.array(Fdata_out))
        full_out.create_dataset('full_spectrum_snakefix',data=data_fix)
        full_out.close()
    #%%
    for index,el in enumerate(grid):
        with open(os.path.join(dir_out_root, root_name,energy_name,'grid_%s.txt' % index), 'w') as f:
            np.savetxt(f,el)
#%%
    try:
        hdfile.close()
        rawfile.close()
        pandafile.close()
#%%
    except:
        continue
    if reduced_list:
        tmp = pd.DataFrame(data=diagnostic,index=diagnostic_head).transpose()
        tmp.to_csv(os.path.join(dir_out_root,'diagnostic_reduced.txt'),sep='\t')
    else:
        tmp = pd.DataFrame(data=diagnostic,index=diagnostic_head).transpose()
        tmp.to_csv(os.path.join(dir_out_root,'diagnostic.txt'),sep='\t')

np.savetxt(os.path.join(dir_out_root,'bad_files.txt'),np.asarray(files_bad))
