#!/home/mbmbertolero/anaconda2/bin/python

import brain_graphs
import matlab
import matlab.engine
import pandas as pd
import numpy.linalg as npl
import numpy as np
import os
import pickle
import sys
import nibabel as nib
from sklearn.metrics import precision_recall_fscore_support
import multiprocessing
from multiprocessing import Pool
from igraph import VertexClustering
from scipy.stats import spearmanr, pearsonr,linregress
from scipy.spatial import distance
from scipy import stats, linalg
from scipy.spatial.distance import pdist
import scipy.io
import scipy
# import statsmodels.api as sm
import copy
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.svm import SVR
import statsmodels.api as sm
import time
from subprocess import Popen, PIPE
from matplotlib import rc
# rc('text', usetex = False)
import glob
from itertools import combinations
import operator
import seaborn as sns
import matplotlib.pylab as plt
sns.plt = plt
import matplotlib as mpl
from matplotlib import patches
import math
from pandas_plink import read_plink
from statsmodels.stats.multitest import fdrcorrection
plt.rcParams['pdf.fonttype'] = 42
# mpl.font_manager.FontProperties(family='sans-serif',style='oblique',fname='/home/mbmbertolero/data/Open Sans/Open Sans-Oblique.ttf')
# # mpl.font_manager.FontProperties(family='sans-serif',style='italic',fname='/home/mbmbertolero/data/Open Sans/Open Sans-Oblique.ttf')
# mpl.font_manager.FontProperties(family='sans-serif',style='normal',fname='/home/mbmbertolero/data/Open Sans/Open Sans.ttf')
plt.rcParams['font.sans-serif'] = "Open Sans"
plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.it'] = 'Open Sans:italic'
plt.rcParams['mathtext.it'] = 'Open Sans:italic'
plt.rcParams['mathtext.cal'] = 'Open Sans'

from subprocess import check_output
tmpdir = check_output(['echo $TMPDIR'],shell=True)
if len(tmpdir.split('\n')[0]) == 0: tmpdir = '/tmp/runtime-mb3152'
os.environ['XDG_RUNTIME_DIR'] = tmpdir.split('\n')[0]

from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
global cortical
global cerebellar
global well_id_2_mni
global well_id_2_idx
global well_id_2_struct
global co_a_matrix
global gene_exp
global random_gene_exp
global co_a
global template
global m
global parcel_x

global features
global measure
global size
global layers
global layers_name

cortical = nib.load('//share/apps/fsl/5.0.9/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz').get_data()
cerebellar = nib.load('//share/apps/fsl/5.0.9/data/atlases/Cerebellum/Cerebellum-MNIflirt-maxprob-thr0-2mm.nii.gz').get_data()

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
	else: raise argparse.ArgumentTypeError('Boolean value expected')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r',action='store',dest='run_type')
parser.add_argument('-m',action='store',dest='matrix',default='fc')
parser.add_argument('-layers',action='store',dest='n_layers',type=int,default=1)
parser.add_argument('-name',action='store',dest='name',type=str,default='participation coefficient')
parser.add_argument('-nodes',action='store',dest='n_nodes',type=int,default=5)
parser.add_argument('-node',action='store',dest='node',type=int,default=0)
parser.add_argument('-n_genes',action='store',dest='n_genes',type=int,default=50)
parser.add_argument('-n',action='store',dest='network')
parser.add_argument('-task',action='store',dest='task',type=int)
parser.add_argument('-s',action='store',dest='subject')
parser.add_argument('-components',action='store',dest='components',default='edges',type=str)
parser.add_argument('-topological',action='store',dest='topological',type=str2bool,default=False)
parser.add_argument('-distance',action='store',dest='distance',type=str2bool,default=True)
parser.add_argument('-prediction',action='store',dest='prediction',type=str,default='genetic')
parser.add_argument('-use_prs',action='store',dest='use_prs',type=str2bool,default=False)
parser.add_argument('-norm',action='store',dest='norm',type=str2bool,default=True)
parser.add_argument('-corr_method',action='store',dest='corr_method',type=str,default='pearsonr')
r = parser.parse_args()
locals().update(r.__dict__)
try:network = int(network)
except:pass
n_genes = int(n_genes)
global ignore_nodes

layers_name = '%s_%s' %(n_layers,n_nodes)
layers = []
for layer in range(n_layers):
	layers.append(n_nodes)
global tasks
tasks = ['WM','RELATIONAL','LANGUAGE','SOCIAL','EMOTION']


"""
Globals for neural network behavior prediction
"""
global subject_pcs
global subject_wmds
global subject_mods
global rest_subject_pcs
global rest_subject_wmds
global rest_subject_mods
global task_perf
global task_matrices
global rest_matrices
global use_matrix
global norm
global corr_method
atlas_path = '/home/mbmbertolero/data/PUBLIC/yeo/fsLR32k/Schaefer2016_400Parcels_17Networks_colors_23_05_16.dlabel.nii'
global nodes_genes
global snp2genes_df

class GraphDist() :
    def __init__(self, size, ax, x=True) :
        self.size = size
        self.ax = ax
        self.x = x

    @property
    def dist_real(self) :
        x0, y0 = self.ax.transAxes.transform((0, 0)) # lower left in pixels
        x1, y1 = self.ax.transAxes.transform((1, 1)) # upper right in pixes
        value = x1 - x0 if self.x else y1 - y0
        return value

    @property
    def dist_abs(self) :
        bounds = self.ax.get_xlim() if self.x else self.ax.get_ylim()
        return bounds[0] - bounds[1]

    @property
    def value(self) :
        return (self.size / self.dist_real) * self.dist_abs

    def __mul__(self, obj) :
        return self.value * obj

def write_cifti(atlas_path,out_path,colors):
	os.system('wb_command -cifti-label-export-table %s 1 temp.txt'%(atlas_path))
	df = pd.read_csv('temp.txt',header=None)
	for i in range(df.shape[0]):
		try:
			d = np.array(df[0][i].split(' ')).astype(int)
		except:
			continue
		real_idx = d[0] -1
		df[0][i] = str(d[0]) + ' ' + str(int(colors[real_idx][0]*255)) + ' ' + str(int(colors[real_idx][1]*255)) + ' ' + str(int(colors[real_idx][2]*255)) + ' ' + str(255)
	df.to_csv('temp.txt',index=False,header=False)
	os.system('wb_command -cifti-label-import %s temp.txt %s.dlabel.nii'%(atlas_path,out_path))

def make_cifti_heatmap(data,cmap="RdBu_r"):
	orig_colors = sns.color_palette(cmap,n_colors=1001)
	norm_data = copy.copy(data)
	if np.nanmin(data) < 0.0: norm_data = norm_data + (np.nanmin(norm_data)*-1)
	elif np.nanmin(data) > 0.0: norm_data = norm_data - (np.nanmin(norm_data))
	norm_data = norm_data / float(np.nanmax(norm_data))
	norm_data = norm_data * 1000
	norm_data = norm_data.astype(int)
	colors = []
	for d in norm_data:
		colors.append(orig_colors[d])
	return colors

def yeo_membership(components=17):
	if components == 7: namedict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	if components == 17: namedict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((200)).astype(str)
	small_membership = np.zeros((200)).astype(str)
	membership_ints = np.zeros((200)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:201]
	for i,n in enumerate(yeo_df):
		membership_ints[i] = int(namedict[n.split('_')[2]])
	return membership_ints

def swap(matrix,membership):
	membership = np.array(membership)
	swap_indices = []
	new_membership = np.zeros(len(membership))
	for i in np.unique(membership):
		for j in np.where(membership == i)[0]:
			swap_indices.append(j)
	return swap_indices

def nan_pearsonr_old(x,y):
	x = np.array(x)
	y = np.array(y)
	isnan = np.sum([x,y],axis=0)
	isnan = np.isnan(isnan) == False
	return pearsonr(x[isnan],y[isnan])

def nan_spearmanr(x,y):
	x,y = np.array(x),np.array(y)
	mask = ~np.logical_or(np.isnan(x), np.isnan(y))
	return spearmanr(x[mask],y[mask])

def nan_pearsonr(x,y):
	x,y = np.array(x),np.array(y)
	mask = ~np.logical_or(np.isnan(x), np.isnan(y))
	return pearsonr(x[mask],y[mask])

def three_d_dist(p1,p2):
	return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)

def real_2_mm(target_image, real_pt):
	aff = target_image.affine
	return nib.affines.apply_affine(npl.inv(aff), real_pt)

def make_well_id_2_mni(subjects=['9861','178238266','178238316','178238373','15697','178238359']):
	"""
	loop through subjects, load thier specific well ids
	well_id_2_idx: assign an index for the well by expression array to each well ID.
	well_id_2_mni: get the locatino of each well in MNI array space, using the Poldrack lab transform (well_id_to_MNI)
	"""
	print 'making well to MNI mapping'
	sys.stdout.flush()
	global well_id_2_mni
	global well_id_2_idx
	global well_id_2_struct
	template = nib.load('//share/apps/fsl/5.0.9/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz')
	well_id_to_mni = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/corrected_mni_coordinates.csv')
	well_id_2_mni = {}
	well_idx = 0
	well_id_2_idx = {}
	well_id_2_struct = {}
	for subject in subjects:
		well_ids = pd.read_csv('/home/mbmbertolero/data/gene_expression/data/%s/SampleAnnot.csv'%(subject))['well_id'].values
		for well_id in well_ids:
			assert well_id not in well_id_2_idx.keys() #make sure there is not some weird duplicate across subjects
			well_id_2_idx[well_id] = well_idx #assign an index for the well by expression array to each well ID.
			well_idx = well_idx + 1 #increase index for next well in loop
			loc=np.where(well_id_to_mni['well_id']==well_id)[0][0] #look up the MNI coordinates in the .csv file for this well
			# get the xyz coordinates into ints
			x,y,z = int(well_id_to_mni.corrected_mni_x[loc]),int(well_id_to_mni.corrected_mni_y[loc]),int(well_id_to_mni.corrected_mni_z[loc])
			mni_brain_loc = real_2_mm(template,[x,y,z]) # go from MNI real space to MNI array space
			x,y,z = int(mni_brain_loc[0]),int(mni_brain_loc[1]),int(mni_brain_loc[2])
			well_id_2_mni[well_id] = [[x,y,z]] # save it to the dictionary
			# we are going to normalize by expression across cortex, ignoring sub-cortical and cerebellar stuff
			# so we find, for each well, where it is.
			if cortical[x,y,z] > 0:
				well_id_2_struct[well_id] = 'cortical'
				continue
			elif cerebellar[x,y,z] > 0:
				well_id_2_struct[well_id] = 'cerebellar'
				continue
			else:
				well_id_2_struct[well_id] = 'sub_cortical'
				continue

make_well_id_2_mni()

# mapping = pd.DataFrame.from_dict(well_id_2_idx,orient='index')  # a mappping of order from well_id to matrix indices, save for lab memebers
# array_mapping = pd.DataFrame.from_dict(well_id_2_mni,orient='index')
# array_mapping.to_csv('volume_2_array_mapping.csv')

def get_genes():
	try:genes= np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')
	except:
		# genes from previous study we want to look at
		gene_df = pd.read_excel('/home/mbmbertolero/data//gene_expression/data/Richiardi_Data_File_S2.xlsx')
		# load the probes, so we can remove some genes from the analysis
		probes = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/9861/Probes.csv')
		genes = []
		for probe in gene_df.probe_id.values:
			genes.append(probes.gene_symbol.values[probes.probe_name.values==probe])
		genes = np.unique(genes)
		genes.sort()
		np.save('/home/mbmbertolero/data//gene_expression/gene_names.npy',genes)
	return genes

def gene_expression_matrix(norm=norm,subjects=['9861','178238266','178238316','178238373','15697','178238359']):
	try:
		well_by_expression_array = np.load('//home/mbmbertolero/data/gene_expression/data/ge_exp_%s.npy'%(norm))
	except:
		print 'making gene expression matrix'
		sys.stdout.flush()
		genes = np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')
		# we want to make numpy array, where d1 is the well (brain location) and d2 is the expression of all genes
		well_by_expression_array = np.zeros((len(well_id_2_idx),len(genes)))

		for subject in subjects:
			#load the wells, so we know where in the brain the expression is
			wells = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/SampleAnnot.csv'%(subject))['well_id'].values
			# load the probes
			probes = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/Probes.csv'%(subject))
			#load the full gene expression by well matrix; this contains many genes we don't want to look at, as well as mutliple expression values for each gene
			full_expression = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/MicroarrayExpression.csv'%(subject),header=None,index_col=0)
			# i assume allen put these in the same order, but let's make sure
			assert (full_expression.index.values == probes.probe_id.values).all()
			full_expression = np.array(full_expression)
			# use the previous arrays in well_id_2_struct to make a boolean array for indexing
			# since each subject has a different number of and location of wells, this is unique for each subjects
			cortical_a = np.zeros((len(wells))).astype(bool)
			cerebellar_a = np.zeros((len(wells))).astype(bool)
			sub_cortical_a = np.zeros((len(wells))).astype(bool)
			for idx,well in enumerate(wells):
				if well_id_2_struct[well] == 'cortical':
					cortical_a[idx] = True
					continue
				elif well_id_2_struct[well] == 'cerebellar':
					cerebellar_a[idx] = True
					continue
				else:
					sub_cortical_a[idx] = True
					continue
			# final expression array, where we take the mean from each gene we care about from full_expression
			expression = np.zeros((len(np.unique(genes)),len(wells)))
			# looping through genes
			for expression_idx,gene in enumerate(genes):
				# use the probe df to find where in full_expression the gene we are looking for is
				# there are multiple probes for each gene
				e = []
				for idx in np.where(probes.gene_symbol.values==gene)[0]:
					e.append(full_expression[idx])
				e = np.array(e)
				if norm:# we want to normalize by mean expression in cortex
					for idx in range(e.shape[0]): #normalize by each probe for that gene's expression across cortex
						e[idx] = e[idx] - np.mean(e[idx][cortical_a])
				expression[expression_idx] = np.mean(e,axis=0) # get the mean across probes for this gene
			#turn into a well by gene expression matrix
			expression = expression.transpose()
			for idx,well in enumerate(wells): #since we saved the well_id_2_idx, we can put them in the matrix in the correct order
				well_by_expression_array[well_id_2_idx[well]] = expression[idx]
		np.save('///home/mbmbertolero/data/gene_expression/data/ge_exp_%s.npy'%(norm),well_by_expression_array)
	return well_by_expression_array

def wells_to_regions(norm=norm):
	print 'making well to region mapping'
	sys.stdout.flush()
	global gene_exp
	try:
		gene_exp = np.load('//home/mbmbertolero/data//gene_expression/data/yeo_400_gene_exp_%s.npy'%(norm))
	except:
		template = nib.load('//home/mbmbertolero/data//gene_expression/data/yeo_400.nii.gz').get_data()
		m = np.zeros((int(np.max(template)),gene_exp.shape[1]))
		n_wells = 0
		for parcel in np.arange(np.max(template)):
			wells = []
			for well in well_id_2_mni.keys():
				if template[well_id_2_mni[well][0],well_id_2_mni[well][1],well_id_2_mni[well][2]] == parcel + 1:
					wells.append(gene_exp[well_id_2_idx[well]])
					n_wells = n_wells + 1
			m[int(parcel)] = np.nanmean(wells,axis=0)
		gene_exp = m
		np.save('//home/mbmbertolero/data//gene_expression/data/yeo_400_gene_exp_%s.npy'%(norm),gene_exp)
		print n_wells

def wells_to_brain():
	sys.stdout.flush()
	global gene_exp
	template = nib.load('//home/mbmbertolero/data//gene_expression/data/yeo_400.nii.gz')
	m = np.zeros((91,109,91))
	for idx,well in enumerate(well_id_2_mni.keys()):
		x,y,z = well_id_2_mni[well][0]
		m[x,y,z] = np.mean(gene_exp[well_id_2_idx[well]])
	brain = nib.Nifti1Image(m,template.affine)
	nib.save(brain,'/home/mbmbertolero/data//gene_expression/data/mean_exp.nii')

def atlas_distance():
	try: distance_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/distance.npy')
	except:
		parcel = nib.load('//home/mbmbertolero/data//gene_expression/data/yeo_400.nii.gz').get_data()
		distance_matrix = np.zeros((int(np.max(parcel)),int(np.max(parcel))))
		for i,j in combinations(range(np.max(parcel)),2):
			print i,j
			r = three_d_dist(np.mean(np.argwhere(parcel==i+1),axis=0),np.mean(np.argwhere(parcel==j+1),axis=0))
			distance_matrix[i,j] = r
			distance_matrix[j,i] = r
		np.save('/home/mbmbertolero/data/gene_expression/results/distance.npy',distance_matrix)
	return distance_matrix

def functional_connectivity(topological=False,distance=True,network=None):
	reduce_dict = {'VisCent':'Visual','VisPeri':'Visual','SomMotA':'Motor','SomMotB':'Motor','DorsAttnA':'Dorsal Attention','DorsAttnB':'Dorsal Attention','SalVentAttnA':'Ventral Attention','SalVentAttnB':'Ventral Attention','Limbic':'Limbic','ContA':'Control','ContB':'Control','ContC':'Control','DefaultA':'Default','DefaultB':'Default','DefaultC':'Default','TempPar':'Temporal Parietal'}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]
	try: matrix = np.load('/home/mbmbertolero/data/gene_expression/data/matrices/mean_fc.npy')
	except:
		matrix = []
		matrix_files = glob.glob('/home/mbmbertolero/data/gene_expression/fc_matrices/yeo/*REST*npy*')
		for m in matrix_files:
			if 'random' in m: continue
			m = np.load(m)
			np.fill_diagonal(m,0.0)
			m = np.arctanh(m)
			m[np.isinf(m)] = np.nan
			matrix.append(m.copy())
		matrix = np.nanmean(matrix,axis=0)
		np.save('/home/mbmbertolero/data/gene_expression/data/matrices/mean_fc.npy',matrix)

	if topological == True:
		temp_matrix = matrix.copy()
		for i,j in combinations(range(matrix.shape[0]),2):
			r = nan_pearsonr(matrix[i,:],matrix[j,:])
			temp_matrix[i,j] = r[0]
			temp_matrix[j,i] = r[0]
		matrix = temp_matrix
	if distance:
		distance_matrix = atlas_distance()
		np.fill_diagonal(matrix,np.nan)
		matrix[np.isnan(matrix)==False] = sm.GLM(matrix[np.isnan(matrix)==False],sm.add_constant(distance_matrix[np.isnan(matrix)==False])).fit().resid_response
	if network != None:
		if type(network) == str: matrix[membership!=network] = np.nan
		else: matrix[np.arange(matrix.shape[0])!=network,:] = np.nan
	return matrix

def structural_connectivity(topological=False,distance=True,network=None):
	reduce_dict = {'VisCent':'Visual','VisPeri':'Visual','SomMotA':'Motor','SomMotB':'Motor','DorsAttnA':'Dorsal Attention','DorsAttnB':'Dorsal Attention','SalVentAttnA':'Ventral Attention','SalVentAttnB':'Ventral Attention','Limbic':'Limbic','ContA':'Control','ContB':'Control','ContC':'Control','DefaultA':'Default','DefaultB':'Default','DefaultC':'Default','TempPar':'Temporal Parietal'}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]
	try:
		matrix = np.load('/home/mbmbertolero/data/gene_expression/data/matrices/sc_matrix.npy')
	except:
		matrix = []
		matrix_files = glob.glob('/home/mbmbertolero/data/gene_expression/sc_matrices/*matrix.npy*')
		for m in matrix_files:
			m = np.load(m)
			matrix.append(m.copy())
		matrix = np.nanmean(matrix,axis=0)
		np.save('/home/mbmbertolero/data/gene_expression/data/matrices/sc_matrix.npy',matrix)
	matrix[np.isinf(matrix)] = np.nan
	if topological == True:
		temp_matrix = matrix.copy()
		for i,j in combinations(range(matrix.shape[0]),2):
			r = nan_pearsonr(matrix[i,:],matrix[j,:])
			temp_matrix[i,j] = r[0]
			temp_matrix[j,i] = r[0]
		matrix = temp_matrix
	if distance:
		distance_matrix = atlas_distance()
		np.fill_diagonal(matrix,np.nan)
		matrix[np.isnan(matrix)==False] = sm.GLM(matrix[np.isnan(matrix)==False],sm.add_constant(distance_matrix[np.isnan(matrix)==False])).fit().resid_response
	if network != None:
		if type(network) == str: matrix[membership!=network] = np.nan
		else: matrix[np.arange(matrix.shape[0])!=network,:] = np.nan
	return matrix

def avg_graph_metrics(matrix,topological=False,distance=True):
	# sns.set(context="notebook",font='Open Sans',style='white',font_scale=1.5)
	# reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	# full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	# membership = np.zeros((400)).astype(str)
	# small_membership = np.zeros((400)).astype(int)
	# membership_ints = np.zeros((400)).astype(int)
	# yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	# # yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	# # colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.
	# yeo_colors = pd.read_csv('yeo_colors.txt',header=None,names=['name','r','g','b'],index_col=0)
	# yeo_colors = yeo_colors.sort_values('name')
	# yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.

	# names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	# network_colors = []
	# for i,n in enumerate(yeo_df):
	# 	small_membership[i] = int(reduce_dict[n.split('_')[2]])
	# 	membership[i] = n.split('_')[2]
	# 	membership_ints[i] = int(full_dict[n.split('_')[2]])

	if matrix == 'fc': m = functional_connectivity(topological,distance,None)
	if matrix == 'sc': m = structural_connectivity(topological,distance,None)
	m = m + m.transpose()
	m = np.tril(m,-1)
	m = m + m.transpose()
	pc = []
	wcd = []
	degree = []
	between = []
	for cost in np.linspace(0.01,0.10):
		g = brain_graphs.matrix_to_igraph(m.copy(),cost=cost,mst=True)
		g = brain_graphs.brain_graph(g.community_infomap(edge_weights='weight'))
		pc.append(g.pc)
		wcd.append(g.wmd)
		degree.append(g.community.graph.strength(weights='weight'))
		between.append(g.community.graph.betweenness())
	pc = np.nanmean(pc,axis=0)
	wcd = np.nanmean(wcd,axis=0)
	degree = np.nanmean(degree,axis=0)
	between = np.nanmean(between,axis=0)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_pc_%s.npy'%(matrix,distance),pc)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_strength_%s.npy'%(matrix,distance),degree)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_wcd_%s.npy'%(matrix,distance),wcd)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_between_%s.npy'%(matrix,distance),between)
	"""
	files = glob.glob('/home/mbmbertolero/gene_expression/data/matrices/*%s_matrix*'%(matrix))
	pc = []
	wcd = []
	degree = []
	between = []
	if distance: distance_matrix = atlas_distance()
	for f in files:
		print f
		m = np.load(f)
		m = m + m.transpose()
		m = np.tril(m,-1)
		m = m + m.transpose()
		if distance:
			np.fill_diagonal(m,np.nan)
			m[np.isnan(m)==False] = sm.GLM(m[np.isnan(m)==False],sm.add_constant(distance_matrix[np.isnan(m)==False])).fit().resid_response
		for cost in np.linspace(0.05,0.10,5):
			g = brain_graphs.matrix_to_igraph(m.copy(),cost=cost,mst=True)
			g = brain_graphs.brain_graph(g.community_infomap(edge_weights='weight'))
			pc.append(g.pc)
			wcd.append(g.wmd)
			degree.append(g.community.graph.strength(weights='weight'))
			between.append(g.community.graph.betweenness())
	pc = np.nanmean(pc,axis=0)
	wcd = np.nanmean(wcd,axis=0)
	degree = np.nanmean(degree,axis=0)
	between = np.nanmean(between,axis=0)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_sub_pc_%s.npy'%(matrix,distance),pc)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_sub_strength_%s.npy'%(matrix,distance),degree)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_sub_wcd_%s.npy'%(matrix,distance),wcd)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_sub_between_%s.npy'%(matrix,distance),between)
	"""

def plot_distance():
	"""
	look at the relationship between edge strength and distance
	"""
	f = functional_connectivity(distance=False)
	d = atlas_distance()
	sns.regplot(f[np.isnan(f)==False].flatten(),d[np.isnan(f)==False].flatten(),order=3,scatter_kws={'alpha':.015})
	sns.plt.ylabel('distance')
	sns.plt.xlabel('connectivty')
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/results/fc_distance_3.pdf')
	sns.plt.close()
	sns.regplot(f[np.isnan(f)==False].flatten(),d[np.isnan(f)==False].flatten(),order=2,scatter_kws={'alpha':.015})
	sns.plt.ylabel('distance')
	sns.plt.xlabel('connectivty')
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/results/fc_distance_2.pdf')
	sns.plt.close()
	sns.regplot(f[np.isnan(f)==False].flatten(),d[np.isnan(f)==False].flatten(),scatter_kws={'alpha':.015})
	sns.plt.ylabel('distance')
	sns.plt.xlabel('connectivty')
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/results/fc_distance.pdf')
	sns.plt.close()

def generate_correlation_map(x, y):
	"""
	Correlate each n with each m.
	----------
	Parameters

	x : np.array, shape N X T.
	y : np.array, shape M X T.
	Returns: np.array, N X M array in which each element is a correlation coefficient.
	----------
	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
	    raise ValueError('x and y must ' +
	                     'have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x,
	             y.T) - n * np.dot(mu_x[:, np.newaxis],
	                              mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def fit_matrix_multi(ignore_idx):
	print ignore_idx
	sys.stdout.flush()
	global co_a_matrix
	global gene_exp
	if corr_method == 'pearsonr': temp_m = np.corrcoef(gene_exp[:,np.arange(gene_exp.shape[1])!=ignore_idx])
	if corr_method == 'spearmanr':
		temp_m = spearmanr(gene_exp[:,np.arange(gene_exp.shape[1])!=ignore_idx],axis=1)[0]
		temp_m[ignore_nodes,:] = np.nan
		temp_m[:,ignore_nodes] = np.nan

	np.fill_diagonal(temp_m,np.nan)
	# if corr_method == 'pearsonr': return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]
	# if corr_method == 'spearmanr': return nan_spearmanr(co_a_matrix.flatten(),temp_m.flatten())[0]
	return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]

def fit_matrix(matrix,topological=True,distance=True,network=None):
	try:
		result = np.load('/home/mbmbertolero/data/gene_expression/norm_results/fit_all_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,network,corr_method,norm))
	except:
		global gene_exp
		result = []
		for i in range(gene_exp.shape[1]):
			result.append(fit_matrix_multi(i))
		np.save('/home/mbmbertolero/data/gene_expression/norm_results/fit_all_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,network,corr_method,norm),np.array(result))
	return np.array(result)

def fit_SA(indices_for_correlation):
	global co_a_matrix #grab the matrix we are working with
	global gene_exp #grab the "timeseries" of gene expression, well id by gene expression value shape
	#get correlation matrix for select genes
	if corr_method == 'pearsonr': temp_m = np.corrcoef(gene_exp[:,np.ix_(indices_for_correlation)][:,0,:])
	if corr_method == 'spearmanr':
		temp_m = spearmanr(gene_exp[:,np.ix_(indices_for_correlation)][:,0,:],axis=1)[0]
		temp_m[ignore_nodes,:] = np.nan
		temp_m[:,ignore_nodes] = np.nan
	np.fill_diagonal(temp_m,np.nan)
	# if corr_method == 'pearsonr': return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]
	# if corr_method == 'spearmanr': return nan_spearmanr(co_a_matrix.flatten(),temp_m.flatten())[0]
	return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]

def SA_find_genes(matrix,topological=False,distance=True,network=None,n_genes=100,start_temp=.5,end_temp=.01,temp_step=.001,n_trys=1000,n_loops=5,cores=0,use_prs=True):
	# matrix = 'fc'
	# topological=False
	# distance=True
	# network=0
	# n_genes=100
	# start_temp=.5
	# end_temp=.01
	# temp_step=.01
	# n_trys=100
	# cores=4
	#set number of cores depending on which node we are on
	# cores= multiprocessing.cpu_count()-1

	# grab the coact/fc/sc matrix global variable, which we will assing the matrix we are anylzing to
	# we have to use global variable, as we want to use a lot of cores
	global co_a_matrix
	#load correlation matrix to fit genes coexp to, assign it to the global co_a_matrix variable
	if matrix == 'fc': co_a_matrix = functional_connectivity(topological,distance,network)[:200,:200]
	if matrix == 'sc': co_a_matrix = structural_connectivity(topological,distance,network)[:200,:200]
	np.fill_diagonal(co_a_matrix,np.nan) #remove diagonal

	# initally, start with the top n genes that increase fit
	# this was previously calcualted, it is just the r value between coexpression and the fc/sc/coact matrix
	# without that gene in the co-expression matrix, lower values means it probably helps with fit of coexpression
	if use_prs: gene_r_impact = 1- fit_matrix(matrix,topological,distance,network)
	else: gene_r_impact = np.random.randint(0,100,gene_exp.shape[1])
	#normalize increases/decreases to probabilities that sum to 1
	prs = gene_r_impact/float(np.sum(gene_r_impact))
	#calculate the fit on the top n genes that increase the fit, use as first estimate
	current_genes = np.arange(gene_exp.shape[1])[np.argsort(gene_r_impact)][-n_genes:]
	initial_m = np.corrcoef(gene_exp[:,np.ix_(current_genes)][:,0,:])
	current_r = nan_pearsonr(co_a_matrix[network].flatten(),initial_m[network].flatten())[0]
	print 'start with a fit of: %s' %(str(current_r))


	# set multiprocessing to have the number of cores requested
	if cores > 1: pool = Pool(cores)
	temp = start_temp # set the initial temp
	save_results = []
	while True:
		# set temperature as a proportion of genes to swap in and out of coexpression matrix formulation
		# as temp lowers, we remove fewer and few genes
		current_temp = n_genes * temp
		#matrix where we store which new genes we are trying for the formulation of the coexpression matrix
		potential_indices = np.zeros((n_trys,n_genes)).astype(int)
		for i in range(n_trys): #find n_trys selections to try
			temp_genes = set(current_genes.copy())
			# find genes to remove, bias towards genes that decreased fit
			if use_prs:
				remove_prs = 1-prs[current_genes]
				remove_prs = remove_prs/np.sum(remove_prs) #renormalize
				remove_genes = set(np.random.choice(current_genes,int(current_temp),False,remove_prs))
			else: remove_genes = set(np.random.choice(current_genes,int(current_temp),False))
			# remove these genes
			for gene in remove_genes: temp_genes.remove(gene)
			# find genes to add, bias towards genes that increased fit
			while True: # we have to ensure it gets back to n_genes, since it can select genes aready in the set
				if use_prs:
					temp_genes.add(np.random.choice(gene_exp.shape[1],1,False,prs)[0])
				else:
					temp_genes.add(np.random.choice(gene_exp.shape[1],1,False)[0])
				if len(temp_genes) == n_genes: break
			# save it to the array we are going to send to the multiprocessing function
			potential_indices[i] = np.array(list(temp_genes))
		#send to fit function, which calculates fit of gene coexpression to whatever matrix was selected
		if cores > 0: results = pool.map(fit_SA,potential_indices)
		else:
			results = []
			for pi in potential_indices: results.append(fit_SA(pi))
		results = np.array(results)
		# what is the value of the best fit?
		new_max = np.max(results)
		# did it do better than we previously did?
		if new_max > current_r:
			# if so set current r and current genes one to the current best
			current_r = new_max
			current_genes = potential_indices[np.argmax(results)]
			# save results
			save_results.append(current_genes)
		#if not, continue
		print np.max(results), temp
		sys.stdout.flush()
		# decrease temp
		temp = temp - temp_step
		# is the temp too low?
		if temp < end_temp:
			break
		if int(n_genes * temp) ==0:
			break
			# for small amount of n_genes, need this or it gets stuck.
	# np.save('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,network,n_genes,use_prs,norm,corr_method),np.array(save_results))
	np.save('/home/mbmbertolero/data/gene_expression/final_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,network,n_genes,use_prs,norm,corr_method),np.array(save_results))

def fit_random(indices_for_correlation):
	global co_a_matrix #grab the matrix we are working with
	global random_gene_exp #grab the "timeseries" of gene expression, well id by gene expression value shape
	#get correlation matrix for select genes
	if corr_method == 'pearsonr': temp_m = np.corrcoef(random_gene_exp[:,np.ix_(indices_for_correlation)][:,0,:])
	if corr_method == 'spearmanr':
		temp_m = spearmanr(random_gene_exp[:,np.ix_(indices_for_correlation)][:,0,:],axis=1)[0]
		temp_m[ignore_nodes,:] = np.nan
		temp_m[:,ignore_nodes] = np.nan
	np.fill_diagonal(temp_m,np.nan)
	# if corr_method == 'pearsonr': return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]
	# if corr_method == 'spearmanr': return nan_spearmanr(co_a_matrix.flatten(),temp_m.flatten())[0]
	return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]

def null_genes(matrix,topological=False,distance=True,network=None,n_genes=100,runs=10000,corr_method = 'pearsonr'):
	# matrix = 'fc'
	# topological=False
	# distance=True
	# network=0
	# n_genes=50
	# grab the coact/fc/sc matrix global variable, which we will assing the matrix we are anylzing to
	# we have to use global variable, as we want to use a lot of cores
	global co_a_matrix
	#load correlation matrix to fit genes coexp to, assign it to the global co_a_matrix variable
	if matrix == 'fc': co_a_matrix = functional_connectivity(topological,distance,network)[:200,:200]
	if matrix == 'sc': co_a_matrix = structural_connectivity(topological,distance,network)[:200,:200]
	np.fill_diagonal(co_a_matrix,np.nan) #remove diagonal
	fits = []
	for run in range(runs):
		random_gene_idx = np.arange(gene_exp.shape[1])
		random_gene_idx = np.random.choice(random_gene_idx, size=n_genes, replace=False, p=None)
		assert len(np.unique(random_gene_idx)) == len(random_gene_idx)
		fits.append(fit_SA(random_gene_idx))
	np.save('/home/mbmbertolero/data/gene_expression/null_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
		%(matrix,topological,distance,network,n_genes,norm,corr_method),np.array(fits))

def random_genes(matrix,topological=False,distance=True,network=None,n_genes=100,runs=1000,corr_method = 'pearsonr'):
	# matrix = 'fc'
	# topological=False
	# distance=True
	# network=0
	# n_genes=50
	# grab the coact/fc/sc matrix global variable, which we will assing the matrix we are anylzing to
	# we have to use global variable, as we want to use a lot of cores
	global co_a_matrix
	global random_gene_exp
	#load correlation matrix to fit genes coexp to, assign it to the global co_a_matrix variable
	if matrix == 'fc': co_a_matrix = functional_connectivity(topological,distance,network)[:200,:200]
	if matrix == 'sc': co_a_matrix = structural_connectivity(topological,distance,network)[:200,:200]
	np.fill_diagonal(co_a_matrix,np.nan) #remove diagonal
	fits = []

	for shuffle in range(10):
		random_gene_exp = gene_exp.copy().flatten()
		min_v,max_v = int(np.around(np.nanmin(gene_exp))),int(np.around(np.nanmax(gene_exp)))
		while True:
			if np.isnan(random_gene_exp).any() == False: break
			for i in np.where(np.isnan(random_gene_exp))[0]: random_gene_exp[i] = np.random.randint(min_v,max_v,1)
		np.random.shuffle(random_gene_exp)
		random_gene_exp = random_gene_exp.reshape(gene_exp.shape)
		for run in range(runs):
			random_gene_idx = np.arange(gene_exp.shape[1])
			random_gene_idx = np.random.choice(random_gene_idx, size=n_genes, replace=False, p=None)
			assert len(np.unique(random_gene_idx)) == len(random_gene_idx)
			fits.append(fit_random(random_gene_idx))
	np.save('/home/mbmbertolero/data/gene_expression/random_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
		%(matrix,topological,distance,network,n_genes,norm,corr_method),np.array(fits))

def analyze_null(matrix,topological=False,distance=True,network=None,n_genes=100,runs=10000,corr_method = 'pearsonr'):
	matrix = 'fc'
	topological=False
	distance=True
	n_genes=50
	corr_method = 'pearsonr'
	null_type = 'random'
	runs = 10000
	fits = np.zeros((200,runs))
	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))[:200]
	for i in range(200):
		if i in ignore_nodes:continue
		fits[i] = np.load('/home/mbmbertolero/data/gene_expression/%s_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
		%(null_type,matrix,topological,distance,i,n_genes,norm,corr_method))
	df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_True_%s_False_fits_df.csv'%(matrix,corr_method))
	df = df[df['n_genes'] == n_genes]
	fit_better = np.zeros((200))
	fit_better [:] = np.nan
	for i in range(200):
		if i in ignore_nodes: continue
		fit_better [i] = scipy.stats.ttest_1samp(fits[i],df.fit[df.node==i])[0] * -1

def plot_random_n_genes(null_type = 'random',topological=False,distance=True,corr_method = 'pearsonr'):
	null_type = 'null'
	topological=False
	distance=True
	corr_method = 'pearsonr'
	source = 'snps'

	if source == 'snps':
		result_dir = 'snp_results'
	if source == 'sa':
		result_dir = 'results'
	fc_pc = np.load('/home/mbmbertolero/data/gene_expression/results/fc_pc.npy')[:200]
	sc_pc = np.load('/home/mbmbertolero/data/gene_expression/results/sc_pc.npy')[:200]
	columns = ['random fit','real fit','number of genes','participation coefficient','node','connectivity']
	df = pd.DataFrame(columns=columns)
	real_fc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/%s/fc_True_%s_False_fits_df.csv'%(result_dir,corr_method))
	real_sc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/%s/sc_True_%s_False_fits_df.csv'%(result_dir,corr_method))
	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		for i in range(200):
			if i in ignore_nodes:continue
			fit = np.load('/home/mbmbertolero/data/gene_expression/%s_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
			%(null_type,'fc',topological,distance,i,n_genes,norm,corr_method))
			tdf = pd.DataFrame(columns=columns)
			tdf['participation coefficient'] = [fc_pc[i]]
			tdf['node'] = [i]
			tdf['number of genes'] = [n_genes]
			tdf['random fit'] = [np.nanmean(fit)]
			tdf['real fit'] = [real_fc_df[(real_fc_df['node']==i) & (real_fc_df['n_genes']==n_genes)].fit.values[0]]
			tdf['connectivity'] = 'functional'
			df = df.append(tdf,ignore_index=True)
			fit = np.load('/home/mbmbertolero/data/gene_expression/%s_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
			%(null_type,'sc',topological,distance,i,n_genes,norm,corr_method))
			tdf = pd.DataFrame(columns=columns)
			tdf['participation coefficient'] = [sc_pc[i]]
			tdf['node'] = [i]
			tdf['number of genes'] = [n_genes]
			tdf['random fit'] = [np.nanmean(fit)]
			tdf['real fit'] = [real_sc_df[(real_sc_df['node']==i) & (real_sc_df['n_genes']==n_genes)].fit.values[0]]
			tdf['connectivity'] = 'structural'
			df = df.append(tdf,ignore_index=True)

	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		print scipy.stats.ttest_ind(df['random fit'][(df.connectivity=='functional')&(df['number of genes'] == n_genes)],df['real fit'][(df.connectivity=='functional')&(df['number of genes'] == n_genes)])
	print '----'

	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		print scipy.stats.ttest_ind(df['random fit'][(df.connectivity=='structural')&(df['number of genes'] == n_genes)],df['real fit'][(df.connectivity=='structural')&(df['number of genes'] == n_genes)])

	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		print scipy.stats.ttest_ind(df['random fit'][(df.connectivity=='structural')&(df['number of genes'] == n_genes)],df['random fit'][(df.connectivity=='functional')&(df['number of genes'] == n_genes)])

	c1,c2 = sns.cubehelix_palette(10, rot=.25, light=.7)[0],sns.cubehelix_palette(10, rot=-.25, light=.7)[0]
	sns.set(context="paper",font_scale=2,font='Open Sans',style='whitegrid',palette="pastel",color_codes=True)
	fig = plt.figure(figsize=(14.3,8))
	g = sns.violinplot(x="number of genes", y="random fit",split=True,hue='connectivity',data=df,inner="quartile",palette={"functional": c1, "structural": c2},width=.9)
	sns.plt.tight_layout()
	sns.despine(left=True)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_random_fits_sc_fc_%s_%s.pdf'%(null_type,corr_method))
	plt.show()
	plt.close()

	n_gene_array = df['number of genes'].unique()
	for connectivity in ['functional','structural']:
		if connectivity == 'functional': pal = sns.cubehelix_palette(10, rot=.25, light=.7)
		else: pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
		sns.set(context="paper",font_scale=1.5,font='Open Sans',style='white',palette="pastel",color_codes=True)
		g = sns.lmplot(x="random fit", y="participation coefficient", ci=None,col="number of genes", sharex=False,sharey=False,palette=pal, col_wrap=5,hue="number of genes", data=df[df.connectivity==connectivity],size=14.3/5.)
		for idx,ax in enumerate(g.axes):
			n_genes = n_gene_array[idx]
			x,y = df[df.connectivity==connectivity]['random fit'].values[df[df.connectivity==connectivity]['number of genes']==n_genes],df[df.connectivity==connectivity]['participation coefficient'][df[df.connectivity==connectivity]['number of genes']==n_genes]
			r,p = nan_pearsonr(x,y)
			ax.set_xlim(np.min(x),np.max(x))
			ax.text(np.max(x),np.max(y),'$\it{r}$=%s\n%s' %(np.around(r,3),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
		for idx,ax in enumerate(g.axes):
			n_genes = n_gene_array[idx]
			ax.set_title(n_genes,color=pal[idx])
			ax.set_ylim(0,.9)
		sns.plt.tight_layout()
		sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_%s_pc_%s_%s.pdf'%(null_type,connectivity,corr_method))
		plt.close()

	columns = ['fit increase','real fit','number of genes','participation coefficient','node','connectivity']
	df = pd.DataFrame(columns=columns)
	real_fc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_True_%s_False_fits_df.csv'%(corr_method))
	real_sc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_True_%s_False_fits_df.csv'%(corr_method))
	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		# fc_mean,fc_std,sc_mean,sc_std = real_fc_df.fit[real_fc_df['n_genes']==n_genes].mean(), real_fc_df.fit[real_fc_df['n_genes']==n_genes].std(),real_sc_df.fit[real_fc_df['n_genes']==n_genes].mean(), real_sc_df.fit[real_fc_df['n_genes']==n_genes].std()
		for i in range(200):
			if i in ignore_nodes:continue
			fits = np.load('/home/mbmbertolero/data/gene_expression/%s_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
			%(null_type,'fc',topological,distance,i,n_genes,norm,corr_method))
			fc_mean,fc_std = np.mean(fits),np.std(fits)
			fits = (fits - fc_mean)/ fc_std
			tdf = pd.DataFrame(columns=columns)
			tdf['participation coefficient'] = [fc_pc[i]]
			tdf['node'] = [i]
			tdf['number of genes'] = [n_genes]
			tdf['real fit'] = [(real_fc_df[(real_fc_df['node']==i) & (real_fc_df['n_genes']==n_genes)].fit.values[0] - fc_mean) / fc_std]
			tdf['fit increase'] = [scipy.stats.ttest_1samp(fits,tdf['real fit'])[0].values[0] * -1]
			tdf['connectivity'] = 'functional'
			df = df.append(tdf,ignore_index=True)
			fits = np.load('/home/mbmbertolero/data/gene_expression/%s_results/random_fit_%s_%s_%s_%s_%s_%s_%s.npy'\
			%(null_type,'sc',topological,distance,i,n_genes,norm,corr_method))
			sc_mean,sc_std = np.mean(fits),np.std(fits)
			fits = (fits - sc_mean)/ sc_std
			tdf = pd.DataFrame(columns=columns)
			tdf['participation coefficient'] = [sc_pc[i]]
			tdf['node'] = [i]
			tdf['number of genes'] = [n_genes]
			tdf['real fit'] = [(real_sc_df[(real_sc_df['node']==i) & (real_sc_df['n_genes']==n_genes)].fit.values[0] - sc_mean) / sc_std]
			tdf['fit increase'] = [scipy.stats.ttest_1samp(fits,tdf['real fit'])[0].values[0] * -1]
			tdf['connectivity'] = 'structural'
			df = df.append(tdf,ignore_index=True)

	print pearsonr(df['fit increase'][df.connectivity=='functional'],df['participation coefficient'][df.connectivity=='functional'])
	print pearsonr(df['fit increase'][df.connectivity=='structural'],df['participation coefficient'][df.connectivity=='structural'])

	c1,c2 = sns.cubehelix_palette(10, rot=.25, light=.7)[0],sns.cubehelix_palette(10, rot=-.25, light=.7)[0]
	sns.set(context="paper",font_scale=2,font='Open Sans',style='whitegrid',palette="pastel",color_codes=True)
	fig = plt.figure(figsize=(14.3,8))
	g = sns.violinplot(x="number of genes", y="fit increase",split=True,hue='connectivity',data=df,inner="quartile",palette={"functional": c1, "structural": c2},width=.9)
	sns.plt.tight_layout()
	sns.despine(left=True)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_fit_increase_%s_%s.pdf'%(null_type,corr_method))
	plt.show()
	plt.close()

def super():
	genes = np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')
	super_genes = pd.read_csv('supergran.csv',header=None).values.reshape(-1)
	fc = functional_connectivity(False,False,None)
	indices_for_correlation = []
	for g in super_genes:
		try:indices_for_correlation.append(np.where(genes==g)[0][0])
		except:continue
	temp_m = np.corrcoef(gene_exp[:,np.ix_(indices_for_correlation)][:,0,:])
	print nan_pearsonr(fc[:200,:200].flatten(),temp_m.flatten())
	r_vals = []
	for i in range(1000):
		indices_for_correlation = np.random.randint(0,genes.shape[0],46)
		temp_m = np.corrcoef(gene_exp[:,np.ix_(indices_for_correlation)][:,0,:])
		r_vals.append(nan_pearsonr(fc[:200,:200].flatten(),temp_m.flatten())[0])

def generate_for_ann(matrix):
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	for i,n in enumerate(yeo_df):
		membership[i] = int(reduce_dict[n.split('_')[2]])
	membership = membership[:200][mask]
	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))[:200][mask]
	pc[pc>np.percentile(pc,80)] = 1
	pc[pc!=1] = 0
	rank = np.load('/home/mbmbertolero/data/gene_expression/norm_results/%s_pearsonr_genes.npy'%(matrix))
	names = np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')
	np.savetxt('/home/mbmbertolero/data/gene_expression/ann/gene_names.txt',names,fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/ann/%s_gene_ranks.txt'%(matrix),rank[mask])
	np.savetxt('/home/mbmbertolero/data/gene_expression/ann/%s_membership.txt'%(matrix),membership.astype(int))
	np.savetxt('/home/mbmbertolero/data/gene_expression/ann/%s_is_connector_hub.txt'%(matrix),pc)

def gorilla(matrix):
	matrix = 'fc'
	genes = np.loadtxt('/home/mbmbertolero/gene_expression/ann/masked_names_%s.txt'%(matrix),dtype=str)
	ranks = np.load('/home/mbmbertolero/data/gene_expression/norm_results/%s_%s_genes.npy'%(matrix,'pearsonr'))
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0),fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s_cut'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0)[:1000],fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s_top'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0)[:8343],fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s_bottom'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0)[8343:],fmt='%s')

	matrix = 'sc'
	genes = np.loadtxt('/home/mbmbertolero/gene_expression/ann/masked_names_%s.txt'%(matrix),dtype=str)
	ranks = np.load('/home/mbmbertolero/data/gene_expression/norm_results/%s_%s_genes.npy'%(matrix,'pearsonr'))
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0),fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s_cut'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0)[:1000],fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s_top'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0)[:8343],fmt='%s')
	np.savetxt('/home/mbmbertolero/data/gene_expression/gorilla_%s_bottom'%(matrix),np.flip(genes[np.argsort(np.sum(ranks,axis=0))],0)[8343:],fmt='%s')

def two_lines(e):
	fin = []
	for string in  e.split(' ')[len(e.split(' '))/2:]:
		fin.append(string)
	fin.append('\n')
	for string in  e.split(' ')[:len(e.split(' '))/2]:
		fin.append(string)
	return' '.join(fin)

def plot_gorilla(matrix):
	head = ['GO Term','Description','P-value','FDR q-value','Enrichment','colors','erank','sig']
	df = pd.DataFrame(columns=head)
	maps= ["ch:2.5,-.2,dark=.3","ch:2.5,.2,dark=.3","ch:0,.2,dark=.3"]
	biggest = 0
	for go,cmap in zip(['PROCESS','COMPONENT','FUNCTION'],maps):
		t_df = pd.read_csv('/home/mbmbertolero/gene_expression/GOrilla/GO%s_%s.csv'%(go,matrix.upper()),header=0,names=head[:-3],usecols=range(5),sep='\t')
		ontology = np.zeros(len(t_df)).astype(str)
		ontology[:] = go.lower()
		t_df['Ontology'] = ontology
		t_df = t_df[t_df['FDR q-value']<0.05]
		t_df.Enrichment = np.char.replace(t_df.Enrichment.values.astype(str),',','').astype(float)
		# t_df.Description = np.char.replace(t_df.Description.values.astype(str),',','\n')
		t_df['erank'] = np.argsort(t_df['FDR q-value'].values)
		t_df['sig'] = np.zeros(len(t_df)).astype(bool)
		t_df['sig'][t_df['FDR q-value']<=0.05] = True
		t_df['colors'] = make_heatmap(t_df['erank'].values,cmap)
		if t_df.shape[0] > biggest: biggest = t_df.shape[0]
		df = df.append(t_df)
	for e in df.Description:
		if len(e)>50:
			df.Description[df.Description==e] = two_lines(e)
	if biggest <= 50: biggest = 50
	all_locs = (np.arange(biggest+2)/float(biggest+2))[1:]
	left, width = 0, 1
	bottom, height = 0, 1
	right = left + width
	top = bottom + height
	fig = plt.figure(figsize=(7.44094,9.72441),frameon=False)
	for col,ontology in zip(np.linspace(.2,.8,3),['process','component','function']):
		order = df[df.Ontology==ontology]['Description'].values[np.argsort(df[df.Ontology==ontology]['FDR q-value'].values*-1)]
		scores = df[df.Ontology==ontology]['Enrichment'].values[np.argsort(df[df.Ontology==ontology]['FDR q-value'].values*-1)]
		p_vals = df[df.Ontology==ontology]['FDR q-value'].values[np.argsort(df[df.Ontology==ontology]['FDR q-value'].values*-1)]
		sigs = df[df.Ontology==ontology]['FDR q-value'].values[np.argsort(df[df.Ontology==ontology]['FDR q-value'].values*-1)]

		for ix,o in enumerate(order):
			order[ix] = order[ix] + ' (%s)'%(scores[ix])
			order[ix] = order[ix].capitalize()
			if sigs[ix] <= 0.05 and sigs[ix] > 0.01: order[ix] = order[ix] + ' *'
			if sigs[ix] <= 0.01: order[ix] = order[ix] + ' **'
		order = np.append(order,ontology.capitalize())
		colors = df[df.Ontology==ontology]['colors'].values[np.argsort(df[df.Ontology==ontology]['erank'].values)]
		colors = list(colors)
		colors.append(np.array(colors[-1]) /1.25)
		locs = all_locs[-len(order):]
		for idx,i,t,c in zip(range(len(locs)),locs,order,colors):
			if idx == len(colors) -1: size = 10
			else: size = 7
			fig.text(col*(left+right), (float(i)*(bottom+top)), t,horizontalalignment='center',verticalalignment='center',fontsize=size, color=c)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/gorilla_%s.pdf'%(matrix))

def log_p_value(p):
	if p > 0.001:
		p = np.around(p,3)
		p = "$\it{p}$=%s"%(p)
	else:
		p = (-1) * np.log10(p)
		p = "-log10($\it{p}$)=%s"%(np.around(p,0).astype(int))
	return p

def convert_r_p(r,p):
	return "$\it{r}$=%s\n%s"%(np.around(r,3),log_p_value(p))

def convert_t_p(t,p):
	return "$\it{t}$=%s\n%s"%(np.around(t,3),log_p_value(p))

def pearsonr_v_spearmanr_ge_figure_and_stats(matrix):
	use_prs=False
	norm=True
	distance=True
	topological=False
	sns.plt.ion()
	if matrix == 'fc': m = functional_connectivity(topological,distance,None)[:200,:200]
	if matrix == 'sc': m = structural_connectivity(topological,distance,None)[:200,:200]
	np.fill_diagonal(m,np.nan)

	df = pd.DataFrame(columns=['fit difference','n_genes'])
	s = []
	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		for node in range(m.shape[0]):
			if node in ignore_nodes: continue
			p_gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,node,n_genes,use_prs,norm,'pearsonr'))[-1]
			p_gene_exp_matrix = np.corrcoef(gene_exp[:,p_gene_exp_matrix])
			p_gene_exp_matrix[ignore_nodes,:] = np.nan
			p_gene_exp_matrix[:,ignore_nodes] = np.nan
			p_gene_exp_matrix = p_gene_exp_matrix[node]
			p_gene_exp_matrix = p_gene_exp_matrix[np.isnan(p_gene_exp_matrix)==False]

			s_gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,node,n_genes,use_prs,norm,'spearmanr'))[-1]
			s_gene_exp_matrix = spearmanr(gene_exp[:,s_gene_exp_matrix],axis=1)[0]
			s_gene_exp_matrix[ignore_nodes,:] = np.nan
			s_gene_exp_matrix[:,ignore_nodes] = np.nan
			s_gene_exp_matrix = s_gene_exp_matrix[node]
			s_gene_exp_matrix = s_gene_exp_matrix[np.isnan(s_gene_exp_matrix)==False]
			t = scipy.stats.ttest_ind(p_gene_exp_matrix.flatten(),s_gene_exp_matrix.flatten())[0]
			df = df.append(pd.DataFrame(np.array([[t],[n_genes]]).transpose(),columns=['fit difference','n_genes']))
	df['fit difference'] = df['fit difference'].astype(float)
	df['n_genes'] = df['n_genes'].astype(int)

	# Initialize the FacetGrid object
	if matrix == 'fc': pal = sns.cubehelix_palette(10, rot=.25, light=.7)
	if matrix == 'sc': pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
	sns.set(context="notebook",font='Open Sans',style='white')
	g = sns.FacetGrid(df, row="n_genes", hue="n_genes",palette=pal)

	# Draw the densities in a few steps
	g.map(sns.kdeplot, "fit difference", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.3)
	g.map(sns.kdeplot, "fit difference", clip_on=False, color="w", lw=2, bw=.3)
	# g.map(plt.axhline, y=0, lw=2, clip_on=False)
	# Define and use a simple function to label the plot in axes coordinates
	def label(x, color, label):
	    ax = plt.gca()
	    ax.text(0, .2, label, fontweight="bold", color=color,
	            ha="left", va="center", transform=ax.transAxes)


	g.map(label, "n_genes")

	# Set the subplots to overlap
	# g.fig.subplots_adjust(hspace=-.25)

	# Remove axes details that don't play well with overlap
	g.set_titles("")
	g.set(yticks=[])
	sns.plt.xlim(-4,4)
	g.despine(bottom=True, left=True)
	# time.sleep(10)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/pearsonr_v_spearmanr_ge_figure_%s.pdf'%(matrix))
	sns.plt.close()

	print df.groupby('n_genes').mean()
	print np.mean(df['fit difference'])

	df = pd.read_csv('/home/mbmbertolero/gene_expression/results/%s_True_pearsonr_False_fits_df.csv'%(matrix))
	df.rename(columns={'n_genes': 'number of genes', 'fit': 'genetic fit, pearsonr'}, inplace=True)
	other_df = pd.read_csv('/home/mbmbertolero/gene_expression/results/%s_True_spearmanr_False_fits_df.csv'%(matrix))
	df['genetic fit, spearmanr'] = other_df.fit

	n_gene_array = df['number of genes'].unique()
	sns.set(context="paper",font_scale=1.5,font='Open Sans',style='white',palette="pastel")
	g = sns.lmplot(x="genetic fit, pearsonr", y="genetic fit, spearmanr", col="number of genes", palette=pal,col_wrap=5,hue="number of genes", data=df, fit_reg=True,size=14.3/5.)
	for idx,ax in enumerate(g.axes):
		n_genes = n_gene_array[idx]
		x,y = df['genetic fit, pearsonr'].values[df['number of genes']==n_genes],df['genetic fit, spearmanr'][df['number of genes']==n_genes]
		r,p = nan_pearsonr(x,y)

		ax.text(.7,.25,'r=%s\n%s' %(np.around(r,3),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
	sns.plt.tight_layout()
	sns.plt.ylim(0.2,0.85)
	sns.plt.xlim(0.2,0.85)
	g.set_titles(col_template="{col_name}", fontweight='bold', fontsize=12)
	for idx,ax in enumerate(g.axes):
		n_genes = n_gene_array[idx]
		ax.set_title(n_genes,color=pal[idx])

	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/pearsonr_v_spearmanr_%s.pdf'%(matrix))
	sns.plt.show()
	"""
	# df = pd.read_csv('/home/mbmbertolero/gene_expression/results/%s_True_pearsonr_False_fits_df.csv'%(matrix))
	# df['similarity metric'] = 'pearsonr'
	# # df.rename(columns={'n_genes': 'number of genes', 'fit': 'genetic fit'}, inplace=True)
	# other_df = pd.read_csv('/home/mbmbertolero/gene_expression/results/%s_True_spearmanr_False_fits_df.csv'%(matrix))
	# other_df['similarity metric'] = 'spearmanr'


	# df = df.append(other_df)
	# df.rename(columns={'n_genes': 'number of genes', 'fit': 'genetic fit'}, inplace=True)
	# n_gene_array = df['number of genes'].unique()

	# sns.set(context="paper",font_scale=1.75,font='Open Sans',style='white',palette="pastel")
	# fig = plt.figure(figsize=(14.3,8))
	# g = sns.violinplot(x="number of genes", y="genetic fit", hue="similarity metric",data=df, palette="pastel",split=True,inner="quartile")
	# for x in range(len(n_gene_array)):
	# 	n_genes = n_gene_array[x]
	# 	# print n_genes, np.mean(fc_df.fit[fc_df.n_genes==n_genes]), np.mean(sc_df.fit[sc_df.n_genes==n_genes])
	# 	t = scipy.stats.ttest_ind(df['genetic fit'][(df['number of genes']==n_genes).values & (df['similarity metric']=='pearsonr').values],df['genetic fit'][(df['number of genes']==n_genes).values & (df['similarity metric']=='spearmanr').values])[0]
	# 	g.text(x,.4,'t: %s' %(np.around(t,1)))
	# # sns.plt.xlabel('number of genes')
	# sns.plt.ylabel('genetic fit')
	# sns.plt.tight_layout()
	# sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/pearsonr_v_spearmanr.pdf')
	# sns.plt.show()
	# print pearsonr(df.groupby('node')['genetic fit, pearsonr'].mean(),df.groupby('node')['genetic fit, spearmanr'].mean())
	"""

def co_exp_figure():
	matrix = 'fc'
	use_prs = False
	norm = True
	corr_method = 'pearsonr'
	sns.set(context="notebook",font='Open Sans',style='white')
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]
		membership_ints[i] = int(full_dict[n.split('_')[2]])

	if matrix == 'fc': m = functional_connectivity(topological,distance,None)[:200,:200]
	if matrix == 'sc': m = structural_connectivity(topological,distance,None)[:200,:200]
	np.fill_diagonal(m,np.nan)
	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
	strength = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))
	wcs = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))
	between = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))
	df = pd.DataFrame(columns=['node','fit','network','participation coefficient','within community strength','strength','betweenness','n_genes'])
	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		for node,name,pc_val,wcs_val,strength_val,b_val in zip(range(m.shape[0]),membership,pc,wcs,strength,between):
			if node in ignore_nodes: continue
			gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,node,n_genes,use_prs,norm,corr_method))[-1]
			ticks = np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')[gene_exp_matrix]
			break
		break
	g = gene_exp[:,gene_exp_matrix]
	corr_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
	np.fill_diagonal(corr_matrix,np.nan)

	# fig, axes = plt.subplots(2,gridspec_kw={'height_ratios': [10, 1,]})
	# h1=sns.heatmap(g[:30,:].transpose(), ax=axes[0], cbar=False,square=True,cmap=sns.diverging_palette(220, 10, as_cmap=True),vmin=-1,vmax=1)
	# # cmap=sns.diverging_palette(220, 10, sep=80, n=100,as_cmap=True)
	# h1.set_title('gene expression')
	# h1.set_ylabel("genes")
	# h1.set_xlabel('nodes')
	# h1.set_yticklabels(ticks,rotation=360)
	# # h1.set_xticks([],[])
	# h1.set_xticklabels(h1.get_xticklabels(),rotation=360)
	# h2=sns.heatmap(corr_matrix[0,:30].reshape(1,30), ax=axes[1], cbar=False,square=True,cmap=sns.diverging_palette(220, 10, as_cmap=True))
	# # cmap=sns.diverging_palette(220, 10, sep=80, n=100,as_cmap=True)
	# h2.set_xlabel("gene coexpression between node 0 and other nodes")
	# h2.set_yticks([],[])
	# h2.set_xticks([],[])
	# plt.savefig('coexpression_example.pdf')

	fig, axes = plt.subplots(2,gridspec_kw={'height_ratios': [10, 1,]})
	h1=sns.heatmap(g[:30,:].transpose(), ax=axes[0], cbar=False,square=True,cmap=sns.diverging_palette(220, 10, as_cmap=True),vmin=-1,vmax=1)
	# cmap=sns.diverging_palette(220, 10, sep=80, n=100,as_cmap=True)
	h1.set_title('gene expression')
	h1.set_ylabel("genes")
	h1.set_xlabel('nodes')
	h1.set_yticklabels(ticks,rotation=360)
	# h1.set_xticks([],[])
	h1.set_xticklabels(h1.get_xticklabels(),rotation=360)
	h2=sns.heatmap(m[0,:30].reshape(1,30), ax=axes[1], cbar=False,square=True,cmap=sns.diverging_palette(220, 10, as_cmap=True))
	# cmap=sns.diverging_palette(220, 10, sep=80, n=100,as_cmap=True)
	h2.set_xlabel("connectivity between node 0 and other nodes")
	h2.set_yticks([],[])
	h2.set_xticks([],[])
	plt.savefig('coexpression_example_con.pdf')

def make_fit_df(matrix,use_prs,norm,corr_method,distance=True,topological=False,source='snps',components='edges'):
	matrix = 'fc'
	use_prs = False
	norm = True
	corr_method = 'pearsonr'
	distance = True
	topological = False
	source = 'snps'
	if source == 'snps':
		result_dir = 'snp_results'
	if source == 'sa':
		result_dir = 'results'
	sns.set(context="notebook",font='Open Sans',style='white')
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]
		membership_ints[i] = int(full_dict[n.split('_')[2]])

	if matrix == 'fc': m = functional_connectivity(topological,distance,None)[:200,:200]
	if matrix == 'sc': m = structural_connectivity(topological,distance,None)[:200,:200]
	np.fill_diagonal(m,np.nan)

	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
	strength = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))
	wcs = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))
	between = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))

	# subjects = get_subjects(done='graphs')
	# static_results = get_graphs(subjects,matrix)
	# pc = np.nanmean(static_results['subject_pcs'],axis=0)
	# wcs = np.nanmean(static_results['subject_wmds'],axis=0)

	df = pd.DataFrame(columns=['node','fit','network','participation coefficient','within community strength','strength','betweenness','n_genes'])
	for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		for node,name,pc_val,wcs_val,strength_val,b_val in zip(range(m.shape[0]),membership,pc,wcs,strength,between):
			if node in ignore_nodes: continue
			if source == 'sa':
				gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,node,n_genes,use_prs,norm,corr_method))[-1]
			if source == 'snps':
				gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/snp_results/%s_%s_%s_gene_snps_mean.npy'%(components,node,matrix))
				gene_exp_matrix[np.isnan(gene_exp_matrix)] = 0.0
				gene_exp_matrix = np.argsort(gene_exp_matrix)[-n_genes:]
			if corr_method == 'pearsonr':
				gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
			if corr_method == 'spearmanr':
				gene_exp_matrix = spearmanr(gene_exp[:,gene_exp_matrix],axis=1)[0]
				gene_exp_matrix[ignore_nodes,:] = np.nan
				gene_exp_matrix[:,ignore_nodes] = np.nan
			r = nan_pearsonr(m[node,:].flatten(),gene_exp_matrix[node,:].flatten())[0]
			df= df.append(pd.DataFrame(np.array([[node],[r],[name],[pc_val],[wcs_val],[strength_val],[b_val],[n_genes]]).transpose(),columns=['node','fit','network','participation coefficient','within community strength','strength','betweenness','n_genes']))
	df.fit = df.fit.astype(float)
	df['participation coefficient'] = df['participation coefficient'].astype(float)
	df['strength'] = df['strength'].astype(float)
	df['within community strength'] = df['within community strength'].astype(float)
	df['betweenness'] = df['betweenness'].astype(float)
	df['fit'] = df['fit'].astype(float)
	df.to_csv('/home/mbmbertolero/data/gene_expression/%s/%s_%s_%s_%s_fits_df.csv'%(result_dir,matrix,norm,corr_method,use_prs))

def compare_n_genes(corr_method = 'pearsonr'):
	fc_df = pd.read_csv('/home/mbmbertolero/gene_expression/results/fc_True_%s_False_fits_df.csv'%(corr_method))
	fc_df['connectivity'] = 'functional'
	fc_df.rename(columns={'n_genes': 'number of genes', 'fit': 'genetic fit'}, inplace=True)
	sc_df = pd.read_csv('/home/mbmbertolero/gene_expression/results/sc_True_%s_False_fits_df.csv'%(corr_method))
	sc_df['connectivity'] = 'structural'
	sc_df.rename(columns={'n_genes': 'number of genes', 'fit': 'genetic fit'}, inplace=True)
	df = fc_df.append(sc_df)
	n_gene_array = df['number of genes'].unique()

	# p_df = pd.read_csv('/home/mbmbertolero/gene_expression/results/fc_True_pearsonr_False_fits_df.csv')
	# s_df = pd.read_csv('/home/mbmbertolero/gene_expression/results/fc_True_spearmanr_False_fits_df.csv')
	# fit = np.mean([p_df.groupby('node')['fit'].mean(),s_df.groupby('node')['fit'].mean()],axis=0)
	# print pearsonr(fc_df.groupby('node')['genetic fit'].mean(),fc_df.groupby('node')['participation coefficient'].mean())
	# print pearsonr(fit,fc_df.groupby('node')['participation coefficient'].mean())


	sns.set(context="paper",font_scale=1.75,font='Open Sans',style='whitegrid',palette="pastel",color_codes=True)
	c1,c2 = sns.cubehelix_palette(10, rot=.25, light=.7)[0],sns.cubehelix_palette(10, rot=-.25, light=.7)[0]
	fig = plt.figure(figsize=(14.3,8))
	g = sns.violinplot(x="number of genes", y="genetic fit", hue="connectivity",data=df,split=True,inner="quartile",palette={"functional": c1, "structural": c2},)
	for x in range(len(n_gene_array)):
		n_genes = n_gene_array[x]
		t,p = scipy.stats.ttest_ind(fc_df['genetic fit'][fc_df['number of genes']==n_genes],sc_df['genetic fit'][sc_df['number of genes']==n_genes])
		g.text(x,.05,'$\it{t}$=%s\n%s' %(np.around(t,1),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
	sns.plt.xlabel('number of genes')
	sns.plt.ylabel('genetic fit')
	sns.plt.ylim(0,1)
	sns.plt.tight_layout()
	sns.despine(left=True)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_sc_v_fc_%s.pdf'%(corr_method))
	sns.plt.close()

	pal = sns.cubehelix_palette(10, rot=.25, light=.7)
	sns.set(context="paper",font_scale=1.5,font='Open Sans',style='white',palette="pastel",color_codes=True)
	g = sns.lmplot(x="genetic fit", y="participation coefficient", col="number of genes", sharex=False,palette=pal, col_wrap=5,hue="number of genes", data=fc_df,size=14.3/5.)
	for idx,ax in enumerate(g.axes):
		n_genes = n_gene_array[idx]
		x,y = fc_df['genetic fit'].values[fc_df['number of genes']==n_genes],fc_df['participation coefficient'][fc_df['number of genes']==n_genes]
		r,p = nan_pearsonr(x,y)
		ax.text(max(x),.7,'$\it{r}$=%s\n%s' %(np.around(r,3),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
	for idx,ax in enumerate(g.axes):
		n_genes = n_gene_array[idx]
		ax.set_title(n_genes,color=pal[idx])
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_pc_fc_%s.pdf'%(corr_method))
	sns.plt.close()

	pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
	sns.set(context="paper",font_scale=1.5,font='Open Sans',style='white',palette="pastel",color_codes=True)
	g = sns.lmplot(x="genetic fit", y="participation coefficient", col="number of genes", sharex=False,palette=pal, col_wrap=5,hue="number of genes", data=sc_df,size=14.3/5.)
	for idx,ax in enumerate(g.axes):
		n_genes = n_gene_array[idx]
		x,y = sc_df['genetic fit'].values[sc_df['number of genes']==n_genes],sc_df['participation coefficient'][sc_df['number of genes']==n_genes]
		r,p = nan_pearsonr(x,y)
		ax.text(max(x),.7,'$\it{r}$=%s\n%s' %(np.around(r,3),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
	for idx,ax in enumerate(g.axes):
		n_genes = n_gene_array[idx]
		ax.set_title(n_genes,color=pal[idx])
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_pc_sc_%s.pdf'%(corr_method))
	sns.plt.close()

	fc_m = np.zeros((len(n_gene_array),len(n_gene_array)))
	for i in range(len(n_gene_array)):
		i_g = n_gene_array[i]
		for j in range(len(n_gene_array)):
			j_g = n_gene_array[j]
			fc_m[i,j] = pearsonr(fc_df['genetic fit'][fc_df['number of genes']==n_gene_array[i]],fc_df['genetic fit'][fc_df['number of genes']==n_gene_array[j]])[0]
	np.fill_diagonal(fc_m,np.nan)
	for i in range(fc_m.shape[0]):
		print n_gene_array[i],np.nanmin(fc_m[i])
	fig = plt.figure(figsize=(15*.75,12*.75))
	heatmap = sns.heatmap(fc_m,cmap=sns.cubehelix_palette(10, rot=.25, light=.7,as_cmap=True),vmin=.4,vmax=.85,square=True)
	heatmap.set_xticklabels(n_gene_array)
	heatmap.set_yticklabels(np.flip(np.array(n_gene_array),0))
	plt.ylabel('number of genes')
	plt.xlabel('number of genes')
	plt.yticks(rotation=0)
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_fc_corr_%s.pdf'%(corr_method))
	sns.plt.close()

	sc_m = np.zeros((len(n_gene_array),len(n_gene_array)))
	for i in range(len(n_gene_array)):
		i_g = n_gene_array[i]
		for j in range(len(n_gene_array)):
			j_g = n_gene_array[j]
			sc_m[i,j] = pearsonr(sc_df['genetic fit'][sc_df['number of genes']==n_gene_array[i]],sc_df['genetic fit'][sc_df['number of genes']==n_gene_array[j]])[0]
	np.fill_diagonal(sc_m,np.nan)
	for i in range(sc_m.shape[0]):
		print n_gene_array[i],np.nanmin(sc_m[i])
	fig = plt.figure(figsize=(15*.75,12*.75))
	heatmap = sns.heatmap(sc_m,cmap=sns.cubehelix_palette(10, rot=-.25, light=.7,as_cmap=True),vmin=.4,vmax=.85,square=True)
	heatmap.set_xticklabels(n_gene_array)
	heatmap.set_yticklabels(np.flip(np.array(n_gene_array),0))
	plt.yticks(rotation=0)
	plt.ylabel('number of genes')
	plt.xlabel('number of genes')
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/n_genes_sc_corr_%s.pdf'%(corr_method))
	sns.plt.close()

def consensus_genes(matrix='fc',n_genes=100):
	distance=True
	topological=False
	use_prs=False
	corr_method='pearsonr'
	n_genes = 100

	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((200)).astype(str)
	small_membership = np.zeros((200)).astype(str)
	membership_ints = np.zeros((200)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:200]

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	for i,n in enumerate(yeo_df):
		small_membership[i] = names[int(reduce_dict[n.split('_')[2]])]
		membership[i] = n.split('_')[2]
		membership_ints[i] = int(full_dict[n.split('_')[2]])

	fc_c = np.zeros((200,10,16699))
	for node in range(200):
		for n_idx,n_genes in enumerate([50]):
			if node in ignore_nodes:
				fc_c[node,n_idx:] = np.nan
			else: fc_c[node,n_idx,np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%('fc',topological,distance,node,n_genes,use_prs,norm,corr_method))[-1]] = 1

	sc_c = np.zeros((200,10,16699))
	for node in range(200):
		for n_idx,n_genes in enumerate([50]):
			if node in ignore_nodes:
				sc_c[node,n_idx:] = np.nan
			else: sc_c[node,n_idx,np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%('sc',topological,distance,node,n_genes,use_prs,norm,corr_method))[-1]] = 1

	fc_c = np.nansum(fc_c,axis=1)
	sc_c = np.nansum(sc_c,axis=1)
	fc_c = fc_c[mask]
	sc_c = sc_c[mask]
	small_membership = small_membership[mask]

	# np.save('/data/jux/mbmbertolero/gene_expression/all_consensus_%s.npy'%(matrix),c)

	# all_c = np.nanmean(c,axis=0)
	# np.save('/data/jux/mbmbertolero/gene_expression/consensus_%s.npy'%(matrix),all_c)

	# network_c = np.zeros((8,16699))
	# for i,n in enumerate(names):
	# 	network_c[i] = np.nanmean(c[small_membership==n],axis=0)
	# np.save('/data/jux/mbmbertolero/gene_expression/network_consensus_%s.npy'%(matrix),network_c)

	# df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s_%s_fits_df.csv'%(matrix,norm,corr_method,use_prs))
	# df = df[df.n_genes==n_genes]
	# pc = df['participation coefficient']

	# pc_c = np.zeros((2,16699))
	# cutoff = np.percentile(pc,80)
	# pc_c[0] = np.nanmean(c[pc>=cutoff],axis=0)
	# pc_c[1] = np.nanmean(c[pc<cutoff],axis=0)
	# np.save('/data/jux/mbmbertolero/gene_expression/hub_consensus_%s.npy'%(matrix),pc_c)

	c1,c2 = sns.cubehelix_palette(10, rot=.25, light=.7)[0],sns.cubehelix_palette(10, rot=-.25, light=.7)[0]

	fc_g = np.nansum(fc_c,axis=0)
	fc_g = fc_g[np.argsort(fc_g)][-500:]
	sc_g = np.nansum(sc_c,axis=0)
	sc_g = sc_g[np.argsort(sc_g)][-500:]
	hist_df = pd.DataFrame()
	con_type = np.zeros(len(fc_g)*2).astype(str)
	con_type[:len(fc_g)] = 'functional'
	con_type[len(fc_g):] = 'structural'

	# hist_df['number of nodes fit by gene'] = np.array([fc_g,sc_g]).flatten()
	# hist_df['connectivity type'] = con_type
	# hist_df['genes'] = np.array([range(len(fc_g)), range(len(fc_g))]).flatten()
	# fig =sns.tsplot(data=hist_df[hist_df.genes>0],unit='connectivity type',time='genes',condition='connectivity type',color=[c1,c2],value='number of nodes fit by gene')
	# sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/how_many_genes.pdf')
	# sns.plt.show()



	fig,subplots = sns.plt.subplots(2,figsize=(7.204724,9.72440945))
	subplots = subplots.reshape(-1)
	g = np.nansum(fc_c,axis=0)
	print len(g[g>0]),np.mean(g[g>0])
	fig = sns.distplot(g[g>0], rug=True, rug_kws={"color": c1,'height':.1}, kde_kws={"color": c1, "shade":True,"bw":.5,"lw": 3, "label": "functional"},hist=False,ax=subplots[1])
	g = np.nansum(sc_c,axis=0)
	print len(g[g>0]),np.mean(g[g>0])
	fig = sns.distplot(g[g>0], rug=True, rug_kws={"color": c2,'height':.1}, kde_kws={"color": c2, "shade":True,"bw":.5,"lw": 3, "label": "structural"},hist=False,ax=subplots[0])

	subplots[0].set_xlim(-2,22)
	subplots[1].set_xlim(-2,22)
	subplots[0].set_xticks(range(-2,23))
	subplots[1].set_xticks(range(-2,23))
	subplots[1].set_xlabel('number of nodes fit')
	subplots[1].set_ylabel('fraction of genes')
	subplots[0].set_xlabel('number of nodes fit')
	subplots[0].set_ylabel('fraction of genes')
	# sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/how_many_genes_dist.pdf')
	sns.plt.show()



	genes = get_genes()
	gene_r_impact = []

	for node in range(400):
		if node in ignore_nodes:continue
		gene_r_impact.append(1 - fit_matrix('fc',topological,distance,node))
	gene_r_impact = np.nanmean(gene_r_impact,axis=0)

	g = np.nansum(c,axis=0)
	gorilla = genes[np.argsort(g)]
	sorted_g = g[np.argsort(g)]
	sorted_genes = genes[np.argsort(g)]
	gorilla[sorted_g==0] = sorted_genes[np.argsort(gene_r_impact[sorted_g==0])]
	gorilla = np.flip(gorilla,axis=0)
	np.savetxt('/home/mbmbertolero/data/gene_expression/results/gorilla_%s.txt'%(matrix),gorilla,fmt="%s")

	g = np.nanmean(c[pc>=cutoff],axis=0)
	pc_gorilla = genes[np.argsort(g)]
	sorted_g = g[np.argsort(g)]
	sorted_genes = genes[np.argsort(g)]
	pc_gorilla[sorted_g==0] = sorted_genes[np.argsort(gene_r_impact[sorted_g==0])]
	pc_gorilla = np.flip(pc_gorilla,axis=0)
	np.savetxt('/home/mbmbertolero/data/gene_expression/results/gorilla_pc_%s.txt'%(matrix),pc_gorilla,fmt="%s")

	g = np.nanmean(c[pc<cutoff],axis=0)
	local_gorilla = genes[np.argsort(g)]
	sorted_g = g[np.argsort(g)]
	sorted_genes = genes[np.argsort(g)]
	local_gorilla[sorted_g==0] = sorted_genes[np.argsort(gene_r_impact[sorted_g==0])]
	local_gorilla = np.flip(local_gorilla,axis=0)
	np.savetxt('/home/mbmbertolero/data/gene_expression/results/gorilla_local_%s.txt'%(matrix),local_gorilla,fmt="%s")

	g = pc_c[0] - pc_c[1]
	pc_gorilla = genes[np.argsort(g)]
	sorted_g = g[np.argsort(g)]
	sorted_genes = genes[np.argsort(g)]
	pc_gorilla[sorted_g==0] = sorted_genes[np.argsort(gene_r_impact[sorted_g==0])]
	pc_gorilla = np.flip(pc_gorilla,axis=0)
	np.savetxt('/home/mbmbertolero/data/gene_expression/results/gorilla_pc_spec_%s.txt'%(matrix),pc_gorilla,fmt="%s")

	g = pc_c[1] - pc_c[0]
	local_gorilla = genes[np.argsort(g)]
	sorted_g = g[np.argsort(g)]
	sorted_genes = genes[np.argsort(g)]
	local_gorilla[sorted_g==0] = sorted_genes[np.argsort(gene_r_impact[sorted_g==0])]
	local_gorilla = np.flip(local_gorilla,axis=0)
	np.savetxt('/home/mbmbertolero/data/gene_expression/results/gorilla_local_spec_%s.txt'%(matrix),local_gorilla,fmt="%s")

def convert_gorilla():
	files = glob.glob('/home/mbmbertolero/gene_expression/GOrilla/**')
	for f in files:
		os.system('mv %s %s'%(f,f[:-3]+ 'csv'))

def plot_fit_results(matrix,corr_method,topological=False,distance=True,return_df=False,n_genes=50):
	sns.set(context="notebook",font='Open Sans',style='white',font_scale=1.5)
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((400)).astype(str)
	small_membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	# yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	# yeo_colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.
	yeo_colors = pd.read_csv('/home/mbmbertolero/gene_expression/yeo_colors.txt',header=None,names=['name','r','g','b'],index_col=0)
	yeo_colors = yeo_colors.sort_values('name')
	yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default']
	network_colors = []
	for i,n in enumerate(yeo_df):
		small_membership[i] = names[int(reduce_dict[n.split('_')[2]])]
		# colors.append(yeo_colors[yeo_colors.name == small_membership[i]][['r','g','b']].values[0].astype(float)/256.)
		membership[i] = n.split('_')[2]
		membership_ints[i] = int(full_dict[n.split('_')[2]])
	membership = membership[:200]
	membership_ints = membership_ints[:200]
	small_membership = small_membership[:200]
	# colors = colors[:200]
	membership = small_membership
	if matrix == 'fc': other_matrix = 'sc'
	if matrix == 'sc': other_matrix = 'fc'

	df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_True_%s_False_fits_df.csv'%(matrix,corr_method))
	df = df.groupby('node').mean()
	df['network'] = membership[mask]
	df['connectivity'] = 'functional'
	# df = df[df.n_genes==n_genes]
	other_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_True_%s_False_fits_df.csv'%(other_matrix,corr_method))
	other_df = other_df.groupby('node').mean()
	# other_df = other_df[other_df.n_genes==n_genes]
	order = np.unique(membership)[np.argsort(df.groupby('network')['fit'].apply(np.nanmean))]
	colors = yeo_colors[np.argsort(df.groupby('network')['fit'].apply(np.nanmean)).values]
	other_df['network'] = membership[mask]
	other_df['connectivity'] = 'structural'
	# df = df.me

	print pearsonr(df.fit,other_df.fit)

	t,p=scipy.stats.ttest_ind(df.fit,other_df.fit)
	print t, np.log10(p)
	print '--------'
	stat_dict = {}
	for n in np.unique(df.network.values):
		t = scipy.stats.ttest_ind(df.fit[df.network==n],other_df.fit[other_df.network==n])
		print n, t
		stat_dict[n] = t

	# network_colors = []
	# for network in order:
	# 	network_colors.append(np.mean(colors[membership==network,:],axis=0))

	def get_axis_limits(ax):
		return ax.get_xlim()[0], ax.get_ylim()[1] + (ax.get_ylim()[1]*.1)

	fig,subplots = sns.plt.subplots(2,figsize=(7.204724,9.72440945))
	# fig,subplots = sns.plt.subplots(2)
	# yeo = sns.plt.imread('/home/mbmbertolero/gene_expression/yeo400.png')
	# s = subplots[0]
	# s.imshow(yeo,origin='upper')
	# s.set_xticklabels([])
	# s.set_yticklabels([])
	# sns.plt.draw()
	if matrix == 'fc': pal = sns.cubehelix_palette(10, rot=.25, light=.7)
	if matrix == 'sc': pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
	r,p = pearsonr(df['participation coefficient'],df.fit)
	# subplots[0].text(np.max(df.fit)*.9,np.max(df['participation coefficient'])*.9,'$\it{r}$=%s\n%s'%(np.around(r,2),log_p_value(p)))
	subplots[0].text(np.max(df.fit)*.9,np.max(df['participation coefficient'])*.9,'$\it{r}$=%s\n%s'%(np.around(r,2),log_p_value(p)))

	rplot = sns.regplot(data=df,y='participation coefficient',x='fit',color=pal[0],ax=subplots[0])
	subplots[0].set_xlabel('genetic fit')
	a = sns.violinplot(data=df,x='network',y='fit',palette=colors,order=order,ax=subplots[1],width=1)
	subplots[1].set_ylabel('mean genetic fit')
	subplots[1].set_ylim(.25,.9)
	for label in subplots[1].get_xmajorticklabels():
		label.set_rotation(90)
	for x,n in enumerate(order):
		t,p = stat_dict[n]
		a.text(x,.27,'$\it{t}$=%s\n%s' %(np.around(t,1),log_p_value(p)),{'fontsize':10},horizontalalignment='center')
	sns.despine()
	# s.spines['bottom'].set_visible(False)
	# s.spines['left'].set_visible(False)
	sns.plt.tight_layout()
	# sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/network_fits_and_pccorr_%s_%s.pdf'%(matrix,corr_method))
	sns.plt.show()

def compare_genetic_fits(corr_method='pearsonr',n_genes=50):

	sns.set(context="notebook",font='Open Sans',style='white')

	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.
	colors = colors[mask]

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]
		membership_ints[i] = int(full_dict[n.split('_')[2]])

	membership = membership[:200]
	membership_ints = membership_ints[:200]
	colors = colors[:200]

	sc_fits = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_True_%s_False_fits_df.csv'%(corr_method))
	sc_fits = sc_fits[sc_fits.n_genes==n_genes]
	sc_fits['network'] = 'structural connectivity'

	fc_fits = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_True_%s_False_fits_df.csv'%(corr_method))
	fc_fits = fc_fits[fc_fits.n_genes==n_genes]
	fc_fits['network'] = 'functional connectivity'
	df = fc_fits.append(sc_fits)

	#plot all the PC fits from sc and fc
	sc_fc_colors = colors.copy()
	sc_fc_colors[:len(colors)/2] = sns.color_palette("cubehelix", 8)[5]
	sc_fc_colors[len(colors)/2:] = sns.color_palette("cubehelix", 8)[6]
	line = np.mean([sns.color_palette("cubehelix", 8)[5],sns.color_palette("cubehelix", 8)[6]],axis=0)
	g = sns.regplot(data=df,y='fit',x='participation coefficient',scatter_kws={'facecolors':sc_fc_colors},line_kws={'color':line})
	# g.figure.set_size_inches(7.204724,7.204724)
	r = str(np.around(pearsonr(df['participation coefficient'],df.fit)[0],4))
	p = str(np.around(pearsonr(df['participation coefficient'],df.fit)[1],4))
	if p == '0.0': p = '<1-e5'
	sns.plt.text(np.max(df['participation coefficient'])*.9,np.max(df.fit)*.9,'r = %s\np= %s'%(r,p))
	sns.plt.ylabel('genetic fit')
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/fc_and_sc_pc_fits.pdf')
	sns.plt.show()

	#plot all the fits
	x,y = sc_fits['fit'], fc_fits['fit']
	g = sns.regplot(data=df,x=x,y=y,scatter_kws={'facecolors':colors},line_kws={'color':[0,0,0]})
	g.set_ylabel('functional connectivity genetic fit')
	g.set_xlabel('structural connectivity genetic fit')
	# g.figure.set_size_inches(7.204724,7.204724)
	r = pearsonr(x,y)[0]
	p = pearsonr(x,y)[1]
	if r > 0: xy=(.1, .87)
	else: xy=(.7, .87)
	g.annotate("r={:.3f}\np={:.3f}".format(r,p),xy=xy, xycoords=g.transAxes)
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/fc_and_sc_fits.pdf')
	sns.plt.show()
	sns.set(context="notebook",font='Open Sans',style='white',palette="muted")
	fig, subplots = sns.plt.subplots(2,2,figsize=(7.204724,9.72440945))
	subplots = subplots.flatten()
	for idx,measure in enumerate(['participation coefficient', 'within community strength', 'strength', 'betweenness']):
		# plot pc correlation between fc and sc
		x,y = sc_fits[measure], fc_fits[measure]
		# sns.regplot(x=x,y=y,ax=subplots[idx])
		sns.regplot(x=x,y=y,ax=subplots[idx])

		subplots[idx].set_ylabel('fc %s'%(measure))
		subplots[idx].set_xlabel('sc %s'%(measure))
		r = pearsonr(x,y)[0]
		p = pearsonr(x,y)[1]
		if r > 0: xy=(.1, .87)
		else: xy=(.7, .87)
		subplots[idx].annotate("r={:.3f}\np={:.3f}".format(r,p),xy=xy, xycoords=subplots[idx].transAxes)
		# subplots[idx].text(np.min(x)+np.min(x)*.9,np.max(y)*.9,'r = %s\np= %s'%(r,p))
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/fc_and_sc_metric_corr.pdf')
	sns.plt.show()

def regress(node):
	global features
	global measure
	global size
	sys.stdout.flush()
	model = Ridge(alpha=1.0)
	model.fit(features[np.arange(size)!=node],measure[np.arange(size)!=node])
	return model.predict(features[node].reshape(1, -1))

def cater(node):
	global features
	global measure
	global layers
	global size
	sys.stdout.flush()
	model = RidgeClassifier(alpha=1.0)
	model.fit(features[np.arange(size)!=node],measure[np.arange(size)!=node])
	return model.predict(features[node].reshape(1, -1))

def role_prediction(matrix,prediction='ge',corr_method='pearsonr',distance=True,topological=False,cores=40):
	global features
	global measure
	global layers
	global layers_name
	global size
	norm = True
	use_prs = False
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.


	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default']
	for i,n in enumerate(yeo_df):
		# membership[i] = n.split('_')[2]
		# membership_ints[i] = int(full_dict[n.split('_')[2]])
		membership_ints[i] = int(reduce_dict[n.split('_')[2]])
		membership[i] = names[int(reduce_dict[n.split('_')[2]])]

	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))[:200]
	degree = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))[:200]
	wcd = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))[:200]
	between = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))[:200]
	membership = membership[:200]
	membership_ints = membership_ints[:200]
	colors = colors[:200]


	if prediction == 'ge':
		unique = []
		genes_found = []
		df = pd.DataFrame(columns=['gene','node','participation coefficient','within community strength','strength','betweenness','network'])
		for node,name,pval,w,d,b,m in zip(range(200),membership,pc,wcd,degree,between,membership):
			if node in ignore_nodes: continue
			gene = np.array([])
			for n_genes in [15,25,35,50,75,100,125,150,175,200]:
				try:xgene = np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,node,n_genes,use_prs,norm,corr_method))[-1]
				except:
					print n_genes,node
					continue
				gene = np.append(gene,xgene)
				for g in xgene:
					unique.append(g)
			df= df.append(pd.DataFrame(np.array([[gene.flatten()],[node],[pval],[w],[d],[b],[m]]).transpose(),columns=['gene','node','participation coefficient','within community strength','strength','betweenness','network']))
		# df.gene = df.gene.astype(float)
		df['participation coefficient'] = df['participation coefficient'].astype(float)
		df['strength'] = df['strength'].astype(float)
		df['within community strength'] = df['within community strength'].astype(float)
		df['betweenness'] = df['betweenness'].astype(float)
		df.node = df.node.values.astype(int)
		unique = np.unique(unique)
		features = np.zeros((200,len(unique)))
		for node in df.node.values:
			g_array = unique.copy()
			g_b_array = unique.copy().astype(int)
			g_b_array[:] = 0
			for ga in df.gene[df.node==node].values:
				for g in ga:
					g_b_array[g_array==g] = g_b_array[g_array==g]  + 1
			features[node,:] = g_b_array
		features = features.astype(int)
		mask = np.ones((200)).astype(bool)
		mask[ignore_nodes] = False
		membership = membership[mask]
		features = features[mask]
		# df = df.groupby('node').mean() #this might break this but I don't care because I alreay ran this
		
	if prediction == 'snp':
		df = pd.DataFrame(columns=['gene','node','participation coefficient','within community strength','strength','betweenness','network'])
		df['node'] = np.arange(200)
		df['participation coefficient'] = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))[:200]
		df['strength'] = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))[:200]
		df['within community strength'] = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))[:200]
		df['betweenness'] = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))[:200]
		gene_names = np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')
		features = np.zeros((200,len(gene_names)))
		features[:,:] = np.nan
		for node in range(200):
			features[node] = np.load('/home/mbmbertolero/data/gene_expression/snp_results/%s_%s_%s_gene_snps_mean.npy'%(components,node,matrix))
		features= features[:,np.isnan(features)[0]==False]
	size = features.shape[0]
	g_names = np.load('/home/mbmbertolero/data//gene_expression/gene_names.npy')
	# np.savetxt('/home/mbmbertolero/gene_expression/ann/masked_names_%s.txt'%(matrix),np.array(g_names)[unique],fmt='%s')
	# 1/0


	"""
	predict the nodal values of a node based on which genes maximize its fit to FC
	"""

	for name in ['participation coefficient','betweenness','strength','within community strength']:
		measure = np.array(df.groupby('node').mean()[name].values)
		# measure = df[name]
		pool = Pool(cores)
		prediction_array = np.array(pool.map(regress,range(size)))[:,0]
		del pool
		print pearsonr(prediction_array,measure)
		np.save('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_%s_%s.npy'%(matrix,name.replace(' ','_'),corr_method,prediction),prediction_array)
		np.save('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_%s_%s_true.npy'%(matrix,name.replace(' ','_'),corr_method,prediction),measure)

	measure = membership_ints
	if prediction == 'ge': measure = membership_ints[mask]

	# prediction_array = np.zeros((size))
	# for i,node in enumerate(range(size)):
	# 	model = RidgeClassifier(alpha=1.0)
	# 	model.fit(features[np.arange(size)!=node],measure[np.arange(size)!=node])
	# 	prediction_array[i] = model.predict(features[node].reshape(1, -1))


	pool = Pool(cores)
	prediction_array = np.array(pool.map(cater,range(size)))[:,0]
	# print len(prediction_array[prediction_array==measure])
	# print f1_score(measure,prediction_array,average='samples')
	np.save('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_%s_%s.npy'%(matrix,'networks',corr_method,prediction),prediction_array)

	df = pd.DataFrame(np.array([precision_recall_fscore_support(measure,prediction_array)[2],names]).transpose(),columns=['f-beta','network'])

def plot_role_prediction(matrix = 'fc',prediction_t='snp'):
	sns.set(context="notebook",font='Open Sans',style='white')
	distance = True
	topological = False
	prediction = 'genetic'
	reduce_membership = False

	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':6}
	full_dict = {'VisCent':0,'VisPeri':1,'SomMotA':2,'SomMotB':3,'DorsAttnA':4,'DorsAttnB':5,'SalVentAttnA':6,'SalVentAttnB':7,'Limbic':8,'ContA':9,'ContB':10,'ContC':11,'DefaultA':12,'DefaultB':13,'DefaultC':14,'TempPar':15}
	membership = np.zeros((400)).astype(str)
	membership_ints = np.zeros((400)).astype(int)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,] /256.

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']
	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]
		membership_ints[i] = int(reduce_dict[n.split('_')[2]])

	membership = membership[:200]
	membership_ints = membership_ints[:200]
	colors = colors[:200]

	mask = np.ones((200)).astype(bool)
	mask[ignore_nodes] = False



	if prediction_t == 'ge':

		pc = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_true.npy'%(matrix,'participation_coefficient'))
		strength = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_true.npy'%(matrix,'strength'))
		wcd = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_true.npy'%(matrix,'within_community_strength'))
		between = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_true.npy'%(matrix,'betweenness'))

		pc_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr.npy'%(matrix,'participation_coefficient'))
		strength_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr.npy'%(matrix,'strength'))
		wcd_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr.npy'%(matrix,'within_community_strength'))
		between_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr.npy'%(matrix,'betweenness'))
		prediction_array = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_networks_pearsonr.npy'%(matrix))
	
	if prediction_t == 'snp':

		pc = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp_true.npy'%(matrix,'participation_coefficient'))
		strength = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp_true.npy'%(matrix,'strength'))
		wcd = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp_true.npy'%(matrix,'within_community_strength'))
		between = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp_true.npy'%(matrix,'betweenness'))
		pc_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp.npy'%(matrix,'participation_coefficient'))
		strength_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp.npy'%(matrix,'strength'))
		wcd_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp.npy'%(matrix,'within_community_strength'))
		between_p = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_%s_pearsonr_snp.npy'%(matrix,'betweenness'))
		prediction_array = np.load('/home/mbmbertolero/data/gene_expression/norm_results/genetic_%s_networks_pearsonr_snp.npy'%(matrix))

	df = pd.DataFrame(columns=['node','nodal metric','observed','prediction'])
	for i in range(len(pc)):
		df = df.append(pd.DataFrame(np.array([[i],['participation coefficient'],[pc[i]],[pc_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))
		df = df.append(pd.DataFrame(np.array([[i],['strength'],[strength[i]],[strength_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))
		df = df.append(pd.DataFrame(np.array([[i],['within community strength'],[wcd[i]],[wcd_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))
		df = df.append(pd.DataFrame(np.array([[i],['betweenness'],[between[i]],[between_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))

	df.observed = df.observed.values.astype(float)
	df.prediction = df.prediction.values.astype(float)
	order = np.unique(df['nodal metric'])
	df = df.sort_values('nodal metric')
	if matrix == 'fc': pal = sns.cubehelix_palette(4, rot=.25, light=.7)
	if matrix == 'sc': pal = sns.cubehelix_palette(4, rot=-.25, light=.7)

	g = sns.lmplot(x='observed',y='prediction',col='nodal metric',data=df,sharex=False,sharey=False,col_wrap=2,palette=pal,hue='nodal metric',size=(7.204724/2))
	for i,m in enumerate(g.hue_names):
		p = pearsonr(df['prediction'][df['nodal metric']==m].values,df['observed'][df['nodal metric']==m].values)
		r,p = p[0],p[1]

		g.axes[i].annotate('$\it{r}$=%s\n%s'%(np.around(r,2),log_p_value(p)),xy=(0.7, 0.1), xycoords=g.axes[i].transAxes)
		g.axes[i].set_title(order[i],color=pal[i])
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/new_figures/predict_%s_with_%s_%s.pdf'%(matrix,prediction, prediction_t))
	sns.plt.show()

	yeo_colors = pd.read_csv('/home/mbmbertolero/gene_expression/yeo_colors.txt',header=None,names=['name','r','g','b'],index_col=0)
	yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.
	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default']

	

	write_cifti_prediction = np.zeros((400,3))
	write_cifti_colors = []
	p_i = 0
	for i in range(200):
		# if mask[i] == False:
		# 	write_cifti_colors.append([201/256.,201/256.,201/256.])
		# 	continue
		# if mask[i] == True:
		# 	write_cifti_colors.append(yeo_colors[prediction_array[p_i]])
		# 	p_i =  p_i  + 1
		write_cifti_colors.append(yeo_colors[prediction_array[i]])
	write_cifti_prediction[:200] = write_cifti_colors
	write_cifti('/home/mbmbertolero/data/PUBLIC/yeo/fsLR32k/Schaefer2016_400Parcels_17Networks_colors_23_05_16.dlabel.nii','/home/mbmbertolero/gene_expression/new_figures/predict_network_%s_%s'%(matrix,prediction_t),write_cifti_prediction)
	# print len(prediction_array[prediction_array==membership_ints[mask]])
	write_cifti_real = np.zeros((400,3))
	write_cifti_colors = []
	for i in range(200):
		# if mask[i] == False:
		# 	write_cifti_colors.append([201/256.,201/256.,201/256.])
		# 	continue
		# if mask[i] == True:
		write_cifti_colors.append(yeo_colors[membership_ints[i]])
	write_cifti_real[:200] = write_cifti_colors
	write_cifti('/home/mbmbertolero/data/PUBLIC/yeo/fsLR32k/Schaefer2016_400Parcels_17Networks_colors_23_05_16.dlabel.nii','/home/mbmbertolero/gene_expression/new_figures/real_network_%s'%(matrix),write_cifti_real)
	"""
	# image = np.ones((7,51,3))
	# # image[ignore_nodes] = [1,1,1]
	# for i in range(7):
	# 	for pidx,p in enumerate(prediction_array[measure==i]):
	# 		print pidx
	# 		image[i,pidx] = yeo_colors[p]
	"""

def task_performance(subjects,task):
	# df = pd.read_csv('//home/mbmbertolero//S900_Release_Subjects_Demographics.csv')
	df = pd.read_csv('//home/mbmbertolero/hcp/S1200.csv')
	performance = []
	if task == 'EMOTION':
		emotion_df = pd.DataFrame(np.array([df.Subject,df['Emotion_Task_Median_RT'],df['Emotion_Task_Acc']]).transpose(),columns=['Subject','RT','ACC']).dropna()
		emotion_df['RT'] = emotion_df['RT'].values * -1
		emotion_df['RT'] = scipy.stats.zscore(emotion_df['RT'].values)
		emotion_df['ACC'] = emotion_df['ACC'].values * emotion_df['RT'].values
		for subject in subjects:
			temp_df = emotion_df[emotion_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(temp_df['ACC'].values[0])
	if task == 'WM':
		wm_df = pd.DataFrame(np.array([df.Subject.values,df['WM_Task_Acc'].values,df['WM_Task_Median_RT'].values]).transpose(),columns=['Subject','ACC','RT']).dropna()
		wm_df['RT'] = wm_df['RT'].values * -1
		wm_df['RT'] = scipy.stats.zscore(wm_df['RT'].values)
		for subject in subjects:
			temp_df = wm_df[wm_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(temp_df['ACC'].values[0])
	if task == 'RELATIONAL':
		for subject in subjects:
			performance.append(df['Relational_Task_Acc'][df.Subject == subject].values[0])
	if task == 'LANGUAGE':
		for subject in subjects:
			performance.append(np.nanmax([df['Language_Task_Story_Avg_Difficulty_Level'][df.Subject == subject].values[0],df['Language_Task_Math_Avg_Difficulty_Level'][df.Subject == subject].values[0]]))
	if task == 'SOCIAL':
		social_df = pd.DataFrame(np.array([df.Subject,df['Social_Task_TOM_Perc_TOM'],df['Social_Task_Random_Perc_Random']]).transpose(),columns=['Subject','ACC_TOM','ACC_RANDOM']).dropna()
		for subject in subjects:
			temp_df = social_df[social_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(np.nanmean([temp_df['ACC_RANDOM'].values[0],temp_df['ACC_TOM'].values[0]]))
	return np.array(performance)

def adult_changes():
	try:
		changes = np.load('/home/mbmbertolero/gene_expression/results/adult_changes.npy')
	except:
		changes = np.zeros((400,400))
		ms = []
		files = glob.glob('/home/mbmbertolero/gene_expression/data/matrices/*fc_matrix*')
		for f in files:
			m = np.load(f)
			ms.append(m)
		ms = np.array(ms)
		for i,j in combinations(range(400),2):
			r = np.var(ms[:,i,j])
			changes[i,j] = r
			changes[j,i] = r
		np.save('/home/mbmbertolero/gene_expression/results/adult_changes.npy',changes)
	return changes

def get_dev_df(task='restingstate'):
	behav = pd.read_csv('//data/joy/BBL-extend/share/pncNbackNmf/subjectData/n1601_cnb_factor_scores_tymoore_20151006.csv')
	df = pd.read_csv('//data/joy/BBL-extend/share/pncNbackNmf/subjectData/n1601_demographics_go1_20161212.csv')
	df = pd.merge(df,behav,on='scanid')
	if task == 'restingstate' or task == 'both':
		m_file = pd.read_csv('//data/joy/BBL-extend/share/pncNbackNmf/subjectData/n1601_RestQAData_20170509.csv')
		df = pd.merge(df,m_file,on='scanid',how='left')
	if task == 'nback' or task == 'both':
		m_file = pd.read_csv('//data/joy/BBL-extend/share/pncNbackNmf/subjectData/n1601_NbackConnectQAData_20170718.csv')
		df = pd.merge(df,m_file,on='scanid',how='left')

	# df['motion'] = np.zeros(df.shape[0])
	# for i,s in enumerate(df.scanid.values):
	# 	df['motion'][i] = m_file['restRelMeanRMSMotion'][m_file.scanid==s]
	#sort by age
	df.sort_values('ageAtScan1',inplace=True)
	subjects = df.scanid.values
	for s in subjects:
		if task == 'both':
			rs = os.path.exists('/home/mbmbertolero/gene_expression/pnc_matrices/%s_%s_matrix.npy'%(s,'restingstate'))
			nb = os.path.exists('/home/mbmbertolero/gene_expression/pnc_matrices/%s_%s_matrix.npy'%(s,'nback'))
			if rs == False or nb == False: df = df[df.scanid!=s]
		else:
			if os.path.exists('/home/mbmbertolero/gene_expression/pnc_matrices/%s_%s_matrix.npy'%(s,task)) == False: df = df[df.scanid!=s]
	df = df.dropna()
	return df

def make_developmental_matrices(task='restingstate'):
	# /data/joy/BBL-extend/share/pncNbackNmf/nbackNetwork_Schaefer2017 (n-back without regression of task model)
	# /data/joy/BBL-extend/share/pncNbackNmf/nbackRegNetwork_Schaefer2017 (n-back with regression of task model)
	# /data/joy/BBL-extend/share/pncNbackNmf/restNetwork_Schaefer2017 (resting)
	if task == 'restingstate': files = glob.glob('//data/joy/BBL-extend/share/pncNbackNmf/restNetwork_Schaefer2017/**')
	if task == 'nback': files = glob.glob('/data/joy/BBL-extend/share/pncNbackNmf/nbackNetwork_Schaefer2017/**')
	for f in files:
		print f
		m = np.loadtxt(f)
		sname = f.split('/')[-1].split('_')[0]
		m = np.corrcoef(m.transpose())
		np.save('/home/mbmbertolero/gene_expression/pnc_matrices/%s_%s_matrix.npy'%(sname,task),m)

def dev_graph(subject):
	"""
	run graph metrics or load them
	"""
	try:
		pc = np.load('/home/mbmbertolero/gene_expression/data//results/%s_%s_pcs.npy' %(subject,matrix))
		wmd = np.load('/home/mbmbertolero/gene_expression/data//results/%s_%s_wmds.npy' %(subject,matrix))
		mod = np.load('/home/mbmbertolero/gene_expression/data/results/%s_%s_mods.npy' %(subject,matrix))
		m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_%s_matrix.npy' %(subject,matrix))
	except:
		print 'running subject %s' %(subject)
		m = np.load('/home/mbmbertolero/gene_expression/pnc_matrices/%s_%s_matrix.npy'%(subject,'restingstate'))
		m = m + m.transpose()
		m = np.tril(m,-1) + np.tril(m,-1).transpose()
		num_nodes = m.shape[0]
		pc = []
		mod = []
		wmd = []
		for cost in np.linspace(1,10)*.01:
			temp_matrix = m.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',mst=True)
			del temp_matrix
			graph = graph.community_infomap(edge_weights='weight')
			graph = brain_graphs.brain_graph(graph)
			pc.append(np.array(graph.pc))
			wmd.append(np.array(graph.wmd))
			mod.append(graph.community.modularity)
			del graph
		pc = np.nanmean(pc,axis=0)
		wmd = np.nanmean(wmd,axis=0)
		mod = np.nanmean(mod)
		np.save('/home/mbmbertolero/gene_expression/data/results/%s_%s_pcs.npy' %(subject,matrix),pc)
		np.save('/home/mbmbertolero/gene_expression/data/results/%s_%s_wmds.npy' %(subject,matrix),wmd)
		np.save('/home/mbmbertolero/gene_expression/data/results/%s_%s_mods.npy' %(subject,matrix),mod)
		np.save('/home/mbmbertolero/gene_expression/data/matrices/%s_%s_matrix.npy' %(subject,matrix),m)
	return pc,wmd,mod,m

def get_dev_graphs(subjects,matrix):
	mods = []
	pcs = []
	wmds = []
	matrices = []
	for subject in subjects:
		subject_pcs,subject_wmds,subject_mods,m = dev_graph(subject)
		pcs.append(subject_pcs)
		wmds.append(subject_wmds)
		mods.append(subject_mods)
		matrices.append(m)
	mods = np.array(mods)
	pcs = np.array(pcs)
	wmds = np.array(wmds)
	matrices = np.array(matrices)
	results = {}
	results['subject_pcs'] = pcs
	results['subject_mods'] = mods
	results['subject_wmds'] = wmds
	results['matrices'] = matrices
	return results

def dev_matrices(subjects,task = 'restingstate',distance=True):

	df = pd.read_csv('//data/joy/BBL-extend/share/pncNbackNmf/subjectData/n1601_demographics_go1_20161212.csv')
	ages = []
	ms = []
	distance_matrix = atlas_distance()
	for s in subjects:
		m = np.load('/home/mbmbertolero/gene_expression/pnc_matrices/%s_%s_matrix.npy'%(s,task))
		if distance:
			np.fill_diagonal(m,np.nan)
			m[np.isnan(m)==False] = sm.GLM(m[np.isnan(m)==False],sm.add_constant(distance_matrix[np.isnan(m)==False])).fit().resid_response
		ms.append(m)
		ages.append(df.ageAtScan1[df.scanid==s].values[0])
	ms = np.array(ms)
	return ages,ms

def dev_sc_matrices(distance=True):
	distance_matrix = atlas_distance()
	ages = pd.read_csv('/data/joy/BBL-extend/share/forMax/n914_dti_demogs.csv')['ageAtScan1'].values
	ms = []
	sc_ms = np.loadtxt('/data/joy/BBL-extend/share/forMax/ptx_edgeVec_connProbability_Schaefer400_n914.txt',dtype='str')
	for m in sc_ms:
		m = scipy.spatial.distance.squareform(np.array(m.split(',')).astype(float))
		if distance:
			np.fill_diagonal(m,np.nan)
			m[np.isnan(m)==False] = sm.GLM(m[np.isnan(m)==False],sm.add_constant(distance_matrix[np.isnan(m)==False])).fit().resid_response
		ms.append(m)
	ms = np.array(ms)
	return ages,ms

def dev_changes(matrix,task='restingstate',distance=True,method='slope',age_limit=13):
	if matrix == 'fc':
		if os.path.exists('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_changes.npy'%(task,distance,method,age_limit)) == True:
			# assert (subjects == np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_subjects_changes.npy'%(task,distance,method))).all()
			changes = np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_changes.npy'%(task,distance,method,age_limit))
		else:
			df = get_dev_df(task)
			subjects = df.scanid.values
			changes = np.zeros((400,400))
			ages,ms = dev_matrices(subjects,task,distance)
			thresh = (np.array(ages)/12.).astype(int)
			ages = np.array(ages)[thresh<=age_limit]
			ms = np.array(ms)[thresh<=age_limit]
			if task =='restingstate': model_vars = np.array([df['restRelMeanRMSMotion'].values][thresh<=age_limit]).transpose()
			if task == 'nback': model_vars = np.array([df['nbackRelMeanRMSMotion'][thresh<=age_limit].values]).transpose()
			ages_r = sm.GLM(ages,sm.add_constant(model_vars)).fit().resid_response
			for i,j in combinations(range(400),2):
				if method == 'spearman': r = pearsonr(ages_r,ms[:,i,j])[0]
				if method == 'var': r = np.var(ms[:,i,j])
				if method == 't_test': r = scipy.stats.ttest_ind(ms[:100,i,j],ms[-100:,i,j])[0]
				if method == 'slope': r = linregress(ages_r,ms[:,i,j])[0]
				changes[i,j] = r
				changes[j,i] = r
			np.save('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_changes.npy'%(task,distance,method,age_limit),changes)
			np.save('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_subjects_changes.npy'%(task,distance,method,age_limit),subjects)
		return changes
	if matrix == 'sc':
		distance_matrix = atlas_distance()
		if os.path.exists('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_sc_changes.npy'%('sc',distance,method,age_limit)) == True:
			changes = np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_sc_changes.npy'%('sc',distance,method,age_limit))
		else:
			ages,ms = dev_sc_matrices(distance)
			changes = np.zeros((400,400))
			thresh = (np.array(ages)/12.).astype(int)
			ages = np.array(ages)[thresh<=age_limit]
			ms = np.array(ms)[thresh<=age_limit]
			motion = pd.read_csv('/data/joy/BBL-extend/share/forMax/n914_dti_demogs.csv')['dti64MeanAbsRMS'].values[thresh<=age_limit]
			model_vars = np.array([motion]).transpose()
			ages_r = sm.GLM(ages,sm.add_constant(model_vars)).fit().resid_response
			assert np.isclose(pearsonr(ages_r,motion)[0],0.0)
			for i,j in combinations(range(400),2):
				if method == 'spearman': r = pearsonr(ages_r,ms[:,i,j])[0]
				if method == 'var': r = np.var(ms[:,i,j])
				if method == 't_test': r = scipy.stats.ttest_ind(ms[:100,i,j],ms[-100:,i,j])[0]
				if method == 'slope': r = linregress(ages_r,ms[:,i,j])[0]
				changes[i,j] = r
				changes[j,i] = r
			np.save('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_sc_changes.npy'%('sc',distance,method,age_limit),changes)
		return changes

def full_gene_coexpression(matrix,topological,distance,n_genes=50):
	final_matrix = np.zeros((200,200))
	for node in range(200):
		if node in ignore_nodes: continue
		# if n_genes == 100: gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/orig_results/SA_fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,node))[-1]
		gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,topological,distance,node,n_genes,False,True,'pearsonr'))[-1]
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
		assert (final_matrix[node:,] == 0.0).all()
		final_matrix[node,:] = gene_exp_matrix[node,:]
	np.fill_diagonal(final_matrix,np.nan)
	return final_matrix

def dev_analyses(matrix='fc',task='nback'):
	sns.set(context="notebook",font='Open Sans',style='white')
	distance = False
	task ='nback'
	# get behavioral metrics
	if matrix == 'fc':
		df = get_dev_df(task=task)
		subjects = df.scanid.values
		if task =='both':
			rs_ages,rs_ms = dev_matrices(subjects,'restingstate',distance)
			nb_ages,nb_ms = dev_matrices(subjects,'nback',distance)
			assert (np.array(rs_ages) == np.array(nb_ages)).all()
			ms = np.zeros(rs_ms.shape)
			ages = nb_ages
			for i in range(ms.shape[0]):
				ms[i] = np.nanmean([nb_ms[i],rs_ms[i]],axis=0)
		if task == 'restingstate':
			df['motion'] = df['restRelMeanRMSMotion'].values
			ages,ms = dev_matrices(subjects,'restingstate',distance)
		if task == 'nback':
			df['motion'] = df['nbackRelMeanRMSMotion'].values
			ages,ms = dev_matrices(subjects,'nback',distance)
		if task == 'both': df['motion'] = np.nanmean([df['nbackRelMeanRMSMotion'].values,df['restRelMeanRMSMotion'].values],axis=0)

	if matrix == 'sc':
		df = pd.read_csv('/data/joy/BBL-extend/share/forMax/n914_dti_demogs.csv')
		df['motion'] = pd.read_csv('/data/joy/BBL-extend/share/forMax/n914_dti_demogs.csv')['dti64MeanRelRMS']
		subjects = df.scanid.values
		ages,ms = dev_sc_matrices(distance)
		behav = pd.read_csv('//data/joy/BBL-extend/share/pncNbackNmf/subjectData/n1601_cnb_factor_scores_tymoore_20151006.csv')
		df = pd.merge(df,behav,on='scanid')

	#make a matrix of what gene-coexpression is
	final_matrix = []
	for n_genes in [50,75,100,125,150,175,200]:
		final_matrix.append(full_gene_coexpression(matrix,False,True,n_genes))
	final_matrix = np.nanmean(final_matrix,axis=0)
	#see how each kids matrix fits the gene coexpression matrix
	age_fits = []
	for i in range(ms.shape[0]):
		age_fits.append(nan_pearsonr(ms[i][:200,:200].flatten(),final_matrix.flatten())[0])

	colors = np.array(sns.husl_palette(6, s=.45))
	# if matrix == 'sc':
	# 	new_colors = colors.copy()
	# 	new_colors[4] = colors[3]
	# 	new_colors[3] = colors[4]
	# 	colors = new_colors
	if matrix == 'fc':
		c = sns.cubehelix_palette(10, rot=.25, light=.7)[0]
		matrix_name = 'functional'
	else:
		c = sns.cubehelix_palette(10, rot=-.25, light=.7)[0]
		matrix_name = 'structural'

	# #how does fit relate to age?
	model_vars = np.array([df['motion'].values]).transpose()
	age_fits_r = sm.GLM(age_fits,sm.add_constant(model_vars)).fit().resid_response
	#how does age relate to fit?
	fit_df = pd.DataFrame()
	fit_df['fit'] = age_fits_r
	fit_df['age'] = np.array(ages)/12.
	fit_df = fit_df[fit_df.age<=13]
	ax = sns.regplot(data=fit_df,y='fit',x='age',order=2,color=c)
	fit_df['age'] = fit_df['age'].values.astype(int)
	sns.plt.scatter(np.unique(fit_df.age.values),fit_df.groupby('age').mean().values,color='black')
	sns.plt.xlabel('age')
	sns.plt.ylabel('fit to gene coexpression')
	r,p = nan_pearsonr(x=fit_df.fit,y=fit_df.age)
	sns.plt.text(0.8, 0.2,convert_r_p(r,p), horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/age_to_fit_to_adult_gene_coexpression_%s_%s.pdf'%(matrix,task))
	sns.plt.close()


	#how does fit relate to executive function?
	model_vars = np.array([ages,df['motion'].values]).transpose()
	age_fits_r = sm.GLM(age_fits,sm.add_constant(model_vars)).fit().resid_response
	y = df['F1_Exec_Comp_Res_Accuracy']
	x = age_fits_r[np.isnan(y)==False]
	y = y[np.isnan(y)==False]
	y = y[:379] # 13 and under
	x = x[:379] # 13 and under
	# y = y[379:] # 14 and over
	# x = x[379:] # 14 and over
	ax = sns.regplot(x,y,order=1,scatter_kws={'facecolors':[0,0,0],'edgecolors':[0,0,0],'alpha':.5},line_kws={'color':[0,0,0]})
	sns.plt.ylabel('executive function')
	sns.plt.xlabel('fit, age + motion regressed')
	r,p = pearsonr(x,y)
	sns.plt.text(0.1, 0.9,convert_r_p(r,p), horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/fit_corr_executive_reg_%s_%s.pdf'%(matrix,task))
	sns.plt.close()

	# how do dev changes relate to PC
	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
	#get matrices/ages of the kids
	changes = dev_changes(matrix,task,distance,'spearman',13)
	np.fill_diagonal(changes,np.nan)

	pos_changes = changes.copy()
	pos_changes[pos_changes<0.0] = np.nan
	neg_changes = changes.copy()
	neg_changes[neg_changes>0.0] = np.nan

	fc_node_by_snp = np.load('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_fc.npy')
	sc_node_by_snp = np.load('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_sc.npy')


	fc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s_%s_fits_df.csv'%(matrix,True,'pearsonr',False))

	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	ax = sns.regplot(y=fc_fit_df.groupby('node').fit.mean(),x=np.nanmean(np.abs(changes),axis=0)[:200][mask],
		scatter_kws={'facecolors':colors[5],'edgecolors':colors[5],'alpha':.5},line_kws={'color':colors[5]})
	sns.plt.ylabel('gene-coexpression fit in adults')
	sns.plt.xlabel('mean absolute nodal developmental\nchange in %s connectivity'%(matrix_name))
	r,p = nan_pearsonr(y=fc_fit_df.groupby('node').fit.mean(),x=np.nanmean(np.abs(changes),axis=0)[:200][mask])
	sns.plt.text(0.8, 0.2,convert_r_p(r,p), horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/dev_fit_adult_gene_fit_all_%s_%s.pdf'%(matrix,task))
	sns.plt.close()

	colors = sns.husl_palette(20, s=.45)
	ax = sns.regplot(y=fc_fit_df.groupby('node').fit.mean(),x=np.nanmean(np.abs(neg_changes),axis=0)[:200][mask],
		scatter_kws={'facecolors':colors[14],'edgecolors':colors[14],'alpha':.5},line_kws={'color':colors[14]})
	ax2 = ax.twinx()
	sns.regplot(y=np.nanmean(fc_node_by_snp,axis=1),x=np.nanmean(np.abs(neg_changes),axis=0)[:200],
		scatter_kws={'facecolors':colors[15],'edgecolors':colors[15],'alpha':.5},line_kws={'color':colors[15]},ax=ax2)
	ax2.set_xlim(0.0325,0.084)
	ax.set_ylabel('genetic fit in adults',color=colors[14],labelpad=-1)
	ax2.set_ylabel('genetically explained\nconnectivity variance',rotation=-90,color=colors[15],labelpad=24)
	ax.set_xlabel('nodal developmental decrease\nin %s connectivity'%(matrix_name))
	r,p = nan_pearsonr(y=fc_fit_df.groupby('node').fit.mean(),x=np.nanmean(np.abs(neg_changes),axis=0)[:200][mask])
	sns.plt.text(0.8, 0.2,convert_r_p(r,p),color=colors[14],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	r,p = nan_pearsonr(y=np.nanmean(fc_node_by_snp,axis=1),x=np.nanmean(np.abs(neg_changes),axis=0)[:200])
	sns.plt.text(0.6, 0.2,convert_r_p(r,p),color=colors[15],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine(top=True,right=False)
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/dev_fit_adult_gene_fit_neg_%s_%s.pdf'%(matrix,task))
	sns.plt.close()

	ax = sns.regplot(y=fc_fit_df.groupby('node').fit.mean(),x=np.nanmean(np.abs(pos_changes),axis=0)[:200][mask],
		scatter_kws={'facecolors':colors[17],'edgecolors':colors[17],'alpha':.5},line_kws={'color':colors[17]})
	ax2 = ax.twinx()
	sns.regplot(y=np.nanmean(fc_node_by_snp,axis=1),x=np.nanmean(np.abs(pos_changes),axis=0)[:200],
		scatter_kws={'facecolors':colors[19],'edgecolors':colors[19],'alpha':.5},line_kws={'color':colors[19]},ax=ax2)
	ax2.set_xlim(0.0325,0.084)
	ax.set_ylabel('genetic fit in adults',color=colors[17],labelpad=-1)
	ax2.set_ylabel('genetically explained\nconnectivity variance',rotation=-90,color=colors[19],labelpad=24)
	ax.set_xlabel('nodal developmental increase\nin %s connectivity'%(matrix_name))
	r,p = nan_pearsonr(y=fc_fit_df.groupby('node').fit.mean(),x=np.nanmean(np.abs(pos_changes),axis=0)[:200][mask])
	sns.plt.text(0.8, 0.2,convert_r_p(r,p),color=colors[17],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	r,p = nan_pearsonr(y=np.nanmean(fc_node_by_snp,axis=1),x=np.nanmean(np.abs(pos_changes),axis=0)[:200])
	sns.plt.text(0.6, 0.1,convert_r_p(r,p),color=colors[19],horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine(top=True,right=False)
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/dev_fit_adult_gene_fit_pos_%s_%s.pdf'%(matrix,task))
	sns.plt.close()

	ax=sns.regplot(y=pc,x=np.nanmean(np.abs(pos_changes),axis=0),scatter_kws={'facecolors':colors[0],'edgecolors':colors[0],'alpha':.5},line_kws={'color':colors[0]})
	sns.plt.ylabel('adult participation coefficient')
	sns.plt.xlabel('nodal developmental increase\nin %s connectivity'%(matrix_name))
	r,p = nan_pearsonr(y=pc,x=np.nanmean(np.abs(pos_changes),axis=0))
	sns.plt.text(0.8, 0.2,convert_r_p(r,p),  horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/dev_fit_pc_pos_%s_%s.pdf'%(matrix,task))
	sns.plt.close()

	ax=sns.regplot(y=pc,x=np.nanmean(np.abs(neg_changes),axis=0),scatter_kws={'facecolors':colors[4],'edgecolors':colors[4],'alpha':.5},line_kws={'color':colors[4]})
	sns.plt.ylabel('adult participation coefficient')
	sns.plt.xlabel('nodal developmental decrease\nin %s connectivity'%(matrix_name))
	r,p = nan_pearsonr(y=pc,x=np.nanmean(np.abs(neg_changes),axis=0))
	sns.plt.text(0.8, 0.2,convert_r_p(r,p), horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/dev_fit_pc_neg_%s_%s.pdf'%(matrix,task))
	sns.plt.close()

	ax = sns.regplot(y=pc,x=np.nanmean(np.abs(changes),axis=0),scatter_kws={'facecolors':colors[5],'edgecolors':colors[5],'alpha':.5},line_kws={'color':colors[5]})
	sns.plt.ylabel('adult participation coefficient')
	sns.plt.xlabel('mean absolute nodal developmental\nchange in %s connectivity'%(matrix_name))
	r = nan_pearsonr(y=pc,x=np.nanmean(np.abs(changes),axis=0))[0]
	sns.plt.text(0.8, 0.2,convert_r_p(r,p), horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/dev_fit_pc_all_%s_%s.pdf'%(matrix,task))
	# sns.plt.show()
	sns.plt.close()

def get_graphs(subjects,matrix):
	mods = []
	pcs = []
	wmds = []
	matrices = []
	for subject in subjects:
		subject_pcs,subject_wmds,subject_mods,m = graph_metrics(subject,matrix)
		pcs.append(subject_pcs)
		wmds.append(subject_wmds)
		mods.append(subject_mods)
		matrices.append(m)
	mods = np.array(mods)
	pcs = np.array(pcs)
	wmds = np.array(wmds)
	matrices = np.array(matrices)
	results = {}
	results['subject_pcs'] = pcs
	results['subject_mods'] = mods
	results['subject_wmds'] = wmds
	results['matrices'] = matrices
	return results

def get_subjects(done=False):
	# rest_subjects = []
	# for subject in np.loadtxt("/home/mbmbertolero/hcp/scripts/rfMRI/subject_lists/subject_runnum_list/all_new_subject_list_sort_post_motion.txt").astype(int):
	# 	fmri_file = '/home/mbmbertolero/hcp/scripts/rfMRI/subject_lists/allsub_MSM_ICA_FIX_fMRI_list/%s_rfMRI_list_post_motion.txt'%(subject)
	# 	if len(pd.read_csv(fmri_file,header=None)[0]) == 4: rest_subjects.append(subject)
	# rest_subjects = np.array(rest_subjects)
	df = pd.read_csv('/home/mbmbertolero/hcp/S1200.csv')
	bedpost_subjects = df.Subject[df['3T_dMRI_PctCompl'].values==100].values
	rest_subjects = df.Subject[df['3T_RS-fMRI_Count'].values==4].values
	subjects = np.intersect1d(bedpost_subjects,rest_subjects)
	if done == 'matrices':
		for i,s in enumerate(subjects):
			if os.path.exists('/home/mbmbertolero/gene_expression/data/matrices/%s_fc_matrix.npy' %(s)) == False \
			or os.path.exists('/home/mbmbertolero/gene_expression/data/matrices/%s_sc_matrix.npy' %(s)) == False:
				subjects = np.delete(subjects,np.where(subjects == s))
	if done == 'graphs':
		for i,s in enumerate(subjects):
			if os.path.exists('/home/mbmbertolero/gene_expression/data/results/%s_fc_mods.npy' %(s)) == False \
			or os.path.exists('/home/mbmbertolero/gene_expression/data/results/%s_sc_mods.npy' %(s)) == False:
				subjects = np.delete(subjects,np.where(subjects == s))
	return subjects

def graph_metrics(subject,matrix):
	"""
	run graph metrics or load them
	"""
	try:
		pc = np.load('/home/mbmbertolero/gene_expression/data//results/%s_%s_pcs.npy' %(subject,matrix))
		wmd = np.load('/home/mbmbertolero/gene_expression/data//results/%s_%s_wmds.npy' %(subject,matrix))
		mod = np.load('/home/mbmbertolero/gene_expression/data/results/%s_%s_mods.npy' %(subject,matrix))
		m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_%s_matrix.npy' %(subject,matrix))
	except:
		print 'running subject %s' %(subject)
		if matrix == 'fc':
			m = []
			files = glob.glob('/home/mbmbertolero/hcp/matrices/yeo/*%s*REST*.npy' %(subject))
			for f in files:
				if 'random' in f: continue
				f = np.load(f)
				np.fill_diagonal(f,0.0)
				f[np.isnan(f)] = 0.0
				f = np.arctanh(f)
				m.append(f)
			if len(m) == 0.0:
				return
			m = np.nanmean(m,axis=0)
			m = np.tril(m,-1) + np.tril(m,-1).transpose()
		if matrix == 'sc':
			m = np.load('/home/mbmbertolero/data/gene_expression/sc_matrices/%s_matrix.npy'%(subject))
			m = m + m.transpose()
			m = np.tril(m,-1) + np.tril(m,-1).transpose()
		np.save('/home/mbmbertolero/gene_expression/data/matrices/%s_%s_matrix.npy' %(subject,matrix),m)
		num_nodes = m.shape[0]
		pc = []
		mod = []
		wmd = []
		for cost in np.linspace(1,10)*.01:
			temp_matrix = m.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost,binary=False,check_tri=True,interpolation='midpoint',mst=True)
			del temp_matrix
			graph = graph.community_infomap(edge_weights='weight')
			graph = brain_graphs.brain_graph(graph)
			pc.append(np.array(graph.pc))
			wmd.append(np.array(graph.wmd))
			mod.append(graph.community.modularity)
			del graph
		pc = np.nanmean(pc,axis=0)
		wmd = np.nanmean(wmd,axis=0)
		mod = np.nanmean(mod)
		np.save('/home/mbmbertolero/gene_expression/data/results/%s_%s_pcs.npy' %(subject,matrix),pc)
		np.save('/home/mbmbertolero/gene_expression/data/results/%s_%s_wmds.npy' %(subject,matrix),wmd)
		np.save('/home/mbmbertolero/gene_expression/data/results/%s_%s_mods.npy' %(subject,matrix),mod)

	return pc,wmd,mod,m

def generate_correlation_map(x, y):
	"""
	Correlate each n with each m.
	----------
	Parameters

	x : np.array, shape N X T.
	y : np.array, shape M X T.
	Returns: np.array, N X M array in which each element is a correlation coefficient.
	----------
	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
	    raise ValueError('x and y must ' +
	                     'have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x,
	             y.T) - n * np.dot(mu_x[:, np.newaxis],
	                              mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def super_edge_predict(t):
	print t
	fit_mask = np.ones((subject_pcs.shape[0])).astype(bool)
	fit_mask[t] = False
	if use_matrix == True:
		flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
		for s in range(subject_pcs.shape[0]):
			m = matrices[s]
			flat_matrices[s] = m[np.tril_indices(264,-1)]

		rest_perf_edge_corr = generate_correlation_map(task_perf[fit_mask].values.reshape(1,-1),flat_matrices[fit_mask].transpose())[0]

		rest_perf_edge_scores = np.zeros((subject_pcs.shape[0]))
		for s in range(subject_pcs.shape[0]):
			rest_perf_edge_scores[s] = pearsonr(flat_matrices[s],rest_perf_edge_corr)[0]

	perf_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		perf_pc_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_pcs[fit_mask,i])[0]
	perf_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_wmds.shape[1]):
		perf_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_wmds[fit_mask,i])[0]

	rest_pc = np.zeros(subject_pcs.shape[0])
	rest_wmd = np.zeros(subject_pcs.shape[0])
	for s in range(subject_pcs.shape[0]):
		rest_pc[s] = nan_pearsonr(subject_pcs[s],perf_pc_corr)[0]
		rest_wmd[s] = nan_pearsonr(subject_wmds[s],perf_wmd_corr)[0]

	if use_matrix: pvals = np.array([rest_pc,rest_wmd,rest_perf_edge_scores,subject_mods]).transpose()
	else: pvals = np.array([rest_pc,rest_wmd,subject_mods]).transpose()
	# pvals = np.array([rest_pc,rest_wmd,subject_mods]).transpose()
	# neurons = (3,3)

	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	model = Ridge(alpha=1.0)
	# model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons,alpha=1e-5,random_state=t)
	model.fit(pvals[train],task_perf[train])
	result = model.predict(pvals[t].reshape(1, -1))[0]
	return result

def super_edge_predict_new(t):
	print t
	fit_mask = np.ones((subject_pcs.shape[0])).astype(bool)
	fit_mask[t] = False
	if use_matrix == True:
		flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
		for s in range(subject_pcs.shape[0]):
			m = matrices[s]
			flat_matrices[s] = m[np.tril_indices(264,-1)]
		rest_perf_edge_corr = generate_correlation_map(task_perf[fit_mask].values.reshape(1,-1),flat_matrices[fit_mask].transpose())[0]
		rest_perf_edge_corr[np.abs(rest_perf_edge_corr) < np.mean(abs(rest_perf_edge_corr))] = np.nan
		rest_perf_edge_scores = np.zeros((subject_pcs.shape[0]))
		for s in range(subject_pcs.shape[0]):
			rest_perf_edge_scores[s] = nan_pearsonr(flat_matrices[s],rest_perf_edge_corr)[0]

	perf_pc_corr = np.zeros(subject_pcs.shape[1])
	for i in range(subject_pcs.shape[1]):
		perf_pc_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_pcs[fit_mask,i])[0]
	perf_wmd_corr = np.zeros(subject_wmds.shape[1])
	for i in range(subject_wmds.shape[1]):
		perf_wmd_corr[i] = nan_pearsonr(task_perf[fit_mask],subject_wmds[fit_mask,i])[0]

	perf_wmd_corr[np.abs(perf_wmd_corr) < np.mean(abs(perf_wmd_corr))] = np.nan
	perf_pc_corr[np.abs(perf_pc_corr) < np.mean(abs(perf_pc_corr))] = np.nan

	rest_pc = np.zeros(subject_pcs.shape[0])
	rest_wmd = np.zeros(subject_pcs.shape[0])
	for s in range(subject_pcs.shape[0]):
		rest_pc[s] = nan_pearsonr(subject_pcs[s],perf_pc_corr)[0]
		rest_wmd[s] = nan_pearsonr(subject_wmds[s],perf_wmd_corr)[0]

	if use_matrix: pvals = np.array([rest_pc,rest_wmd,rest_perf_edge_scores,subject_mods]).transpose()
	else: pvals = np.array([rest_pc,rest_wmd,subject_mods]).transpose()
	# pvals = np.array([rest_pc,rest_wmd,subject_mods]).transpose()
	neurons = (4,6,4,6)

	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	# model = Ridge(alpha=1.0)
	model = SVR()
	# model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons,alpha=1e-5,random_state=t)
	model.fit(pvals[train],task_perf[train])
	result = model.predict(pvals[t].reshape(1, -1))[0]
	return result

def task_performance(subjects,task):
	# df = pd.read_csv('//home/mbmbertolero//S900_Release_Subjects_Demographics.csv')
	df = pd.read_csv('//home/mbmbertolero/hcp/S1200.csv')
	performance = []
	if task == 'WM':
		wm_df = pd.DataFrame(np.array([df.Subject.values,df['WM_Task_Acc'].values]).transpose(),columns=['Subject','ACC']).dropna()
		for subject in subjects:
			temp_df = wm_df[wm_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(temp_df['ACC'].values[0])
	if task == 'RELATIONAL':
		for subject in subjects:
			try:performance.append(df['Relational_Task_Acc'][df.Subject == subject].values[0])
			except: performance.append(np.nan)
	if task == 'LANGUAGE':
		for subject in subjects:
			try:performance.append(np.nanmean([df['Language_Task_Story_Avg_Difficulty_Level'][df.Subject == subject].values[0],df['Language_Task_Math_Avg_Difficulty_Level'][df.Subject == subject].values[0]]))
			except: performance.append(np.nan)
	if task == 'SOCIAL':
		social_df = pd.DataFrame(np.array([df.Subject,df['Social_Task_TOM_Perc_TOM'],df['Social_Task_Random_Perc_Random']]).transpose(),columns=['Subject','ACC_TOM','ACC_RANDOM']).dropna()
		for subject in subjects:
			temp_df = social_df[social_df.Subject==subject]
			if len(temp_df) == 0:
				performance.append(np.nan)
				continue
			performance.append(np.nanmean([temp_df['ACC_RANDOM'].values[0],temp_df['ACC_TOM'].values[0]]))
	performance = np.array(performance)
	performance[np.where(np.array(subjects).astype(int) == 142626)[0]] = np.nan #repeat subject
	return performance

def behavior(subjects,all_b=False):
	subjects = subjects.astype(int)
	df = pd.read_csv('//home/mbmbertolero/hcp/S1200.csv')
	task_perf = pd.DataFrame(columns=['WM','RELATIONAL','SOCIAL','LANGUAGE'])
	for task in task_perf.columns.values:
		task_perf[task] = task_performance(df.Subject.values,task)
	task_perf['Subject'] =df.Subject.values

	fin = pd.merge(task_perf,df,how='outer',on='Subject')

	if all_b == True:
		to_keep = ['MMSE_Score','PicSeq_AgeAdj','CardSort_AgeAdj','Flanker_AgeAdj','PMAT24_A_CR',\
		'ReadEng_AgeAdj','PicVocab_AgeAdj','ProcSpeed_AgeAdj','DDisc_AUC_40K','DDisc_AUC_200',\
		'SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_AgeAdj',\
		'ER40_CR','ER40ANG','ER40FEAR','ER40HAP','ER40NOE','ER40SAD',\
		'AngAffect_Unadj','AngHostil_Unadj','AngAggr_Unadj','FearAffect_Unadj','FearSomat_Unadj','Sadness_Unadj',\
		'LifeSatisf_Unadj','MeanPurp_Unadj','PosAffect_Unadj','Friendship_Unadj','Loneliness_Unadj',\
		'PercHostil_Unadj','PercReject_Unadj','EmotSupp_Unadj','InstruSupp_Unadj'\
		'PercStress_Unadj','SelfEff_Unadj','Endurance_AgeAdj','GaitSpeed_Comp','Dexterity_AgeAdj','Strength_AgeAdj',\
		'NEOFAC_A','NEOFAC_O','NEOFAC_C','NEOFAC_N','NEOFAC_E','PainInterf_Tscore','PainIntens_RawScore','PainInterf_Tscore','Taste_AgeAdj'\
		'Mars_Final','PSQI_Score','VSPLOT_TC','WM','RELATIONAL','SOCIAL','LANGUAGE']
	else:
		to_keep = ['PicSeq_AgeAdj','CardSort_AgeAdj','Flanker_AgeAdj','PMAT24_A_CR',\
		'ReadEng_AgeAdj','PicVocab_AgeAdj','ProcSpeed_AgeAdj','DDisc_AUC_40K','DDisc_AUC_200',\
		'SCPT_SEN','SCPT_SPEC','IWRD_TOT','ListSort_AgeAdj',\
		'ER40_CR','WM','RELATIONAL','SOCIAL','LANGUAGE']

	translation = {'Loneliness_Unadj': 'Lonelisness','PercReject_Unadj':'Percieved Rejection','AngHostil_Unadj':'Hostility','Sadness_Unadj':'Sadness','PercHostil_Unadj':'Percieved Hostility','NEOFAC_N':'Neuroticism',\
	'FearAffect_Unadj':'Fear','AngAggr_Unadj':'Agressive Anger','PainInterf_Tscore':'Pain Interferes With Daily Life','Strength_AgeAdj':'Physical Strength','FearSomat_Unadj':'Somatic Fear','PSQI_Score':'Poor Sleep',\
	'SCPT_SPEC':'Sustained Attention Specificity','SCPT_SEN':'Sustained Attention Sensativity','ER40HAP':'Emotion, Happy Identifications','DDisc_AUC_200':'Delay Discounting:$200',\
	'GaitSpeed_Comp':'Gait Speed','DDisc_AUC_40K':'Delay Discounting: $40,000','ER40NOE':'Emotion, Neutral Identifications','ER40ANG':'Emotion, Angry Identifications',\
	'ER40FEAR':'Emotion, Fearful Identifications','ER40SAD':'Emotion, Sad Identifications','ER40_CR':'Emotion Recognition','MMSE_Score':'Mini Mental Status Exam','NEOFAC_O':'Openness','IWRD_TOT':'Verbal Memory','PMAT24_A_CR':'Penn Matrix','NEOFAC_C':'Conscientiousness',\
	'NEOFAC_A':'Agreeableness','Flanker_AgeAdj':'Flanker Task','CardSort_AgeAdj':'Card Sorting Task','NEOFAC_E':'Extraversion','Dexterity_AgeAdj':'Dexterity','Endurance_AgeAdj':'Endurance','ReadEng_AgeAdj':'Oral Reading Recognition',\
	'PicVocab_AgeAdj':'Picture Vocabulary','ProcSpeed_AgeAdj':'Processing Speed','SelfEff_Unadj':'Percieved Stress','PosAffect_Unadj':'Positive Affect','MeanPurp_Unadj':'Meaning and Purpose','Friendship_Unadj':'Friendship',\
	'PicSeq_AgeAdj':'Picture Sequence Memory','LifeSatisf_Unadj':'Life Satisfaction','EmotSupp_Unadj':'Emotional Support','ListSort_AgeAdj':'Working Memory','VSPLOT_TC':'Spatial','AngAffect_Unadj':'Anger, Affect'}

	for s in fin.Subject.values:
		if int(s) not in subjects: fin.drop(fin[fin.Subject.values == s].index,axis=0,inplace=True)
	assert (np.array(fin.Subject.values) == np.array(subjects).astype(int)).all()
	for c in fin.columns:
		if c not in to_keep: fin = fin.drop(c,axis=1)
	for c in fin.columns:
		a = fin[c][np.isnan(fin[c])]
		assert len(a[a==True]) == 0
		# print len(a[a==True])
		fin[c][np.isnan(fin[c])] = np.nanmean(fin[c])
	for c in fin.columns:
		if c in translation.keys():fin.rename(columns={'%s'%(c): translation[c]}, inplace=True)
	fin['Working Memory'] = np.mean([fin['Working Memory'],fin['WM']],axis=0)
	fin = fin.drop('WM',axis=1)
	fin['Delay Discounting'] = np.mean([fin['Delay Discounting:$200'],fin['Delay Discounting: $40,000']],axis=0)
	fin = fin.drop('Delay Discounting:$200',axis=1)
	fin = fin.drop('Delay Discounting: $40,000',axis=1)
	fin['Sustained Attention'] = np.mean([fin['Sustained Attention Specificity'],fin['Sustained Attention Sensativity']],axis=0)
	fin = fin.drop('Sustained Attention Specificity',axis=1)
	fin = fin.drop('Sustained Attention Sensativity',axis=1)
	return fin

def adult_fit_analysis(matrix='fc',distance=True):
	subjects = get_subjects(done='graphs')
	static_results = get_graphs(subjects,matrix)
	matrices = static_results['matrices']
	distance_matrix = atlas_distance()
	for m,smatrix in enumerate(matrices):
		if distance:
			np.fill_diagonal(matrices[m],np.nan)
			matrices[m][np.isnan(matrices[m])==False] = sm.GLM(matrices[m][np.isnan(matrices[m])==False],sm.add_constant(distance_matrix[np.isnan(matrices[m])==False])).fit().resid_response
	task_perf = behavior(subjects,all_b=True)
	final_matrix = full_gene_coexpression(matrix,False,distance)
	gene_similarity = []
	matrix_similarity = []
	mean = np.nanmean(matrices,axis=0)
	for i in range(matrices.shape[0]):
		gene_similarity.append(nan_pearsonr(matrices[i].flatten(),final_matrix.flatten())[0])
		matrix_similarity.append(nan_pearsonr(matrices[i].flatten(),mean.flatten())[0])
	for t in task_perf.columns:
		print t,pearsonr(task_perf[t].values,gene_similarity)

def which_edges():
	subjects = get_subjects(done='matrices')
	b = behavior(subjects,all_b=True)
	fc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_fits_df.csv'%('fc'))
	sc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_fits_df.csv'%('sc'))

	fc_matrices = []
	sc_matrices = []
	for s in subjects:
		m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_fc_matrix.npy' %(s))
		np.fill_diagonal(m,np.nan)
		fc_matrices.append(m.copy())
		m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_sc_matrix.npy' %(s))
		np.fill_diagonal(m,np.nan)
		sc_matrices.append(m.copy())
	fc_matrices = np.array(fc_matrices)
	sc_matrices = np.array(sc_matrices)

	for task in b.columns.values:
		print task
		task_perf = b[task].values.reshape(1,-1)
		fc_corrs = generate_correlation_map(task_perf,fc_matrices.reshape(fc_matrices.shape[0],-1).transpose())
		sc_corrs = generate_correlation_map(task_perf,sc_matrices.reshape(sc_matrices.shape[0],-1).transpose())
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_fc_corrs.npy'%(task),fc_corrs)
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_sc_corrs.npy'%(task),sc_corrs)

def make_heatmap(data,cmap="RdBu_r",dmin=None,dmax=None):
	minflag = False
	maxflag = False
	orig_colors = sns.color_palette(cmap,n_colors=1001)
	norm_data = np.array(copy.copy(data))
	if dmin != None:
		if dmin > np.min(norm_data):norm_data[norm_data<dmin]=dmin
		else:
			norm_data=np.append(norm_data,dmin)
			minflag = True
	if dmax != None:
		if dmax < np.max(norm_data):norm_data[norm_data>dmax]=dmax
		else:
			norm_data=np.append(norm_data,dmax)
			maxflag = True
	if np.nanmin(data) < 0.0: norm_data = norm_data + (np.nanmin(norm_data)*-1)
	elif np.nanmin(data) > 0.0: norm_data = norm_data - (np.nanmin(norm_data))
	norm_data = norm_data / float(np.nanmax(norm_data))
	norm_data = norm_data * 1000
	norm_data = norm_data.astype(int)
	colors = []
	for d in norm_data:
		colors.append(orig_colors[d])
	if maxflag: colors = colors[:-1]
	if minflag: colors = colors[:-1]
	return colors

def make_cifti_heatmap(data,cmap="RdBu_r"):
	orig_colors = sns.color_palette(cmap,n_colors=1001)
	norm_data = copy.copy(data)
	if np.nanmin(data) < 0.0: norm_data = norm_data + (np.nanmin(norm_data)*-1)
	elif np.nanmin(data) > 0.0: norm_data = norm_data - (np.nanmin(norm_data))
	norm_data = norm_data / float(np.nanmax(norm_data))
	norm_data = norm_data * 1000
	norm_data = norm_data.astype(int)
	colors = []
	for d in norm_data:
		colors.append(orig_colors[d])
	return colors

def plot_which_edges(all_b=False):
	corr_method = 'pearsonr'
	direction='absolute'
	metric='fit'
	corr_method='pearsonr'
	n_genes= 50
	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	subjects = get_subjects(done='matrices')
	b = behavior(subjects,all_b=all_b)
	# fc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_fits_df.csv'%('fc'))
	# fc_fit_df = fc_fit_df[fc_fit_df.n_genes==100]

	fc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_True_%s_False_fits_df.csv'%(corr_method))
	# fc_fit_df = fc_fit_df[fc_fit_df.n_genes==n_genes]
	# fc_fit_df= fc_fit_df.groupby('node').fit.mean()

	# sc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_fits_df.csv'%('sc'))
	# sc_fit_df = sc_fit_df[sc_fit_df.n_genes==100]
	sc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_True_%s_False_fits_df.csv'%(corr_method))
	# sc_fit_df = sc_fit_df[sc_fit_df.n_genes==n_genes]
	# sc_fit_df = sc_fit_df.groupby('n_genes').fit.mean()

	fc_pc = np.load('/home/mbmbertolero/data/gene_expression/results/fc_pc.npy')
	sc_pc = np.load('/home/mbmbertolero/data/gene_expression/results/sc_pc.npy')
	behavior_df = pd.DataFrame(columns=['Connectivity','r','p','Behavioral Measure'])
	edges_p = []
	nodes_p = []
	similarity_df = pd.DataFrame(columns=['Behavioral Measure','r','p','comparison'])
	for task in b.columns.values:
		fc_corrs = np.load('/home/mbmbertolero/data/gene_expression/results/%s_fc_corrs.npy'%(task))
		sc_corrs = np.load('/home/mbmbertolero/data/gene_expression/results/%s_sc_corrs.npy'%(task))

		# print task, pearsonr(np.nanmean(fc_corrs.reshape((400,400)),axis=0)[mask],df['participation coefficient'])

		if direction == 'positive':
			fc_corrs[fc_corrs<0.0] = np.nan
			sc_corrs[sc_corrs<0.0] = np.nan
		if direction == 'negative':
			fc_corrs[fc_corrs>0.0] = np.nan
			sc_corrs[sc_corrs>0.0] = np.nan

		r,p = nan_pearsonr(fc_corrs,sc_corrs)
		similarity_df = similarity_df.append(pd.DataFrame([[task,r,p,'edge wise']],columns=['Behavioral Measure','r','p','comparison']))
		r,p=nan_pearsonr(np.nanmean(fc_corrs.reshape((400,400)),axis=0),np.nanmean(sc_corrs.reshape((400,400)),axis=0))
		similarity_df = similarity_df.append(pd.DataFrame([[task,r,p,'node wise']],columns=['Behavioral Measure','r','p','comparison']))

		fc_r = nan_pearsonr(np.nanmean(fc_corrs.reshape((400,400)),axis=0)[:200][mask],fc_fit_df.groupby('node').fit.mean())
		sc_r = nan_pearsonr(np.nanmean(sc_corrs.reshape((400,400)),axis=0)[:200][mask],sc_fit_df.groupby('node').fit.mean())
		behavior_df = behavior_df.append(pd.DataFrame(np.array([['functional'],[fc_r[0]],[fc_r[1]],[task]]).transpose(),columns=['Connectivity','r','p','Behavioral Measure']))
		behavior_df = behavior_df.append(pd.DataFrame(np.array([['structural'],[sc_r[0]],[sc_r[1]],[task]]).transpose(),columns=['Connectivity','r','p','Behavioral Measure']))


	similarity_df['r'] = similarity_df['r'].values.astype(float)
	similarity_df['p'] = similarity_df['p'].values.astype(float)
	similarity_df['colors'] = make_heatmap(similarity_df['r'].values,'coolwarm',-.3,.3)

	fig, ax = sns.plt.subplots(figsize=(3.74016,3.74016*.75))
	sns.violinplot(y='r',x='comparison',data=similarity_df,order=['node wise','edge wise'],ax=ax)
	sns.plt.ylabel('pearson r, structural and functional\n correlations with behavior')
	sns.plt.xlabel('')
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/which_edges_fit_and_predict_similarity_%s_violin.pdf'%(all_b))

	sns.plt.show()
	# 1/0

	left, width = 0, 1
	if all_b: bottom, height = 0, 1
	else: bottom, height = 0, .4
	right = left + width
	top = bottom + height
	fig = plt.figure(figsize=(4,5.5))
	for col,connectivity in zip(np.linspace(.25,.75,2),['node wise','edge wise']):
		order = similarity_df[similarity_df.comparison==connectivity]['Behavioral Measure'].values[np.argsort(similarity_df[similarity_df.comparison==connectivity]['r'].values)]
		scores = similarity_df[similarity_df.comparison==connectivity]['r'].values[np.argsort(similarity_df[similarity_df.comparison==connectivity]['r'].values)]
		p_vals = similarity_df[similarity_df.comparison==connectivity]['p'].values[np.argsort(similarity_df[similarity_df.comparison==connectivity]['r'].values)]
		p_vals = fdrcorrection(p_vals,0.05)[1]
		for ix,o in enumerate(order):
			if float(scores[ix]) < 0.0:
				s = '-' + str(scores[ix])[1:5]
				order[ix] = order[ix] + ' (%s)'%(s)
				order[ix] = order[ix].capitalize()
				continue
			s = str(scores[ix])[1:4]
			order[ix] = order[ix] + ' (%s)'%(s)
			order[ix] = order[ix].capitalize()
			if p_vals[ix] <= 0.05: order[ix] = order[ix] + ' *'
		order = np.append(order,connectivity.capitalize())
		colors = similarity_df[similarity_df.comparison==connectivity]['colors'].values[np.argsort(similarity_df[similarity_df.comparison==connectivity]['r'].values)]
		colors = list(colors)
		colors.append((0,0,0))
		locs = (np.arange(len(order)+1)/float(len(order)+1))[1:]
		for i,t,c in zip(locs,order,colors):
			if t == 'Wm': t = 'Working Memory'
			fig.text(col*(left+right), float(i)*(bottom+top), t,horizontalalignment='center',verticalalignment='center',fontsize=7, color=c)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/which_edges_fit_and_predict_similarity_%s.pdf'%(all_b))
	sns.plt.show()


	behavior_df['r'] = behavior_df['r'].values.astype(float)
	behavior_df['p'] = behavior_df['p'].values.astype(float)
	behavior_df['colors'] = make_heatmap(behavior_df['r'].values,'coolwarm',-.3,.3)

	# left, width = 0, 1
	# bottom, height = 0, 1
	right = left + width
	top = bottom + height
	fig = plt.figure(figsize=(4,5.5))
	for col,connectivity in zip(np.linspace(.25,.75,2),['structural','functional']):
		order = behavior_df[behavior_df.Connectivity==connectivity]['Behavioral Measure'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['r'].values)]
		scores = behavior_df[behavior_df.Connectivity==connectivity]['r'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['r'].values)]
		p_vals = behavior_df[behavior_df.Connectivity==connectivity]['p'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['r'].values)]
		p_vals = fdrcorrection(p_vals,0.05)[1]
		for ix,o in enumerate(order):
			if float(scores[ix]) < 0.0:
				s = '-' + str(scores[ix])[1:5]
				order[ix] = order[ix] + ' (%s)'%(s)
				order[ix] = order[ix].capitalize()
				continue
			s = str(scores[ix])[1:4]
			order[ix] = order[ix] + ' (%s)'%(s)
			order[ix] = order[ix].capitalize()
			if p_vals[ix] <= 0.05: order[ix] = order[ix] + ' *'
		order = np.append(order,connectivity.capitalize())
		colors = behavior_df[behavior_df.Connectivity==connectivity]['colors'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['r'].values)]
		colors = list(colors)
		colors.append((0,0,0))
		locs = (np.arange(len(order)+1)/float(len(order)+1))[1:]
		for i,t,c in zip(locs,order,colors):
			if t == 'Wm': t = 'Working Memory'
			fig.text(col*(left+right), float(i)*(bottom+top), t,horizontalalignment='center',verticalalignment='center',fontsize=7, color=c)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/which_edges_fit_and_predict_behavior_%s_%s_%s.pdf'%(direction,metric,all_b))
	sns.plt.show()

def performance(matrix='fc',task='Working Memory',version='new'):
	global subject_pcs
	global subject_wmds
	global subject_mods
	global task_perf
	global matrices
	global use_matrix
	use_matrix = True
	# if os.path.exists('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_prediction1.npy'%(matrix,task,use_matrix,version)):
	# 	subjects = get_subjects(done='graphs')
	# 	task_perf = behavior(subjects,all_b=True)[task]
	# 	if version == 'new':nodal_prediction = np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_prediction.npy'%(matrix,task,use_matrix,version))
	# 	else:nodal_prediction = np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_prediction.npy'%(matrix,task,use_matrix,version))
	# else:
	subjects = get_subjects(done='graphs')
	static_results = get_graphs(subjects,matrix)
	subject_pcs = static_results['subject_pcs'].copy()
	subject_wmds = static_results['subject_wmds']
	matrices = static_results['matrices']
	subject_mods = static_results['subject_mods']
	task_perf = behavior(subjects,all_b=True)[task]
	"""
	prediction / cross validation
	"""
	if version == 'new':
		nodal_prediction = []
		for i in range(len(task_perf)): nodal_prediction.append(super_edge_predict_new(i))
	# if version == 'old': nodal_prediction = pool.map(super_edge_predict,range(len(task_perf)))
	np.save('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_prediction.npy'%(matrix,task,use_matrix,version),nodal_prediction)
	# del pool
	result = pearsonr(np.array(nodal_prediction).reshape(-1),task_perf)
	print 'Prediction of Performance: ', result

def print_performance(matrix='fc',version='old',use_matrix = True):
	subjects = get_subjects(done='graphs')
	b = behavior(subjects)
	old_results = []
	new_results = []
	for task in b.columns.values:
		task_perf = b[task]
		new_nodal_prediction = np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_prediction.npy'%(matrix,task,use_matrix,'new'))
		old_nodal_prediction = np.load('/home/mbmbertolero/gene_expression/results/%s_%s_%s_%s_prediction.npy'%(matrix,task,use_matrix,'old'))
		old_results.append(pearsonr(np.array(old_nodal_prediction).reshape(-1),task_perf)[0])
		new_results.append(pearsonr(np.array(new_nodal_prediction).reshape(-1),task_perf)[0])

def plot_perfomance(all_b=True):
	use_matrix=True
	subjects = get_subjects(done='graphs')
	b = behavior(subjects,all_b=all_b)
	behavior_df = pd.DataFrame(columns=['Connectivity','Prediction Accuracy','p','Behavioral Measure'])
	for task in b.columns.values:
		fc_nodal_prediction = np.load('/home/mbmbertolero/gene_expression/results/fc_%s_%s_new_prediction.npy'%(task,use_matrix))
		sc_nodal_prediction = np.load('/home/mbmbertolero/gene_expression/results/sc_%s_%s_new_prediction.npy'%(task,use_matrix))
		fc_r = nan_pearsonr(fc_nodal_prediction,b[task])
		sc_r = nan_pearsonr(sc_nodal_prediction,b[task])
		if np.isnan(sc_r[0]): sc_r = (0.0,1.0)
		behavior_df = behavior_df.append(pd.DataFrame(np.array([['structural','functional'],[sc_r[0],fc_r[0]],[sc_r[1],fc_r[1]],[task,task]]).transpose(),columns=['Connectivity','Prediction Accuracy','p','Behavioral Measure']))

	behavior_df['Prediction Accuracy'] = behavior_df['Prediction Accuracy'].values.astype(float)
	behavior_df['p'] = behavior_df['p'].values.astype(float)
	# behavior_df = behavior_df.dropna()
	behavior_df['colors'] = make_heatmap(behavior_df['Prediction Accuracy'].values,'coolwarm',-.3,.3)

	sns.set(context="notebook",font='Open Sans',style='white')

	# sns.violinplot(data=behavior_df,y='Prediction Accuracy',x='Connectivity',inner='quartile')
	print scipy.stats.ttest_ind(behavior_df['Prediction Accuracy'][behavior_df.Connectivity=='functional'],behavior_df['Prediction Accuracy'][behavior_df.Connectivity=='structural'])
	# 1/0
	ax = sns.regplot(x=behavior_df['Prediction Accuracy'][behavior_df.Connectivity=='functional'],y=behavior_df['Prediction Accuracy'][behavior_df.Connectivity=='structural'])
	sns.plt.ylabel('Structural Connectivity\nPrediction Accuracy')
	sns.plt.xlabel('Functional Connectivity\nPrediction Accuracy')
	r,p = pearsonr(x=behavior_df['Prediction Accuracy'][behavior_df.Connectivity=='functional'],y=behavior_df['Prediction Accuracy'][behavior_df.Connectivity=='structural'])
	sns.plt.text(0.1, 0.9,convert_r_p(r,p), horizontalalignment='center',verticalalignment='center',transform=ax.transAxes)
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/prediction_corrs_%s.pdf'%(all_b))
	sns.plt.show()
	# 1/0

	left, width = 0, 1
	if all_b: bottom, height = 0, 1
	else: bottom, height = 0, .4
	right = left + width
	top = bottom + height
	fig = plt.figure(figsize=(4,5.5))
	for col,connectivity in zip(np.linspace(.25,.75,2),['structural','functional']):
		order = behavior_df[behavior_df.Connectivity==connectivity]['Behavioral Measure'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['Prediction Accuracy'].values)]
		scores = behavior_df[behavior_df.Connectivity==connectivity]['Prediction Accuracy'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['Prediction Accuracy'].values)]
		p_vals = behavior_df[behavior_df.Connectivity==connectivity]['p'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['Prediction Accuracy'].values)]
		# p_vals = p_vals * float(len(b.columns))
		p_vals = fdrcorrection(p_vals,0.05)[1]
		for ix,o in enumerate(order):
			if float(scores[ix]) < 0.0:
				s = '-' + str(scores[ix])[1:5]
				order[ix] = order[ix] + ' (%s)'%(s)
				order[ix] = order[ix].capitalize()
				continue
			s = str(scores[ix])[1:4]
			order[ix] = order[ix] + ' (%s)'%(s)
			order[ix] = order[ix].capitalize()
			if p_vals[ix] <= 0.05: order[ix] = order[ix] + ' *'
		order = np.append(order,connectivity.capitalize())
		colors = behavior_df[behavior_df.Connectivity==connectivity]['colors'].values[np.argsort(behavior_df[behavior_df.Connectivity==connectivity]['Prediction Accuracy'].values)]
		colors = list(colors)
		colors.append((0,0,0))
		locs = (np.arange(len(order)+1)/float(len(order)+1))[1:]
		for i,t,c in zip(locs,order,colors):
			if t == 'Wm': t = 'Working Memory'
			fig.text(col*(left+right), (float(i)*(bottom+top)), t,horizontalalignment='center',verticalalignment='center',fontsize=7, color=c)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/new_figures/prediction_acc_%s.pdf'%(all_b))
	sns.plt.show()

def con_variance(matrix):
	rdf = pd.read_csv('/home/mbmbertolero//hcp/restricted.csv')
	df = pd.read_csv('/home/mbmbertolero/hcp/S1200.csv')
	df = pd.merge(df,rdf,on='Subject')
	done_subjects = get_subjects(done='matrices')
	subjects = []
	for s in df.Subject:
		# if df['ZygositySR'][df.Subject==s].values[0] == 'NotTwin':
		if s in done_subjects:
			subjects.append(s)
	matrices = []
	for s in subjects:
		if matrix == 'fc':m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_fc_matrix.npy'%(s))
		if matrix == 'sc': m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_sc_matrix.npy'%(s))
		np.fill_diagonal(m,0.0)
		matrices.append(m)
	var = np.zeros((400,400))
	matrices = np.array(matrices)
	for i,j in combinations(range(400),2):
		a = matrices[:,i,j]
		a = a[np.isnan(a)== False]
		a = a - np.min(a)
		a = a / np.sum(a)
		r = scipy.stats.entropy(a)
		var[i,j] = r
		var[j,i] = r
	np.save('/home/mbmbertolero/data/gene_expression/%s_var.npy'%(matrix),var)

def heritability(matrix):
	# https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/software/apace
	eng = matlab.engine.start_matlab()
	eng.addpath(eng.genpath('/home/mbmbertolero/data/APACE/'))
	rdf = pd.read_csv('/home/mbmbertolero//hcp/restricted.csv')

	df = pd.read_csv('/home/mbmbertolero/hcp/S1200.csv')
	df = pd.merge(df,rdf,on='Subject')
	subjects = get_subjects(done='matrices')
	for s in df.Subject:
		if len(df['ZygositySR'][df.Subject==s].values[0]) == 1:
			df = df[df.Subject!=s]
			continue
		if s not in subjects: df = df[df.Subject!=s]

	# assert (df.Subject == subjects).all() == True
	matrices = []
	for s in df.Subject:
		if matrix == 'fc':m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_fc_matrix.npy'%(s))
		if matrix == 'sc': m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_sc_matrix.npy'%(s))
		m[np.isnan(m)] = 0.0
		np.fill_diagonal(m,0.0)
		matrices.append(m[np.triu_indices(400,1)])
	matrices = np.array(matrices).transpose()
	ACEfit_Par = {}

	w_df = df[['Subject','Mother_ID','Father_ID','ZygositySR']].astype(str)
	w_df.to_csv('/home/mbmbertolero/data/gene_expression/twin_data_%s.csv'%(matrix),header=True,index=False)
	dm = df[['Gender']]
	dm[dm.Gender=='M'] = 0
	dm[dm.Gender=='F'] = 1
	dm = dm.astype(int)
	ages = df[['Age']]
	ages[ages=='22-25'] = 1
	ages[ages=='26-30'] = 2
	ages[ages=='31-35'] = 3
	ages[ages=='36+'] = 4
	ACEfit_Par['P_nm'] = matlab.double(matrices.tolist())
	ACEfit_Par['InfMx'] = '/home/mbmbertolero/data/gene_expression/twin_data_%s.csv'%(matrix)
	ACEfit_Par['ResDir'] = '/home/mbmbertolero/data/gene_expression/heritability/%s/'%(matrix)
	ACEfit_Par['Pmask'] = []
	ACEfit_Par['Dsnmtx'] = matlab.double(np.array([dm.values.astype(int),ages.values.astype(int)]).astype(int).swapaxes(0,2)[0].tolist())
	ACEfit_Par['Nlz'] = 1
	ACEfit_Par['AggNlz'] = 0
	ACEfit_Par['ContSel'] = []
	ACEfit_Par['NoImg'] = 0
	ACEfit_Par['Model'] = 'ACE'
	ACEfit_Par['N'] =  int(matrices.shape[1])

	ACEfit_Par = eng.PrepData(ACEfit_Par)

	ACEfit_Par['alpha_CFT'] = []

	ACEfit_Par = eng.ACEfit(ACEfit_Par)


	ACEfit_Par['nPerm'] = 0
	ACEfit_Par['nBoot'] = 0
	ACEfit_Par['nParallel'] = 1


	eng.PrepParallel(ACEfit_Par,nargout=0)

	Palpha = 0.05
	Balpha = 0.05


	eng.AgHe_Method(ACEfit_Par,Palpha,Balpha,nargout=0)
	ACEfit_Par['AggNlz']  = 1  #de-meaning and scaling to have stdev of 1.0
	eng.AgHe_Method(ACEfit_Par,Palpha,Balpha,'_Norm',nargout=0)

def analyze_heritability():
	homedir = '/home/mbmbertolero/'
	# homedir = '/Users/maxwell/upenn'
	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	smatrix = np.zeros((400,400))
	fmatrix = np.zeros((400,400))
	sm = scipy.io.loadmat('/%s/gene_expression/heritability/sc/ACEfit_Par.mat'%(homedir))
	fm = scipy.io.loadmat('%s/gene_expression/heritability/fc/ACEfit_Par.mat'%(homedir))
	smatrix[np.triu_indices(400,1)] = np.array(sm['ACEfit_Par']['Stats'][0][0]).flatten()
	fmatrix[np.triu_indices(400,1)] = np.array(fm['ACEfit_Par']['Stats'][0][0]).flatten()
	smatrix = smatrix.transpose() + smatrix
	fmatrix = fmatrix.transpose() + fmatrix
	np.fill_diagonal(smatrix,np.nan)
	np.fill_diagonal(fmatrix,np.nan)
	sm = np.array(sm['ACEfit_Par']['Stats'][0][0]).flatten()
	fm = np.array(fm['ACEfit_Par']['Stats'][0][0]).flatten()

	fc_var = np.load('/%s/gene_expression/%s_var.npy'%(homedir,'fc'))[:200,:200]
	#we correlate with entropy, a measure of conservation.
	np.fill_diagonal(fc_var,np.nan)
	sc_var = np.load('/%s/gene_expression/%s_var.npy'%(homedir,'sc'))[:200,:200]
	np.fill_diagonal(sc_var,np.nan)

	fpc = np.load('/%s/gene_expression/results/fc_pc.npy'%(homedir))[:200]
	spc = np.load('/%s/gene_expression/results/sc_pc.npy'%(homedir))[:200]

	fc_node_by_snp = np.load('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_fc.npy')
	sc_node_by_snp = np.load('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_sc.npy')

	# fdf = pd.read_csv('/%s/gene_expression/results/fc_fits_df.csv'%(homedir))
	# fdf = fdf[fdf.n_genes==n_genes]

	# sdf = pd.read_csv('/%s/gene_expression/results/sc_fits_df.csv'%(homedir))
	# sdf = sdf[sdf.n_genes==n_genes]

	fdf = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_True_%s_False_fits_df.csv'%(corr_method))
	# fdf = fdf[fdf.n_genes==n_genes]

	# sc_fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_fits_df.csv'%('sc'))
	# sc_fit_df = sc_fit_df[sc_fit_df.n_genes==100]
	sdf = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_True_%s_False_fits_df.csv'%(corr_method))
	# sdf = sdf[sdf.n_genes==n_genes]

	sm_nodes, fm_nodes = np.nanmean(smatrix,axis=0)[:200],np.nanmean(fmatrix,axis=0)[:200]

	print 'fit, con', nan_pearsonr(fdf.groupby('node').max().fit,np.nanmean(fc_var,axis=0)[mask])
	print 'pc, con', nan_pearsonr(fpc,np.nanmean(fc_var,axis=0))
	print 'herit, nodes, con', nan_pearsonr(fm_nodes,np.nanmean(fc_var,axis=0))

	print 'fit, con', nan_pearsonr(sdf.groupby('node').mean().fit,np.nanmean(sc_var,axis=0)[mask])
	print 'pc, con', nan_pearsonr(spc,np.nanmean(sc_var,axis=0))
	print 'herit, nodes, con', nan_pearsonr(sm_nodes,np.nanmean(sc_var,axis=0))

	fmodel_vars = np.array([fdf.groupby('node').max().fit]).transpose()
	fpc_r = sm.GLM(fpc[:200][mask],sm.add_constant(fmodel_vars)).fit().resid_response
	print pearsonr(fm_nodes[mask],fpc_r)

	df = pd.DataFrame(columns=['heritability','connectivity'])
	df['heritability'] = np.array([sm,fm]).flatten()
	names = np.ones(len(sm)*2).astype(str)
	names[:len(sm)] = 'structural'
	names[len(sm):] = 'functional'
	df['connectivity'] = names

	colors = sns.color_palette('mako',6)

	fc_c = sns.cubehelix_palette(10, rot=.25, light=.7)[0]
	sc_c = sns.cubehelix_palette(10, rot=-.25, light=.7)[0]

	t,p= scipy.stats.ttest_ind(fm,sm)
	a= sns.violinplot(x="connectivity", y="heritability", data=df,order=['structural','functional'],palette=[sc_c,fc_c])
	a.text(0.5,30,convert_t_p(t,p))
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/heritability_versus.pdf')
	sns.plt.show()

	a = sns.regplot(sm,fm,color=colors[3])
	r,p = pearsonr(sm,fm)
	a.text(25,50,convert_r_p(r,p))
	sns.plt.xlabel('functional connectivity heritability')
	sns.plt.ylabel('structural connectivity heritability')
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/edge_corr_hertiability.pdf')
	sns.plt.show()


	r,p = pearsonr(sm_nodes,fm_nodes)
	a = sns.regplot(sm_nodes,fm_nodes,color=colors[2])
	a.text(7,10,convert_r_p(r,p))
	sns.plt.xlabel('functional connectivity heritability')
	sns.plt.ylabel('structural connectivity heritability')
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/node_corr_hertiability.pdf')
	sns.plt.show()

	fcm = functional_connectivity(False,True,None)[:200,:200]
	# fcm  = fcm  + fcm.transpose()
	# fcm  = np.tril(fcm ,-1)
	# fcm  = fcm  + fcm.transpose()

	scm = structural_connectivity(False,True,None)[:200,:200]
	# scm  = scm  + scm.transpose()
	# scm  = np.tril(scm ,-1)
	# scm  = scm  + scm.transpose()

	r,p = pearsonr(sm_nodes,spc)
	a = sns.regplot(sm_nodes,spc,color=colors[1])
	a.text(7,.4,convert_r_p(r,p))
	sns.plt.xlabel('structural connectivity heritability')
	sns.plt.ylabel('structural connectivity participation coefficient')
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/structural_pc_corr_hertiability.pdf')
	sns.plt.show()

	r,p = pearsonr(fm_nodes,fpc)
	a = sns.regplot(fm_nodes,fpc,color=colors[0])
	a.text(8,.4,convert_r_p(r,p))
	sns.plt.xlabel('functional connectivity heritability')
	sns.plt.ylabel('functional connectivity participation coefficient')
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/functional_pc_corr_hertiability.pdf')
	sns.plt.show()


	r,p = pearsonr(sm_nodes[mask],sdf.groupby('node').fit)
	a = sns.regplot(sm_nodes[mask],sdf.fit,color=sc_c)
	a.text(7,.7,convert_r_p(r,p))
	sns.plt.xlabel('structural connectivity heritability')
	sns.plt.ylabel('structural connectivity genetic fit')
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/new_figures/structural_gene_corr_hertiability.pdf')
	sns.plt.show()

	r,p = pearsonr(fm_nodes[mask],fdf.fit)
	a = sns.regplot(fm_nodes[mask],fdf.fit,color=fc_c)
	a.text(8,.7,convert_r_p(r,p))
	sns.plt.xlabel('functional connectivity heritability')
	sns.plt.ylabel('functional connectivity genetic fit')
	sns.plt.tight_layout()
	sns.despine()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/new_figures/functional_gene_corr_hertiability.pdf')
	sns.plt.show()

def sge(run_type,matrix,n_genes=100):
	"""
	SGE stuff
	"""
	if run_type == 'graph_metrics':
		subjects = get_subjects()
		for s in subjects:
			# if os.path.exists('/home/mbmbertolero/data/gene_expression/sc_matrices/%s_matrix.npy'%(s)) == False:continue
			# if len(glob.glob('/home/mbmbertolero/data/gene_expression/data/results/%s_sc_**'%(s))) == 3: continue
			os.system("qsub -q all.q,basic.q -N gm_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
			 /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r graph_metrics -s %s "%(s,s))

	if run_type == 'find_genes':
		for node in range(400):
			os.system('qsub -binding linear:4 -pe unihost 4 -N fg_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
			 /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r find_genes -m %s -n %s -n_genes %s'%(node,matrix,node,n_genes))

	if run_type == 'predict':
		subjects = get_subjects(done='graphs')
		df = behavior(subjects,all_b=True)
		for i,task in enumerate(df.columns.values):
			# if task != 'Working Memory':continue
			# os.system('qsub -q all.q,basic.q -binding linear:8 -pe unihost 8 -N predict -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
			os.system('qsub -q all.q,basic.q -l h_vmem=6G,s_vmem=6G -N predict -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
			 /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r predict -task %s '%(i))

	if run_type == 'predict_role':
		os.system('qsub -q all.q,basic.q -binding linear:8 -pe unihost 8 -N pred_role -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
		 /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r predict_role -m %s' %(matrix))

def run_all(matrix,topological,distance,network):
	fit_matrix(matrix,topological,distance,network)
	SA_find_genes(matrix,topological,distance,network,n_trys=50)

def get_paris_df():
	names = np.array(['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'])

	fc_df = pd.DataFrame(columns=['Gene','RNMI.symmetric.MEAN','network','connectivity'])
	fc_files = glob.glob('/home/mbmbertolero/gene_expression/paris/*fc_paris_comm*')

	sc_df = pd.DataFrame(columns=['Gene','RNMI.symmetric.MEAN','network','connectivity'])
	sc_files = glob.glob('/home/mbmbertolero/gene_expression/paris/*sc_paris_comm*')

	size = (len(fc_files),pd.read_csv(fc_files[0]).shape[0])
	for i in range(size[0]):
		t_df= pd.read_csv(fc_files[i],sep='\t').sort_values('Gene')
		t_df = t_df[['Gene','RNMI.symmetric.MEAN']]
		t_df['network'] = np.zeros((size[1])).astype(str)
		t_df['network'] = names[i]
		t_df['connectivity'] = np.zeros((size[1])).astype(str)
		t_df['connectivity'] = 'functional'
		fc_df = fc_df.append(t_df,ignore_index=True)

	size = (len(sc_files),pd.read_csv(sc_files[0]).shape[0])
	for i in range(size[0]):
		t_df= pd.read_csv(sc_files[i],sep='\t').sort_values('Gene')
		t_df = t_df[['Gene','RNMI.symmetric.MEAN']]
		t_df['network'] = np.zeros((size[1])).astype(str)
		t_df['network'] = names[i]
		t_df['connectivity'] = np.zeros((size[1])).astype(str)
		t_df['connectivity'] = 'structural'
		sc_df = sc_df.append(t_df,ignore_index=True)

	df = pd.merge(fc_df,sc_df,how='inner',on='Gene')
	print pearsonr(df[df.connectivity_x == 'functional']['RNMI.symmetric.MEAN_x'],df[df.connectivity_y == 'structural']['RNMI.symmetric.MEAN_y'])
	for network in names:
		t_df = df[(df.network_x==network).values & (df.network_y==network).values]
		print (t_df[t_df.connectivity_x == 'functional']['Gene']==t_df[t_df.connectivity_y == 'structural']['Gene']).all()
		print pearsonr(t_df[t_df.connectivity_x == 'functional']['RNMI.symmetric.MEAN_x'],t_df[t_df.connectivity_y == 'structural']['RNMI.symmetric.MEAN_y'])

def get_paris(matrix,n):
	files = glob.glob('/home/mbmbertolero/gene_expression/paris/*%s_paris_comm*'%(matrix))
	files.sort()
	size = (len(files),pd.read_csv(files[0]).shape[0])
	m = np.zeros(size)
	top_m = np.zeros((size[0],n))
	for i in range(size[0]):
		m[i] = pd.read_csv(files[i],sep='\t').sort_values('Gene')['RNMI.symmetric.MEAN'].values
		top_m[i] = m[i][np.argsort(m[i])][-n:]
	top = np.zeros((len(files),n)).astype(str)
	top_FDR = np.zeros((len(files),n)).astype(float)
	top_NMI = np.zeros((len(files),n)).astype(float)
	for i in range(size[0]):
		top[i] = pd.read_csv(files[i],sep='\t').sort_values('RNMI.symmetric.MEAN')['Gene'][-n:]
		top_NMI[i] = pd.read_csv(files[i],sep='\t').sort_values('RNMI.symmetric.MEAN')['RNMI.symmetric.MEAN'][-n:]
		top_FDR[i] = pd.read_csv(files[i],sep='\t').sort_values('RNMI.symmetric.MEAN')['FDR1'][-n:]
	return m,top,top_FDR,top_NMI

def paris(matrix,n=15):
	sns.set(context="notebook",font='Open Sans',style='white',font_scale=1.1)
	c1,c2 = sns.cubehelix_palette(10, rot=.25, light=.7)[0],sns.cubehelix_palette(10, rot=-.25, light=.7)[0]
	if matrix == 'fc': c = c1
	else: c = c2
	names = np.array(['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default'])


	sc_genes = pd.read_csv(glob.glob('/home/mbmbertolero/gene_expression/paris/*sc_paris_comm*')[0],sep='\t').sort_values('Gene')['Gene'].values
	fc_genes = pd.read_csv(glob.glob('/home/mbmbertolero/gene_expression/paris/*fc_paris_comm*')[0],sep='\t').sort_values('Gene')['Gene'].values
	gene_mask = np.zeros((sc_genes.shape))
	gene_mask = np.zeros((sc_genes.shape)).astype(bool)
	for idx,g in enumerate(sc_genes):
		if g in fc_genes:gene_mask[idx]= True
		else:gene_mask[idx]= False


	m,top,top_FDR,top_NMI = get_paris(matrix,n)
	sc_m,sc_top,sc_top_FDR,sc_top_NMI = get_paris('sc',250)
	fc_m,fc_top,fc_top_FDR,fc_top_NMI = get_paris('fc',250)

	a =sns.violinplot(data=fc_top_NMI.transpose(),palette=yeo_colors,cut=True)
	sns.plt.ylabel('RNMI')
	sns.plt.xticks(range(7),names,rotation=90)
	plt.tight_layout()
	plt.ylim(0.00,.25)
	t_vals = scipy.stats.ttest_ind(fc_top_NMI.transpose(),sc_top_NMI.transpose())[0]
	p_vals = scipy.stats.ttest_ind(fc_top_NMI.transpose(),sc_top_NMI.transpose())[1]
	for i in range(7):
		a.text(i,.005,'$\it{t}$=%s\n%s' %(np.around(t_vals[i],2),log_p_value(p_vals[i])),{'fontsize':8},horizontalalignment='center')
	plt.savefig('/home/mbmbertolero/gene_expression/new_figures/nmi_fc_violin.pdf')
	plt.show()


	a = sns.violinplot(data=sc_top_NMI.transpose(),palette=yeo_colors,cut=True)
	sns.plt.ylabel('RNMI')
	sns.plt.xticks(range(7),names,rotation=90)
	plt.tight_layout()
	plt.ylim(0.00,.25)
	t_vals = scipy.stats.ttest_ind(sc_top_NMI.transpose(),fc_top_NMI.transpose())[0]
	p_vals = scipy.stats.ttest_ind(sc_top_NMI.transpose(),fc_top_NMI.transpose())[1]
	for i in range(7):
		a.text(i,.005,'$\it{t}$=%s\n%s' %(np.around(t_vals[i],2),log_p_value(p_vals[i])),{'fontsize':8},horizontalalignment='center')
	plt.savefig('/home/mbmbertolero/gene_expression/new_figures/nmi_sc_violin.pdf')
	plt.show()


	yeo_colors = pd.read_csv('/home/mbmbertolero/gene_expression/yeo_colors.txt',header=None,names=['name','r','g','b'],index_col=0)
	yeo_colors = yeo_colors.sort_values('name')
	yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.

	print scipy.stats.ttest_ind(fc_top_NMI.flatten(),sc_top_NMI.flatten())

	left, width = 0, 1
	bottom, height = 0, 1
	right = left + width
	top_f = bottom + height
	fig = plt.figure(figsize=(7.44094,9.72441),frameon=False)
	locs = (np.arange(65)/float(65))[10:]
	heat = make_heatmap(top_NMI.flatten(),'Reds',0.1,.25)
	c_idx = 0
	for n_idx,col,network in zip(range(7),np.linspace(.075,.925,7),names):
		for idx in range(n+1):
			if idx == n:
				fig.text(col*(left+right), (float(locs[idx])*(bottom+top_f)), network,horizontalalignment='center',verticalalignment='center',fontsize=10, color=(0,0,0))
				continue
			else:
				t = top[n_idx,idx] + ', ' + str(top_NMI[n_idx,idx])
				if top_FDR[n_idx,idx] < .05: t = t + '*'
				fig.text(col*(left+right), (float(locs[idx])*(bottom+top_f)), t,horizontalalignment='center',verticalalignment='center',fontsize=7, color=heat[c_idx])
				c_idx = c_idx + 1
	plt.savefig('/home/mbmbertolero/gene_expression/new_figures/%s_top_paris_genes.pdf'%(matrix))
	plt.show()

	m,top,top_FDR,top_NMI = get_paris(matrix,n)
	membership = np.loadtxt('/home/mbmbertolero/data/gene_expression/ann/%s_membership.txt'%(matrix))
	ranks = np.loadtxt('/home/mbmbertolero/data/gene_expression/ann/%s_gene_ranks.txt'%(matrix))
	m_ranks = np.zeros((7,ranks.shape[1]))
	branks = ranks.copy()
	branks[branks>0.] = 1
	for i in range(7):
		m_ranks[i] = np.nansum(branks[membership==i],axis=0) / len(membership[membership==i])
	print pearsonr(np.argmax(m_ranks,axis=0),np.argmax(m,axis=0))
	# max_ranks = np.argmax(m_ranks,axis=0)
	# max_nmi = np.argmax(m,axis=0)
	# rank_nmi_heatmap = np.zeros((7,7))



	# for i in range(size[1]):
	# 	rank_nmi_heatmap[max_ranks[i],max_nmi[i]] = rank_nmi_heatmap[max_ranks[i],max_nmi[i]] + 1
	# # plt.scatter(np.argmax(m_ranks,axis=0)+np.random.normal(-.05, .05, m_ranks.shape[1]),np.argmax(m,axis=0) + np.random.normal(-.05,.05, m_ranks.shape[1]),alpha=.15,linewidths=0.0,edgecolors='none',color=c)
	# sns.heatmap(rank_nmi_heatmap,vmin=0,vmax=2000,square=True)
	# # sns.plt.ylabel("Community the gene's coexpression most frequently fits",{'size':12})
	# # sns.plt.xlabel('Community the gene is most predictive of (PARIS RNMI)',{'size':12})
	# sns.plt.yticks(range(7),names,rotation=360)
	# sns.plt.xticks(range(7),names,rotation=90)
	# plt.tight_layout()
	# plt.savefig('/home/mbmbertolero/gene_expression/new_figures/paris_corr_with_rank_%s.pdf'%(matrix))
	# plt.show()
	# # plt.close()


	m = np.corrcoef(m)
	np.fill_diagonal(m,np.nan)
	sum_m = np.nansum(m,axis=0)
	sns.heatmap(m[np.argsort(sum_m)][:,np.argsort(sum_m)],vmin=-0.3,vmax=0.00,cmap='Blues_r',square=True)
	sns.plt.yticks(range(7),names[np.argsort(sum_m)],rotation=360)
	sns.plt.xticks(range(7),names[np.argsort(sum_m)],rotation=90)
	plt.tight_layout()
	plt.savefig('/home/mbmbertolero/gene_expression/new_figures/paris_corr_%s.pdf'%(matrix))
	plt.show()

def sge_perm_n_genes():
	run_type = 'find_genes'
	for corr_method in ['spearmanr','pearsonr']:
		# for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		for n_genes in [50,100,200]:
			for matrix in ['fc','sc']:
				for use_prs in [False]:
					for norm in [True]:
						for node in range(200):
							if node in ignore_nodes:continue
							# if len(np.load('/home/mbmbertolero/data/gene_expression/final_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,False,True,node,n_genes,False,True,corr_method))) > 0:
							# 	continue
							# else: print node
							# print '/home/mbmbertolero/gene_expression/gene_expression_analysis.py -r find_genes -m %s -n %s -n_genes %s -corr_method %s -norm %s -use_prs %s'%(matrix,node,n_genes,corr_method,norm,use_prs)
							# 1/0
							os.system('qsub -q basic.q -N fg_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ python /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r find_genes -m %s -n %s -n_genes %s -corr_method %s -norm %s -use_prs %s'%(node,matrix,node,n_genes,corr_method,norm,use_prs))

def sge_random():
	for corr_method in ['pearsonr']:
		for n_genes in [15,25,35,50,75,100,125,150,175,200]:
		# for n_genes in [50]:
			for matrix in ['fc','sc']:
				for use_prs in [False]:
					for norm in [True]:
						for node in range(200):
							if node in ignore_nodes:continue
							os.system('qsub -q all.q,basic.q -N fg_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
							 python /home/mbmbertolero/gene_expression/gene_expression_analysis.py \
							 -r random_random -m %s -n %s -n_genes %s -corr_method %s -norm %s -use_prs %s'\
							 %(node,matrix,node,n_genes,corr_method,norm,use_prs))
							os.system('qsub -q all.q,basic.q -N fg_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
							 python /home/mbmbertolero/gene_expression/gene_expression_analysis.py \
							 -r random -m %s -n %s -n_genes %s -corr_method %s -norm %s -use_prs %s'\
							 %(node,matrix,node,n_genes,corr_method,norm,use_prs))

def filter_snps():

	"""
	Identification of individuals with elevated missing data rates or outlying heterozygosity rate
	"""

	c = '/home/mbmbertolero/data/plink-1.07-x86_64/./plink --bfile /home/mbmbertolero/ncbi/dbGaP-18176/matrix2/MEGA_Chip --missing \
	--out /home/mbmbertolero/gene_expression/snp_results/raw-GWA-data'
	os.system(c)
	
	c = '/home/mbmbertolero/data/plink-1.07-x86_64/./plink --bfile /home/mbmbertolero/ncbi/dbGaP-18176/matrix2/MEGA_Chip --het --out raw-GWA-data'
	os.system(c)

	c = 'R CMD BATCH imiss-vs-het.Rscript'
	os.system('command')

def prep_gwas(node,matrix='fc',components='edges'):
	(bim, fam, G) = read_plink('/home/mbmbertolero/ncbi/dbGaP-18176/matrix/MEGA_Chip')
	fam.iid = fam.iid.astype('int64')
	df = pd.read_csv('//home/mbmbertolero/hcp/S1200.csv')
	df = df.rename(index=str, columns={"Subject": "iid"})
	df = pd.merge(df,fam,on=['iid'],how='inner')
	fam = fam.drop('i',axis=1).drop('father',axis=1).drop('mother',axis=1).drop('gender',axis=1).drop('trait',axis=1)
	
	if components == 7 or components == 17:
		matrices = []
		for s in df.iid.values:
			try: 
				m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_%s_matrix.npy'%(s,matrix))
				np.fill_diagonal(m,np.nan)
			except:
				m = np.zeros((400,400))
				m[:] = np.nan
			matrices.append(m[node])
		matrices = np.array(matrices)
		labels = yeo_membership(components)
		for n in range(components):
			pheno = np.nanmean(matrices[:,np.where(labels==n)[0]],axis=1)
			pheno[np.isnan(pheno)] = -9
			fam[str(n)] = pheno

	if components == 'edges':
		matrices = np.zeros((len(df.iid.values),400,400))
		for sidx,s in enumerate(df.iid.values):
			try: 
				m = np.load('/home/mbmbertolero/gene_expression/data/matrices/%s_%s_matrix.npy'%(s,matrix))
				np.fill_diagonal(m,np.nan)
			except:
				m = np.zeros((400,400))
				m[:] = np.nan
			matrices[sidx] = m
		mean = np.nanmean(matrices,axis=0)
		mean = brain_graphs.threshold(mean,0.05,mst=True)
		real_edges = np.argwhere(mean[node]>0)
		matrices[np.isnan(matrices)] = -9
		for n in real_edges:
			fam[str(n)] = matrices[:,node,n]


	if components == 2:
		fam['pc'] = -9.
		fam['wcs'] = -9.
		for sidx,subject in enumerate(fam.iid):
			try:
				fam['pc'][sidx] = np.load('/home/mbmbertolero/gene_expression/data//results/%s_%s_pcs.npy' %(subject,matrix))[node]
				fam['wcs'][sidx] = np.load('/home/mbmbertolero/gene_expression/data//results/%s_%s_wmds.npy' %(subject,matrix))[node]
			except:
				continue
	
	pheno_fn = "/home/mbmbertolero/gene_expression/snp_results/%s_%s_phenotype.txt"%(node,matrix)
	fam.to_csv(pheno_fn,sep=' ',header=None,index=False)

def snp_2_gene_idx():
	try:
		with open("/home/mbmbertolero/gene_expression/indices", "rb") as fp: indices = pickle.load(fp)
	except:
		gene_names = get_genes()
		df = pd.read_csv('/home/mbmbertolero/gene_expression/snp_results/snp_%s_%s_%s.P%s.assoc.linear'%('edges',0,'fc',1),header=0,sep='\s+',memory_map=True,usecols=[1,6],engine='c',low_memory=False)
		gene_names = get_genes()
		snp2genes_df = snp2genes()
		snp2genes_df.rename(columns={'snp': 'SNP'}, inplace=True)
		snp2genes_df=snp2genes_df.drop('chrom',axis=1)
		df = pd.merge(df,snp2genes_df,on='SNP',how='left')		
		df = df.dropna()
		indices = []
		for gidx,gene in enumerate(gene_names):
			print gidx
			indices.append(np.where(df['closest']==gene))
		with open("/home/mbmbertolero/gene_expression/indices", "wb") as fp: pickle.dump(indices, fp)
	return indices

def check_snp_2_gene_idx():
	with open("/home/mbmbertolero/gene_expression/indices", "rb") as fp: old_indices = pickle.load(fp)
	gene_names = get_genes()
	df = pd.read_csv('/home/mbmbertolero/gene_expression/snp_results/snp_%s_%s_%s.P%s.assoc.linear'%('edges',0,'fc',1),header=0,sep='\s+',memory_map=True,usecols=[1,6],engine='c',low_memory=False)
	gene_names = get_genes()
	snp2genes_df = snp2genes()
	snp2genes_df.rename(columns={'snp': 'SNP'}, inplace=True)
	snp2genes_df=snp2genes_df.drop('chrom',axis=1)
	df = pd.merge(df,snp2genes_df,on='SNP',how='left')		
	df = df.dropna()
	for gidx,gene in enumerate(gene_names):
		print gidx
		assert (old_indices[gidx][0] == np.where(df['closest']==gene)[0]).all()

def csv_to_npy(node,matrix='fc',components='edges'):
	print node
	if components == 'edges': 
		eiter = len(glob.glob('/home/mbmbertolero/gene_expression/snp_results/snp_%s_%s_%s.P**.assoc.linear'%(components,node,matrix)))
	if components == 2 or components == 7 or components == 17: eiter = components
	for edge in range(eiter):
		print edge
		if edge == 0: 
			df = pd.read_csv('/home/mbmbertolero/gene_expression/snp_results/snp_%s_%s_%s.P%s.assoc.linear'%(components,node,matrix,edge+1),header=0,sep='\s+',memory_map=True,usecols=[1,6],engine='c',low_memory=False)
			df.BETA = abs(df.BETA.values)
		else:
			tdf = pd.read_csv('/home/mbmbertolero/gene_expression/snp_results/snp_%s_%s_%s.P%s.assoc.linear'%(components,node,matrix,edge+1),header=0,sep='\s+',memory_map=True,usecols=[1,6],engine='c',low_memory=False)
			df.BETA = df.BETA.values + abs(tdf.BETA.values)
			del tdf
	df.BETA = df.BETA.values / float(eiter)
	gene_names = get_genes()
	snp2genes_df = snp2genes()
	snp2genes_df.rename(columns={'snp': 'SNP'}, inplace=True)
	snp2genes_df=snp2genes_df.drop('chrom',axis=1)

	df = pd.merge(df,snp2genes_df,on='SNP',how='left')
	df = df.dropna()

	indices = snp_2_gene_idx()
	
	df = df.values.transpose()[1]

	snp_by_gene = np.zeros((len(gene_names)))
	snp_by_gene[:] = np.nan
	for gidx,gene in enumerate(gene_names):
		if len(indices[gidx][0]) == 0:continue
		snp_by_gene[gidx] = np.mean(df[indices[gidx]])
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/%s_%s_%s_gene_snps_mean'%(components,node,matrix),snp_by_gene)

def snp2genes():
	try: bim = pd.read_csv('/home/mbmbertolero/gene_expression/snp2gene.csv')
	except:
		mapping = pd.DataFrame(columns=['snp','gene'])
		our_genes = get_genes()
		genes = pd.read_csv('/home/mbmbertolero/ncbi/glist-hg19',sep=' ',header=None)
		genes['mean pos'] = np.nanmean([genes[1],genes[2]],axis=0)
		(bim, fam, G) = read_plink('/home/mbmbertolero/ncbi/dbGaP-18176/matrix/MEGA_Chip')

		closest = []
		for snp,pos in zip(bim.snp.values,bim.pos.values):
			closest.append(genes[3][np.argmin(abs(genes['mean pos'].values-pos))])
		bim['closest'] = closest

		bim = bim.drop('pos',axis=1).drop('cm',axis=1).drop('a0',axis=1).drop('a1',axis=1).drop('i',axis=1)
		bim.to_csv('/home/mbmbertolero/gene_expression/snp2gene.csv',index=False)
	return bim

def gwas_sge(runtype='mean'):
	# for matrix in ['fc','sc']:
	os.system('cp /home/mbmbertolero/gene_expression/gene_expression_analysis.py /home/mbmbertolero/gene_expression/gene_expression_analysis_sge.py')
	for matrix in ['fc','sc']:
		for node in range(200):
			if runtype =='prep': 
				os.system("qsub -q all.q,basic.q -l h_vmem=6G,s_vmem=6G -N gwas%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis_sge.py -r pgwas -m %s -node %s"%(node,matrix,node))
			if runtype =='run':
				pheno_fn = "/home/mbmbertolero/gene_expression/snp_results/%s_%s_phenotype.txt"%(node,matrix)
				command = '/home/mbmbertolero/data/plink-1.07-x86_64/./plink --noweb --bfile /home/mbmbertolero/ncbi/dbGaP-18176/matrix2/MEGA_Chip --nonfounders --maf 0.01 --geno 0.05 --hwe 0.00001 --pheno %s --linear --all-pheno --out /home/mbmbertolero/gene_expression/snp_results/snp_%s_%s_%s'%(pheno_fn,components,node,matrix)
				os.system("qsub -q all.q,basic.q,himem.q -l h_vmem=3G,s_vmem=3G -N gwas%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ %s"%(node,command))
			if runtype == 'mean':
				os.system("qsub -q all.q,basic.q -l h_vmem=2G,s_vmem=2G -N ag%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r collapse -m %s -node %s"%(node,matrix,node))
	# 1/0
	# os.system("qsub -q all.q,basic.q -l h_vmem=10G,s_vmem=10G -N a_fc -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r top -m fc")
	# os.system("qsub -q all.q,basic.q -l h_vmem=10G,s_vmem=10G -N a_sc -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r top -m sc")

def snp2expression_analysis(node,matrix='fc',components=7,distance=True,use_prs=False,norm=True,corr_method='pearsonr'):
	global nodes_genes
	global snp2genes_df
	gene_names = get_genes()
	snp2genes_df = snp2genes()
	snp2genes_df.rename(columns={'snp': 'SNP'}, inplace=True)
	snp2genes_df=snp2genes_df.drop('chrom',axis=1)
	if components == 0: eiter = 400
	else: eiter = components
	df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s_snp_df.csv'%(node,components,matrix))
	snp_by_gene = np.zeros((len(gene_names),eiter))
	snp_by_gene[:,:] = np.nan
	snp_by_gene_max = np.zeros((len(gene_names),eiter))
	snp_by_gene_max[:,:] = np.nan
	# result_df = pd.merge(df[df.EDGE==edge],snp2genes_df,on='SNP',how='left')
	result_df = pd.merge(df,snp2genes_df,on='SNP',how='left')
	result_df = result_df.dropna()
	for gidx,gene in enumerate(gene_names):
		print gidx
		try: snp_by_gene[gidx] = result_df[result_df.closest==gene].groupby('EDGE')['STAT'].mean().values.reshape(-1)
		except: snp_by_gene[gidx] = np.nan
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/%s_%s_%s_gene_snps_mean'%(components,node,matrix),snp_by_gene)

def top_expression_snp(components='edges',matrix='fc',distance = True,use_prs = False,norm=True,corr_method = 'pearsonr'):
	n_nodes = 200
	gene_names = get_genes()
	node_by_snp = np.zeros((n_nodes,len(gene_names)))
	node_by_snp[:,:] = np.nan
	node_by_fit = np.zeros((n_nodes,len(gene_names))).astype(bool)
	
	for node in range(n_nodes):
		try:
			node_by_snp[node] = np.load('/home/mbmbertolero/data/gene_expression/snp_results/%s_%s_%s_gene_snps_mean.npy'%(components,node,matrix))
			for n_genes in [15,25,35,50,75,100,125,150,175,200]:
				node_by_fit[node,np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%(matrix,False,distance,node,n_genes,use_prs,norm,corr_method))[-1]] = True
		except:continue

	num_fits = np.sum(node_by_fit,axis=0)
	top_fits = np.zeros((len(gene_names))).astype(bool)
	top_fits[np.where(num_fits>=np.median(num_fits))[0]] = True
	# top_fits[np.where(num_fits>=15)] = True
	x = node_by_snp[:,top_fits==True].flatten()
	y = node_by_snp[:,top_fits==False].flatten()
	x = x[np.isnan(x) == False]
	y = y[np.isnan(y) == False]
	print scipy.stats.ttest_ind(x,y)

	# 1/0
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/top_fits_%s'%(matrix),top_fits)
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/num_fits_%s'%(matrix),num_fits)
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_%s'%(matrix),node_by_snp)
	

	mean_snp = np.nanmean(node_by_snp,axis=1)
	corr_method = 'pearsonr'
	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	subjects = get_subjects(done='matrices')
	b = behavior(subjects,all_b=True)
	columns=['node','snp_r','fit_r','behavioral_measure']
	behavior_df = pd.DataFrame(columns=columns)
	fit_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/%s_True_%s_False_fits_df.csv'%(matrix,corr_method))
	for task in b.columns.values:
		fc_corrs = np.load('/home/mbmbertolero/data/gene_expression/results/%s_%s_corrs.npy'%(task,matrix))
		fc_r = nan_pearsonr(np.nanmean(fc_corrs.reshape((400,400)),axis=0)[:n_nodes],mean_snp)
		fit = np.zeros(200)
		fit[:] = np.nan
		fit[mask] = fit_df.groupby('node').fit.mean()
		fit = fit[:n_nodes]
		fit_r = nan_pearsonr(np.nanmean(fc_corrs.reshape((400,400)),axis=0)[:n_nodes],fit)
		behavior_df = behavior_df.append(pd.DataFrame(np.array([['functional'],[fc_r[0]],[fit_r[0]],[task]]).transpose(),columns=columns))
	behavior_df.snp_r = behavior_df.snp_r.values.astype(float)
	behavior_df.fit_r = behavior_df.fit_r.values.astype(float)

	np.save('/home/mbmbertolero/data/gene_expression/snp_results/behav_fit_r_%s'%(matrix),behavior_df.fit_r.values.astype(float))
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/behav_snp_r_%s'%(matrix),behavior_df.snp_r.values.astype(float))
	print pearsonr(behavior_df.snp_r,behavior_df.fit_r)

	ts = np.zeros((200))
	null_ts = np.zeros((200,100))
	# node_by_fit = node_by_fit.sum(axis=0)
	# node_by_fit[node_by_fit==10] = 1
	# # node_by_fit[node_by_fit!=0] = 1
	# node_by_fit = node_by_fit.astype(bool)
	for node in range(200):
		print node
		x = node_by_snp[node,node_by_fit[node]==True]
		y = node_by_snp[node,node_by_fit[node]==False]
		x = x[np.isnan(x) == False]
		y = y[np.isnan(y) == False]
		ts[node] = scipy.stats.ttest_ind(x,y)[0]
		n_genes = len(np.where(node_by_fit[node]==True)[0])
		for null_run in range(100):
			np.random.seed(int(null_run))
			random_node_by_fit = np.zeros((len(gene_names))).astype(bool)
			random_node_by_fit[np.random.choice(len(gene_names),n_genes,replace=False)] = True
			x = node_by_snp[node,random_node_by_fit==True]
			y = node_by_snp[node,random_node_by_fit==False]
			x = x[np.isnan(x) == False]
			y = y[np.isnan(y) == False]

			null_ts[node,null_run] = scipy.stats.ttest_ind(x,y)[0]
	print scipy.stats.ttest_ind(ts[np.isnan(ts)==False],null_ts[np.isnan(null_ts)==False])
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/null_ts_%s'%(matrix),null_ts)
	np.save('/home/mbmbertolero/data/gene_expression/snp_results/ts_%s'%(matrix),ts)

def snp_figure():
	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	fcpal = sns.cubehelix_palette(10, rot=.25, light=.7)
	scpal = sns.cubehelix_palette(10, rot=-.25, light=.7)
	fc_top_fits = np.load('/home/mbmbertolero/data/gene_expression/snp_results/top_fits_fc.npy')
	sc_top_fits = np.load('/home/mbmbertolero/data/gene_expression/snp_results/top_fits_sc.npy')
	fc_top_fits_n = np.load('/home/mbmbertolero/data/gene_expression/snp_results/num_fits_fc.npy')
	sc_top_fits_n = np.load('/home/mbmbertolero/data/gene_expression/snp_results/num_fits_sc.npy')
	fc_node_by_snp = np.load('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_fc.npy')
	sc_node_by_snp = np.load('/home/mbmbertolero/data/gene_expression/snp_results/node_by_snp_sc.npy')

	n_nodes = 200
	gene_names = get_genes()
	fc_node_by_fit_sum = np.zeros((10,n_nodes,len(gene_names))).astype(int)
	sc_node_by_fit_sum = np.zeros((10,n_nodes,len(gene_names))).astype(int)
	
	for node in range(n_nodes):
		try:
			for idx,n_genes in enumerate([15,25,35,50,75,100,125,150,175,200]):
				fc_node_by_fit_sum[idx,node,np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%('fc',False,distance,node,n_genes,use_prs,norm,corr_method))[-1]] = 1
				sc_node_by_fit_sum[idx,node,np.load('/home/mbmbertolero/data/gene_expression/norm_results/SA_fit_all_%s_%s_%s_%s_%s_%s_%s_%s.npy'%('sc',False,distance,node,n_genes,use_prs,norm,corr_method))[-1]] = 1
		except:continue

	yeo_colors = pd.read_csv('/home/mbmbertolero/gene_expression/yeo_colors.txt',header=None,names=['name','r','g','b'],index_col=0)
	yeo_colors = np.array([yeo_colors['r'],yeo_colors['g'],yeo_colors['b']]).transpose() /256.
	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default']
	
	labels = yeo_membership(7)
	network_colors = []
	for c in range(200):
		network_colors.append(yeo_colors[labels[c]])

	for null in ['shift','spatial']:
		fc_e_nulls = []
		fc_snp_nulls = []
		sc_e_nulls = []
		sc_snp_nulls = []
		for matrix in ['fc','sc']:
			labels = yeo_membership(17)
			n_nets = 16
			if matrix == 'fc':
					em = np.corrcoef(np.sum(fc_node_by_fit_sum,axis=0))
					snp_m = np.corrcoef(fc_node_by_snp[:,np.isnan(fc_node_by_snp[0])==False])
			if matrix == 'sc':
					em = np.corrcoef(np.sum(sc_node_by_fit_sum,axis=0))
					snp_m = np.corrcoef(sc_node_by_snp[:,np.isnan(sc_node_by_snp[0])==False])
			
			np.fill_diagonal(snp_m,0.0)
			np.fill_diagonal(em,0.0)
			
			em[np.isnan(em)] = 0.0
			snp_m[np.isnan(snp_m)] = 0.0
			snp_m[snp_m==1] = 0.0
			
			snp_m = np.arctanh(snp_m)
			em = np.arctanh(em)

			e_graph = brain_graphs.matrix_to_igraph(em,0.15)
			snp_graph = brain_graphs.matrix_to_igraph(snp_m,0.15)
			pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
			print nan_pearsonr(brain_graphs.brain_graph(snp_graph.community_fastgreedy(weights='weight').as_clustering()).pc,pc[:200])
			print nan_pearsonr(brain_graphs.brain_graph(e_graph.community_fastgreedy(weights='weight').as_clustering()).pc,pc[:200])

			print nan_pearsonr(brain_graphs.brain_graph(VertexClustering(snp_graph,labels)).pc,pc[:200])
			print nan_pearsonr(brain_graphs.brain_graph(VertexClustering(e_graph,labels)).pc,pc[:200])

			if matrix == 'fc':
				fc_snp_q_e = snp_graph.community_infomap(edge_weights='weight').modularity
				fc_e_q_e = e_graph.community_infomap(edge_weights='weight').modularity
				fc_snp_q = snp_graph.modularity(labels, weights='weight')
				fc_e_q = e_graph.modularity(labels, weights='weight')
			if matrix == 'sc':
				sc_snp_q = snp_graph.modularity(labels, weights='weight')
				sc_e_q = e_graph.modularity(labels, weights='weight')
				sc_snp_q_e = snp_graph.community_infomap(edge_weights='weight').modularity
				sc_e_q_e = e_graph.community_infomap(edge_weights='weight').modularity


			dm = atlas_distance()[:200,:200]
			dm = abs(dm - dm.max())
			i = 0
			break_flag = False
			while True:
				# print i
				if null == 'shift':
					r_labels = labels.copy()
					# np.random.shuffle(r_labels)
					r_labels = np.roll(r_labels,np.random.choice(199,1))
				if null == 'spatial':
					r_labels = np.zeros((200))
					r_labels[:] = np.nan
					r_labels[np.random.choice(200,n_nets,replace=False).astype(int)] = np.arange(n_nets)
					for n in np.random.choice(200,200,replace=False):
						p = dm[n].copy()
						p[n] = 0
						p[np.isnan(r_labels)] = 0.0
						r_labels[n] = r_labels[np.argmax(p)]
					if min(np.unique(r_labels,return_counts=True)[1]) <= 5:continue
					if break_flag:
						break_flag = False
						continue
				if normalized_mutual_info_score(r_labels,labels)==1:continue
				i = i + 1 
				if matrix == 'sc':
					sc_snp_nulls.append(snp_graph.modularity(r_labels, weights='weight'))
					sc_e_nulls.append(e_graph.modularity(r_labels, weights='weight'))
				if matrix == 'fc':
					fc_snp_nulls.append(snp_graph.modularity(r_labels, weights='weight'))
					fc_e_nulls.append(e_graph.modularity(r_labels, weights='weight'))
				if i == 10000:break
		
		print scipy.stats.ttest_1samp(fc_snp_nulls,fc_snp_q)
		print scipy.stats.ttest_1samp(fc_e_nulls,fc_e_q)

		print scipy.stats.ttest_1samp(sc_snp_nulls,sc_snp_q)
		print scipy.stats.ttest_1samp(sc_e_nulls,sc_e_q)




		x,y,i,j = fc_e_nulls,fc_snp_nulls,sc_e_nulls,sc_snp_nulls
		x.append(fc_e_q)
		y.append(fc_snp_q)
		i.append(sc_e_q)
		j.append(sc_snp_q)
		x.append(fc_e_q_e)
		y.append(fc_snp_q_e)
		i.append(sc_e_q_e)
		j.append(sc_snp_q_e)
		g = sns.boxenplot(data=[x,y,i,j],palette=[fcpal[0],fcpal[7],scpal[0],scpal[7]])


		g.plot(0,fc_e_q,'s',fillstyle='none',markersize=10,c=fcpal[0])
		g.plot(1,fc_snp_q,'s',fillstyle='none',markersize=10,c=fcpal[7])
		g.plot(2,sc_e_q,'s',fillstyle='none',markersize=10,c=scpal[0])
		g.plot(3,sc_snp_q,'s',fillstyle='none',markersize=10,c=scpal[7])

		g.plot(0,fc_e_q_e,'bo',fillstyle='none',markersize=10,c=fcpal[0])
		g.plot(1,fc_snp_q_e,'bo',fillstyle='none',markersize=10,c=fcpal[7])
		g.plot(2,sc_e_q_e,'bo',fillstyle='none',markersize=10,c=scpal[0])
		g.plot(3,sc_snp_q_e,'bo',fillstyle='none',markersize=10,c=scpal[7])



		if null == 'shift':
			g.plot(0,.035,'bo',fillstyle='none',markersize=10,c='black')
			g.plot(0,.05,'s',fillstyle='none',markersize=10,c='black')
			g.text(0.5,.035,'emperical',{'fontsize':12,'fontweight':"bold"},horizontalalignment='center',verticalalignment='center')
			g.text(0.5,.05,'a priori',{'fontsize':12,'fontweight':"bold"},horizontalalignment='center',verticalalignment='center')


		if null == 'spatial':
			g.plot(2.5,.175,'s',fillstyle='none',markersize=10,c='black')
			g.plot(2.5,.16,'bo',fillstyle='none',markersize=10,c='black')
			g.text(3,.175,'a priori',{'fontsize':12,'fontweight':"bold"},horizontalalignment='center',verticalalignment='center')
			g.text(3,.16,'emperical',{'fontsize':12,'fontweight':"bold"},horizontalalignment='center',verticalalignment='center')

		g.text(.5,.16,'functional',{'fontsize':12,'color':fcpal[0],'fontweight':"bold"},horizontalalignment='center')
		g.text(2.5,.13,'structural',{'fontsize':12,'color':scpal[0],'fontweight':"bold"},horizontalalignment='center')
		# plt.tight_layout()
		plt.xticks(range(4),['coexpression matrix',"SNP matrix",'coexpression matrix',"SNP matrix"])
		sns.despine()
		sns.plt.tight_layout()
		plt.savefig('/home/mbmbertolero/gene_expression/figures/Q_v_%s_null.pdf'%(null))
		plt.show()
		# plt.close()


	for matrix in ['fc','sc']:
		if matrix == 'fc':
				em = np.corrcoef(np.sum(fc_node_by_fit_sum,axis=0))
				snp_m = np.corrcoef(fc_node_by_snp[:,np.isnan(fc_node_by_snp[0])==False])
		if matrix == 'sc':
				em = np.corrcoef(np.sum(sc_node_by_fit_sum,axis=0))
				snp_m = np.corrcoef(sc_node_by_snp[:,np.isnan(sc_node_by_snp[0])==False])
				
		np.fill_diagonal(snp_m,0.0)
		np.fill_diagonal(em,0.0)
		
		em[np.isnan(em)] = 0.0
		snp_m[np.isnan(snp_m)] = 0.0
		snp_m[snp_m==1] = 0.0
		
		snp_m = np.arctanh(snp_m)
		em = np.arctanh(em)
		import matplotlib.patches as patches
		g = sns.clustermap(em,vmin=.025,row_cluster=False, col_cluster=False,row_colors=network_colors, col_colors=network_colors,linewidths=0, xticklabels=False, yticklabels=False,rasterized=True)
		for i,network,color, in zip(np.arange(len(labels)),labels,network_colors):
			if network != labels[i - 1]:
				g.ax_heatmap.add_patch(patches.Rectangle((i+len(labels[labels==network]),(i+len(labels[labels==network]))),len(labels[labels==network]),len(labels[labels==network]),facecolor="none",edgecolor=color,linewidth="1",angle=180))
		plt.savefig('/home/mbmbertolero/gene_expression/figures/e_matrix_%s.pdf'%(matrix),rasterized=True)
		plt.close()
		g = sns.clustermap(snp_m,vmin=4.2,row_cluster=False, col_cluster=False,row_colors=network_colors, col_colors=network_colors,linewidths=0, xticklabels=False, yticklabels=False,rasterized=True)
		for i,network,color, in zip(np.arange(len(labels)),labels,network_colors):
			if network != labels[i - 1]:
				g.ax_heatmap.add_patch(patches.Rectangle((i+len(labels[labels==network]),(i+len(labels[labels==network]))),len(labels[labels==network]),len(labels[labels==network]),facecolor="none",edgecolor=color,linewidth="1",angle=180))
		plt.savefig('/home/mbmbertolero/gene_expression/figures/snp_matrix_%s.pdf'%(matrix))
		plt.close()






	fc_fits = np.zeros((58))
	n_genes = []
	for n in range(2,60):
		top_fits = np.zeros((fc_top_fits.shape[0])).astype(bool)
		top_fits[np.where(fc_top_fits_n>=n)[0]] = True
		x,y = fc_node_by_snp[:,top_fits==True].flatten(),fc_node_by_snp[:,top_fits==False].flatten()
		x = x[np.isnan(x) == False]
		y = y[np.isnan(y) == False]
		fc_fits[n-2] = scipy.stats.ttest_ind(x,y)[0]
		n_genes.append(len(top_fits[top_fits==True]))


	from scipy.interpolate import interp1d
	f = interp1d(range(2,60),fc_fits, kind='cubic')
	xnew = np.linspace(2,59, num=1000, endpoint=True)
	fig = sns.scatterplot(range(2,60),fc_fits,color=fcpal[0])
	sns.plt.plot(xnew,f(xnew),color=fcpal[0])
	for datapoint in [5,15,25,35,45,45,55]:
		fig.text(datapoint,fc_fits[datapoint],n_genes[datapoint],{'fontsize':12,'color':fcpal[2]},horizontalalignment='center')


	sc_fits = []
	for n in range(2,sc_top_fits_n.max()):
		top_fits = np.zeros((sc_top_fits.shape[0])).astype(bool)
		top_fits[np.where(sc_top_fits_n>=n)[0]] = True
		x,y = sc_node_by_snp[:,top_fits==True].flatten(),sc_node_by_snp[:,top_fits==False].flatten()
		x = x[np.isnan(x) == False]
		y = y[np.isnan(y) == False]
		sc_fits.append(scipy.stats.ttest_ind(x,y)[0])
	sc_fits = np.array(sc_fits)

	from scipy.interpolate import interp1d
	f = interp1d(range(2,31),sc_fits, kind='cubic')
	xnew = np.linspace(2,30, num=1000, endpoint=True)
	sns.scatterplot(range(2,31),sc_fits,color=scpal[0])
	sns.plt.plot(xnew,f(xnew),color=scpal[0])
	sns.plt.xlabel('number of nodes genes fit')
	sns.plt.ylabel('t-value, snp betas\nat coexpresson genes\nversus other genes')
	fig.text(45,-10,'functional',{'fontsize':12,'color':fcpal[0],'fontweight':"bold"},horizontalalignment='center')
	fig.text(45,-15,'structural',{'fontsize':12,'color':scpal[0],'fontweight':"bold"},horizontalalignment='center')
	plt.tight_layout()
	sns.despine()
	plt.savefig('/home/mbmbertolero/gene_expression/figures/fit_snp_v_others_dist.pdf')
	plt.show()
	plt.close()

	fc_nulls = np.load('/home/mbmbertolero/data/gene_expression/snp_results/null_ts_fc.npy')
	fc_ts = np.load('/home/mbmbertolero/data/gene_expression/snp_results/ts_fc.npy')

	sc_nulls = np.load('/home/mbmbertolero/data/gene_expression/snp_results/null_ts_sc.npy')
	sc_ts = np.load('/home/mbmbertolero/data/gene_expression/snp_results/ts_sc.npy')

	fc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_True_pearsonr_False_fits_df.csv')
	sc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_True_pearsonr_False_fits_df.csv')
	fc_df,sc_df = fc_df.groupby('node').mean(),sc_df.groupby('node').mean()
	fc_df['snp'],sc_df['snp'] = np.nanmean(fc_node_by_snp,axis=1)[mask],np.nanmean(sc_node_by_snp,axis=1)[mask]
	fc_df['membership'],sc_df['membership'] =yeo_membership(components=7)[mask],yeo_membership(components=7)[mask]


	fc_fit_r,sc_fit_r = np.load('/home/mbmbertolero/data/gene_expression/snp_results/behav_fit_r_fc.npy'),np.load('/home/mbmbertolero/data/gene_expression/snp_results/behav_fit_r_sc.npy')
	fc_snp_r,sc_snp_r = np.load('/home/mbmbertolero/data/gene_expression/snp_results/behav_snp_r_fc.npy'),np.load('/home/mbmbertolero/data/gene_expression/snp_results/behav_snp_r_sc.npy')


	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	smatrix = np.zeros((400,400))
	fmatrix = np.zeros((400,400))
	sm = scipy.io.loadmat('//home/mbmbertolero//gene_expression/heritability/sc/ACEfit_Par.mat')
	fm = scipy.io.loadmat('/home/mbmbertolero//gene_expression/heritability/fc/ACEfit_Par.mat')
	smatrix[np.triu_indices(400,1)] = np.array(sm['ACEfit_Par']['Stats'][0][0]).flatten()
	fmatrix[np.triu_indices(400,1)] = np.array(fm['ACEfit_Par']['Stats'][0][0]).flatten()
	smatrix = smatrix.transpose() + smatrix
	fmatrix = fmatrix.transpose() + fmatrix
	np.fill_diagonal(smatrix,np.nan)
	np.fill_diagonal(fmatrix,np.nan)
	sm = np.array(sm['ACEfit_Par']['Stats'][0][0]).flatten()
	fm = np.array(fm['ACEfit_Par']['Stats'][0][0]).flatten()
	fc_var = np.load('//home/mbmbertolero///gene_expression/fc_var.npy')[:200,:200]
	np.fill_diagonal(fc_var,np.nan)
	sc_var = np.load('//home/mbmbertolero///gene_expression/sc_var.npy')[:200,:200]
	np.fill_diagonal(sc_var,np.nan)
	sm_nodes, fm_nodes = np.nanmean(smatrix,axis=0)[:200],np.nanmean(fmatrix,axis=0)[:200]
	fpc = np.load('//home/mbmbertolero//gene_expression/results/fc_pc.npy')[:200]
	spc = np.load('//home/mbmbertolero//gene_expression/results/sc_pc.npy')[:200]


	x,y = fc_node_by_snp[:,fc_top_fits].flatten(),fc_node_by_snp[:,fc_top_fits==False].flatten()
	i,j = sc_node_by_snp[:,sc_top_fits].flatten(),sc_node_by_snp[:,sc_top_fits==False].flatten()
	x = x[np.isnan(x) == False]
	y = y[np.isnan(y) == False]
	i = i[np.isnan(i) == False]
	j = j[np.isnan(j) == False]

	t,p = scipy.stats.ttest_ind(x,y)
	t1,p1 = scipy.stats.ttest_ind(i,j)
	#plot
	g = sns.boxenplot(data=[x,y,i,j],palette=[fcpal[0],fcpal[7],scpal[0],scpal[7]])  
	g.text(.5,1.5,'$\it{t}$=%s\n%s' %(np.around(t,1),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
	g.text(2.5,1.5,'$\it{t}$=%s\n%s' %(np.around(t1,1),log_p_value(p1)),{'fontsize':12},horizontalalignment='center')
	g.text(.5,2,'functional',{'fontsize':12,'color':fcpal[0],'fontweight':"bold"},horizontalalignment='center')
	g.text(2.5,2,'structural',{'fontsize':12,'color':scpal[0],'fontweight':"bold"},horizontalalignment='center')
	plt.tight_layout()
	plt.xticks(range(4),['SNP betas @ genes\nthat fit coexpression',"other genes'\nSNP betas",'SNP betas @ genes\nthat fit coexpression',"other genes'\nSNP betas"])
	sns.despine()
	sns.plt.tight_layout()
	plt.savefig('/home/mbmbertolero/gene_expression/figures/fit_snp_v_others.pdf')
	plt.show()
	plt.close()

	x,y,i,j = fc_ts,fc_nulls,sc_ts,sc_nulls
	x = x[np.isnan(x) == False]
	y = y[np.isnan(y) == False]
	i = i[np.isnan(i) == False]
	j = j[np.isnan(j) == False]
	t,p = scipy.stats.ttest_ind(x,y)
	t1,p1 = scipy.stats.ttest_ind(i,j)
	#plot
	g = sns.boxenplot(data=[x,y,i,j],palette=[fcpal[0],fcpal[7],scpal[0],scpal[7]])  
	g.text(.5,1.5,'$\it{t}$=%s\n%s' %(np.around(t,1),log_p_value(p)),{'fontsize':12},horizontalalignment='center')
	g.text(2.5,1.5,'$\it{t}$=%s\n%s' %(np.around(t1,1),log_p_value(p1)),{'fontsize':12},horizontalalignment='center')
	g.text(.5,4,'functional',{'fontsize':12,'color':fcpal[0],'fontweight':"bold"},horizontalalignment='center')
	g.text(2.5,4,'structural',{'fontsize':12,'color':scpal[0],'fontweight':"bold"},horizontalalignment='center')
	plt.tight_layout()
	sns.despine()
	plt.xticks(range(4),['t-values',"null t-values",'t-values',"null t values"])
	plt.savefig('/home/mbmbertolero/gene_expression/figures/nodal_v_null.pdf')
	plt.show()
	plt.close()
	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default']
	membership = yeo_membership(components=7)

	sns.set(context="notebook",font='Open Sans',style='white',palette="pastel")
	fig,axes = sns.plt.subplots(2,2,figsize=(7.204724,7.204724),sharex='none',sharey='none')
	h2=sns.regplot(x=fc_fit_r,y=fc_snp_r,ax=axes[0][1],color=fcpal[0])
	h1=sns.regplot(x=sc_fit_r,y=sc_snp_r,ax=axes[0][0],color=scpal[0])
	h4=sns.regplot(x= fm_nodes,y=np.nanmean(fc_node_by_snp,axis=1),ax=axes[1][1],color=fcpal[3])
	h3=sns.regplot(x= sm_nodes,y=np.nanmean(sc_node_by_snp,axis=1),ax=axes[1][0],color=scpal[3])
	h3.set_ylim(.3272,.3281)

	r,p=pearsonr(sc_fit_r,sc_snp_r)
	h1.text(0.65,.1,convert_r_p(r,p),transform=h1.transAxes)
	
	r,p=pearsonr(fc_fit_r,fc_snp_r)
	h2.text(0.65,.1,convert_r_p(r,p),transform=h2.transAxes)
	
	r,p=pearsonr(sm_nodes,np.nanmean(sc_node_by_snp,axis=1))
	h4.text(0.65,.8,convert_r_p(r,p),transform=h3.transAxes)
	
	r,p=pearsonr(fm_nodes,np.nanmean(fc_node_by_snp,axis=1))
	h4.text(0.4,.8,convert_r_p(r,p),transform=h4.transAxes)

	h1.set_title('structural',color=scpal[2])
	h2.set_title('functional',color=fcpal[2])
	h1.set_ylabel("high snp nodes predict behavior")
	h2.set_xlabel("high fit nodes predict behavior")
	h1.set_xlabel("high fit nodes predict behavior")
	h4.set_xlabel("node's mean SNP beta")
	h3.set_xlabel("node's mean SNP beta")
	h3.set_ylabel("node's heritability")
	plt.tight_layout()
	plt.savefig('/home/mbmbertolero/gene_expression/figures/snp_corrs.pdf')
	plt.show()
	plt.close()
	vals = np.nanmean(fc_node_by_snp,axis=1)
	vals[np.isnan(vals)] = np.nanmean(vals)
	max_v,min_v = (np.std(vals)*1) + np.mean(vals), np.mean(vals) - (np.std(vals)*1)
	write_cifti(atlas_path,'snps',make_heatmap(np.array([vals,vals]).flatten(),dmin=min_v,dmax=max_v))

	fig,axes = sns.plt.subplots(1,2,figsize=(7.204724,7.204724/2.),sharex='none',sharey='none')
	h2 = sns.regplot(fc_df.groupby('node').fit.mean(),np.nanmean(fc_node_by_snp,axis=1)[mask],ax=axes[1],color=fcpal[0])
	h1 = sns.regplot(sc_df.groupby('node').fit.mean(),np.nanmean(sc_node_by_snp,axis=1)[mask],ax=axes[0],color=scpal[0])
	x=np.nanmean(sc_node_by_snp,axis=1)[mask]
	h1.set_ylim(x.min()-.0001,x.max()+.0001)
	r,p=pearsonr(sc_df.groupby('node').fit.mean(),np.nanmean(sc_node_by_snp,axis=1)[mask])
	h1.text(0.65,.8,convert_r_p(r,p),transform=h1.transAxes)
	r,p=pearsonr(fc_df.groupby('node').fit.mean(),np.nanmean(fc_node_by_snp,axis=1)[mask])
	h2.text(0.65,.1,convert_r_p(r,p),transform=h2.transAxes)

	h1.set_ylabel("node's mean SNP beta")
	h2.set_xlabel("node's genetic fit")
	h1.set_xlabel("node's genetic fit")
	h1.set_title('structural',color=scpal[2])
	h2.set_title('functional',color=fcpal[2])
	plt.tight_layout()
	plt.savefig('/home/mbmbertolero/gene_expression/figures/snp_fit_corrs.pdf')

	print pearsonr(fpc,np.nanmean(fc_node_by_snp,axis=1))
	print pearsonr(spc,np.nanmean(sc_node_by_snp,axis=1))

def find_top_pc_gene():
	r = np.zeros((gene_exp.shape[1]))
	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
	for i in range(len(r)):
		print i
		r[i] = nan_pearsonr(pc[:200],gene_exp[:,i])[0]
	vals = gene_exp[:,np.argmax(r)]
	vals[np.isnan(vals)] = np.nanmean(vals)
	max_v,min_v = (np.std(vals)*1.75) + np.mean(vals), np.mean(vals) - (np.std(vals)*1.75)
	write_cifti(atlas_path,'top_pc_gene',make_heatmap(np.array([vals,vals]).flatten(),dmin=min_v,dmax=max_v))

gene_exp = gene_expression_matrix()
wells_to_regions()
gene_exp = gene_exp[:200]
ignore_nodes = np.where(np.nansum(gene_exp,axis=1)==0)[0]
mask = np.ones((200)).astype(bool)
mask[ignore_nodes] = False

if run_type == 'find_genes': SA_find_genes(matrix,topological=False,distance=True,network=network,n_genes=n_genes,start_temp=.5,end_temp=.01,temp_step=.01,n_trys=1000,cores=0,use_prs=use_prs)
if run_type == 'predict_role': role_prediction(matrix,'genetic',distance=True,topological=False)
if run_type == 'graph_metrics':
	avg_graph_metrics('fc',topological=False,distance=True)
	avg_graph_metrics('sc',topological=False,distance=True)
	avg_graph_metrics('fc',topological=False,distance=False)
	avg_graph_metrics('sc',topological=False,distance=False)
	# graph_metrics(subject,'sc')
	# graph_metrics(subject,'fc')
if run_type == 'predict':
	subjects = get_subjects(done='graphs')
	df = behavior(subjects,all_b=True)
	task = df.columns.values[int(task)]
	performance('fc',task)
	performance('sc',task)
if run_type == 'pgwas': prep_gwas(node,matrix,'edges')
if run_type == 'random': null_genes(matrix,topological=False,distance=True,network=network,n_genes=n_genes,runs=10000,corr_method = 'pearsonr')
if run_type == 'random_random': random_genes(matrix,topological=False,distance=True,network=network,n_genes=n_genes,runs=1000,corr_method = 'pearsonr')
if run_type == 'collapse': csv_to_npy(node,matrix)
if run_type == 'top': top_expression_snp(matrix=matrix)













# run_type = 'find_genes'
# for matrix in ['fc']:
# 	for n_genes in [100]:
# 		for use_prs in [False]:
# 			for norm in [True]:
# 				for corr_method in ['pearsonr']:
# 					for node in range(200):
# 						if node in ignore_nodes: continue
# 						os.system('qsub -q all.q,basic.q -N fg_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/\
# 						 /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r find_genes -m %s -n %s -n_genes %s -corr_method %s -norm %s -use_prs %s'%(node,matrix,node,n_genes,corr_method,norm,use_prs))


# os.system('rm /home/mbmbertolero/gene_expression/data/ge_exp_True.npy')
# os.system('rm /home/mbmbertolero/gene_expression/data/yeo_400_gene_exp_True.npy')
# os.system('rm -f -r /home/mbmbertolero/gene_expression/norm_results')
# os.system('mkdir /home/mbmbertolero/gene_expression/norm_results')


# make_fit_df('fc',use_prs=False,norm=True,corr_method='pearsonr')
# make_fit_df('sc',use_prs=False,norm=True,corr_method='pearsonr')
# make_fit_df('fc',use_prs=False,norm=True,corr_method='spearmanr')
# make_fit_df('sc',use_prs=False,norm=True,corr_method='spearmanr')

# for corr_method in ['spearmanr']:
# 	sc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_True_%s_False_fits_df.csv'%(corr_method))
# 	fc_df = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_True_%s_False_fits_df.csv'%(corr_method))
# 	print pearsonr(fc_df.groupby('node').fit.mean(),fc_df.groupby('node')['participation coefficient'].mean())
# 	print pearsonr(sc_df.groupby('node').fit.mean(),sc_df.groupby('node')['participation coefficient'].mean())
# 	print fc_df.groupby('n_genes').fit.mean(), sc_df.groupby('n_genes').fit.mean()
# 	print scipy.stats.ttest_ind(fc_df.groupby('node').fit.mean(),sc_df.groupby('node').fit.mean())
# 	means = []
# 	pcs = []
# 	for n_genes in [15,25,35,50,75,100,200]:
# 		pcs.append(fc_df[fc_df.n_genes==n_genes].groupby('network')['participation coefficient'].mean().values)
# 		means.append(fc_df[fc_df.n_genes==n_genes].groupby('network')['fit'].mean().values)
# 		print n_genes
# 		print pearsonr(fc_df.fit[fc_df.n_genes==n_genes],fc_df['participation coefficient'][fc_df.n_genes==n_genes])
# 		print pearsonr(sc_df.fit[sc_df.n_genes==n_genes],sc_df['participation coefficient'][sc_df.n_genes==n_genes])
# 		# print scipy.stats.ttest_ind(fc_df.fit[fc_df.n_genes==n_genes],sc_df.fit[sc_df.n_genes==n_genes])


# heritability('fc')
# pearsonr(df.fit[:df.shape[0]],df['participation coefficient'][:df.shape[0]])

# pearsonr(df2.fit[:df.shape[0]],df2['participation coefficient'][:df.shape[0]])
# analyze_heritability()

		# if gidx in [1000,5000,10000,15000] : print gidx
		# g = df[:,df[2]==gene]
		# if len(g[0]) == 0:
		# 	snp_by_gene[gidx] = np.nan
		# 	continue
		# snp_by_gene[gidx] = g[0].reshape(nphenos,-1).mean(axis=1)
		# snp_by_gene[gidx] = df[df.closest==gene].groupby('EDGE')['BETA'].mean().values.reshape(-1)
