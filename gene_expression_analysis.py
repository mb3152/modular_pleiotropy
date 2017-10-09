#!/home/mbmbertolero/anaconda2/bin/python
import brain_graphs
import numpy.linalg as npl
import numpy as np
import os
import sys
import nibabel as nib
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from igraph import VertexClustering
from scipy.stats import pearsonr
from scipy.spatial import distance
from scipy import stats, linalg
from scipy.spatial.distance import pdist
import scipy.io
import scipy
import statsmodels.api as sm

from sklearn.neural_network import MLPRegressor, MLPClassifier

import glob
from itertools import combinations
import operator
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import patches
plt.rcParams['pdf.fonttype'] = 42
import math
import statsmodels.api as sm

global cortical
global cerebellar
global well_id_2_mni
global well_id_2_idx
global well_id_2_struct
global co_a_matrix
global gene_exp
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r',action='store',dest='run_type')
r = parser.parse_args()
locals().update(r.__dict__)

def swap(matrix,membership):
	membership = np.array(membership)
	swap_indices = []
	new_membership = np.zeros(len(membership))
	for i in np.unique(membership):
		for j in np.where(membership == i)[0]:
			swap_indices.append(j)
	return swap_indices

def nan_pearsonr(x,y):
	x = np.array(x)
	y = np.array(y)
	isnan = np.sum([x,y],axis=0)
	isnan = np.isnan(isnan) == False
	return pearsonr(x[isnan],y[isnan])

def three_d_dist(p1,p2):
	return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)  

def real_2_mm(target_image, real_pt):
	aff = target_image.affine
	return nib.affines.apply_affine(npl.inv(aff), real_pt)

def make_well_id_2_mni(subjects=['9861','178238266','178238316','178238373','15697','178238359']):
	print 'making well to MNI mapping'
	sys.stdout.flush()
	global well_id_2_mni
	global well_id_2_idx
	global well_id_2_struct
	template = nib.load('//share/apps/fsl/5.0.9/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz')
	well_id_to_mni = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/corrected_mni_coordinates.csv')
	#list of all the genes in the project
	# genes_df = pd.read_excel('/home/mbmbertolero/data/allen_gene_expression/alleninf/data/Allen_Genes.xlsx','sheet1')
	well_id_2_mni = {}
	well_idx = 0
	well_id_2_idx = {}
	well_id_2_struct = {}
	for subject in subjects:
		well_ids = pd.read_csv('/home/mbmbertolero/data/gene_expression/data/%s/SampleAnnot.csv'%(subject))['well_id'].values
		for well_id in well_ids:
			if well_id not in well_id_2_idx.keys():
				well_id_2_idx[well_id] = well_idx
				well_idx = well_idx + 1
			loc=np.where(well_id_to_mni['well_id']==well_id)[0][0]
			x,y,z = int(well_id_to_mni.corrected_mni_x[loc]),int(well_id_to_mni.corrected_mni_y[loc]),int(well_id_to_mni.corrected_mni_z[loc])
			mni_brain_loc = real_2_mm(template,[x,y,z])
			x,y,z = int(mni_brain_loc[0]),int(mni_brain_loc[1]),int(mni_brain_loc[2])
			well_id_2_mni[well_id] = [x,y,z]
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

def genes():
	# genes from previous study we want to look at
	gene_df = pd.read_excel('/home/mbmbertolero/data//gene_expression/data/Richiardi_Data_File_S2.xlsx')

	# load the probes, so we can remove some genes from the analysis
	probes = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/9861/Probes.csv')

	genes = []
	for probe in gene_df.probe_id.values:
		genes.append(probes.gene_symbol.values[probes.probe_name.values==probe])

	genes = np.unique(genes)
	genes.sort()

def gene_expression_matrix(subjects=['9861','178238266','178238316','178238373','15697','178238359']):
	try:
		well_by_expression_array = np.load('//home/mbmbertolero/data/gene_expression/data/ge_exp.npy')
	except:
		print 'making gene expression matrix'
		sys.stdout.flush()
		# genes from previous study we want to look at
		gene_df = pd.read_excel('//home/mbmbertolero/data//gene_expression/data/Richiardi_Data_File_S2.xlsx')

		# load the probes, so we can remove some genes from the analysis
		probes = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/Probes.csv'%(subjects[0]))

		genes = []
		for probe in gene_df.probe_id.values:
			genes.append(probes.gene_symbol.values[probes.probe_name.values==probe])

		genes = np.unique(genes)
		genes.sort()
		
		# we want to make numpy array, where d1 is the well (brain location) and d2 is the expression of all genes
		well_by_expression_array = np.zeros((len(well_id_2_idx),len(genes)))
		
		for subject in subjects:
			
			#load the full gene expression by well matrix; this contains many genes we don't want to look at, as well as mutliple expression values for each gene
			full_expression = np.array(pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/MicroarrayExpression.csv'%(subject),header=None,index_col=0))

			#load the wells, so we know where in the brain the expression is
			wells = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/SampleAnnot.csv'%(subject))['well_id'].values

			# load the probes, so we can remove some genes from the analysis
			probes = pd.read_csv('//home/mbmbertolero/data//gene_expression/data/%s/Probes.csv'%(subjects[0]))

			# final expression array, where we take the mean from each gene we care about from full_expression
			expression = np.zeros((len(np.unique(genes)),len(wells)))

			for expression_idx,gene in enumerate(genes):
				expression[expression_idx] = np.nanmean(full_expression[np.where(probes.gene_symbol.values==gene)],axis=0)

			# we want to normalize by mean expression in cortex, cerebellar, or sub-cortical
			cortical_a = np.zeros((len(wells))).astype(bool)
			cerebellar_a = np.zeros((len(wells))).astype(bool)
			sub_cortical_a = np.zeros((len(wells))).astype(bool)
			for idx,well in enumerate(wells):
				if well_id_2_struct == 'cortical':
					cortical_a[idx] = True
					continue
				elif well_id_2_struct == 'cerebellar':
					cerebellar_a[idx] = True
					continue
				else:
					sub_cortical_a[idx] = True
					continue
			for idx in range(expression.shape[0]):
				expression[idx][cortical_a] = expression[idx][cortical_a] - np.mean(expression[idx][cortical_a])
				expression[idx][cerebellar_a] = expression[idx][cerebellar_a] - np.mean(expression[idx][cerebellar_a])
				expression[idx][sub_cortical_a] = expression[idx][sub_cortical_a] - np.mean(expression[idx][sub_cortical_a])
			#now that we have normalized by location, 
			#turn into a well by gene expression matrix
			expression = expression.transpose() 
			for idx,well in enumerate(wells):
				well_by_expression_array[well_id_2_idx[well]] = expression[idx]
		np.save('//home/mbmbertolero/data//gene_expression/data/ge_exp.npy',well_by_expression_array)
	return well_by_expression_array

def wells_to_regions():
	print 'making well to region mapping'
	sys.stdout.flush()
	global gene_exp
	try: gene_exp = np.load('//home/mbmbertolero/data//gene_expression/data/yeo_400_gene_exp.npy')
	except:
		template = nib.load('//home/mbmbertolero/data//gene_expression/data/yeo_400.nii.gz').get_data()
		m = np.zeros((int(np.max(template)),gene_exp.shape[1]))
		for parcel in np.arange(np.max(template)):
			wells = []
			for idx,well in enumerate(well_id_2_mni.keys()):
				x,y,z, = well_id_2_mni[well]
				if template[x,y,z] == parcel + 1: wells.append(gene_exp[well_id_2_idx[well]])
			m[int(parcel)] = np.nanmean(wells,axis=0)
		gene_exp = m
		np.save('//home/mbmbertolero/data//gene_expression/data/yeo_400_gene_exp.npy',gene_exp)

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

gene_exp = gene_expression_matrix()

wells_to_regions()

def functional_connectivity(topological=False,distance=True,network=None):
	reduce_dict = {'VisCent':'Visual','VisPeri':'Visual','SomMotA':'Motor','SomMotB':'Motor','DorsAttnA':'Dorsal Attention','DorsAttnB':'Dorsal Attention','SalVentAttnA':'Ventral Attention','SalVentAttnB':'Ventral Attention','Limbic':'Limbic','ContA':'Control','ContB':'Control','ContC':'Control','DefaultA':'Default','DefaultB':'Default','DefaultC':'Default','TempPar':'Temporal Parietal'}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]
	try: matrix = np.load('/home/mbmbertolero/data/gene_expression/data/matrices/mean_fc.npy')
	except:
		matrix = []
		matrix_files = glob.glob('/home/mbmbertolero/data/gene_expression/fc_matrices/yeo/*REST*')
		for m in matrix_files:
			m = np.loadtxt(m)
			matrix.append(m.copy())
		matrix = np.nanmean(matrix,axis=0)
		np.save('/home/mbmbertolero/data/gene_expression/data/matrices/mean_fc.npy',matrix)
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

def structural_connectivity(topological=False,distance=True,network=None):
	reduce_dict = {'VisCent':'Visual','VisPeri':'Visual','SomMotA':'Motor','SomMotB':'Motor','DorsAttnA':'Dorsal Attention','DorsAttnB':'Dorsal Attention','SalVentAttnA':'Ventral Attention','SalVentAttnB':'Ventral Attention','Limbic':'Limbic','ContA':'Control','ContB':'Control','ContC':'Control','DefaultA':'Default','DefaultB':'Default','DefaultC':'Default','TempPar':'Temporal Parietal'}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/PUBLIC/yeo/MNI152/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]
	try: matrix = np.load('/home/mbmbertolero/data/gene_expression/data/matrices/sc_matrix.npy')
	except:
		matrix = []
		matrix_files = glob.glob('/home/mbmbertolero/data/gene_expression/sc_matrices/**/matrix.npy')
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

def single(ctype,topological=False,distance=True):
	df = pd.DataFrame(columns=['network','gene_co - fc'])
	for network in ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']:
		try:
			a_matrix = functional_connectivity(topological,distance,network)
			gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(ctype,topological,distance,network))[-1]
			gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
			mask = np.sum([gene_exp_matrix,a_matrix],axis=0)
			mask[np.isnan(mask)==True] = False
			mask[mask!=False] = True
			mask = mask.astype(bool)
			for i,j in zip(a_matrix[mask].flatten(),gene_exp_matrix[mask].flatten()):
				df= df.append(pd.DataFrame(np.array([[j-i],[network]]).transpose(),columns=['gene_co - fc','network']))
			# print network,scipy.stats.ttest_ind(a_matrix[mask].flatten(),gene_exp_matrix[mask].flatten())
			print network, pearsonr(a_matrix[mask].flatten(),gene_exp_matrix[mask].flatten())
		except: continue

def barplot(fitwhat,topological,distance):
	sns.set_style('whitegrid')
	sns.set(font='Helvetica')

	if fitwhat=='brainmap':
		a_matrix = brainmap_coact()
		gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_None.npy'%(fitwhat,topological,distance))[-1]
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
		diff_matrix = np.abs(a_matrix-gene_exp_matrix)	
	if fitwhat =='fc':
		a_matrix = functional_connectivity(topological,distance,None)
		gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_None.npy'%(fitwhat,topological,distance))[-1]
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])

	np.fill_diagonal(a_matrix,np.nan)
	
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]

	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']

	r = a_matrix.copy()
	# mean_fc = np.load('/home/mbmbertolero/data/gene_expression/data/matrices/mean_fc.npy')
	# r[mean_fc<.5] = np.nan
	mask = np.sum([a_matrix,gene_exp_matrix],axis=0)

	print pearsonr(a_matrix[np.isnan(mask)==False].flatten(),gene_exp_matrix[np.isnan(mask)==False].flatten())
	# 1/0
	r[np.isnan(mask)==False] = abs(sm.GLM(a_matrix[np.isnan(mask)==False].flatten(),sm.add_constant(gene_exp_matrix[np.isnan(mask)==False].flatten())).fit().resid_response)
	plot_df = pd.DataFrame(columns=['Error','Network'])
	for i in range(8):
		for e in np.nanmean(r,axis=0)[membership=='%s'%(i)]:
			plot_df = plot_df.append(pd.DataFrame(data=[[e,names[i]]],columns=['Error','Network']))
	plot_df['Error'] = plot_df['Error'].astype(float)
	sns.barplot(data=plot_df,y='Error',x='Network')
	sns.plt.xticks(rotation=90)
	sns.plt.tight_layout()
	# sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/error_%s_t_%s_d_%s.pdf'%(fitwhat,topological,distance))
	sns.plt.show()
	
	tiny_r = np.zeros((8,8))
	for i in range(8):
		for j in range(8):
			tiny_r[i,j] = np.nanmean(r[membership==str(i)][:,membership==str(j)])
		# tiny_r[j,i] = np.nanmean(r[membership==str(i)][:,membership==str(j)])
	# tiny_fc = np.zeros((7,7))
	# for i in range(7):
	# 	for j in range(7):
	# 		tiny_fc[i,j] = np.nanmean(a_matrix[membership==str(i)][:,membership==str(j)])
	# 	# tiny_r[j,i] = np.nanmean(r[membership==str(i)][:,membership==str(j)])
	# tiny_g = np.zeros((7,7))
	# for i in range(7):
	# 	for j in range(7):
	# 		tiny_g[i,j] = np.nanmean(gene_exp_matrix[membership==str(i)][:,membership==str(j)])
	# 	# tiny_r[j,i] = np.nanmean(r[membership==str(i)][:,membership==str(j)])

	# sns.plt.yticks(range(7),reversed(names),rotation=360)
	# sns.plt.xticks(range(7),names,rotation=90)
	# sns.plt.tight_layout()

	sns.heatmap(tiny_r,square=True,rasterized=True)
	sns.plt.yticks(range(7),reversed(names),rotation=360)
	sns.plt.xticks(range(7),names,rotation=90)
	sns.plt.tight_layout()
	# sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/error_matrix_%s_t_%s_d_%s.pdf'%(fitwhat,topological,distance))
	sns.plt.show()

def plot(fitwhat,topological,distance):

	sns.set_style('dark')
	sns.set(font='Helvetica',rc={'axes.facecolor':'.5','axes.grid': False})

	reduce_networks = False
	global gene_exp
	fig,subplots = sns.plt.subplots(1,2,figsize=(12,6))
	subplots = subplots.reshape(-1)
	if fitwhat=='brainmap':
		a_matrix = brainmap_coact()
		gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s.npy'%(fitwhat,topological,distance))[-1]
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
		diff_matrix = np.abs(a_matrix-gene_exp_matrix)

	if fitwhat == 'sc':
		sc = scipy.io.loadmat('/home/mbmbertolero/data/gene_expression/data/yeo_400.mat')['connectivity']
		a_matrix= np.zeros((400,400))
		for i,j in combinations(range(400),2):
			r = pearsonr(sc[i,:],sc[j,:])[0]
			a_matrix[i,j] = r
			a_matrix[j,i] = r
		gene_exp_matrix = np.load(glob.glob('/home/mbmbertolero/data/gene_expression/results/*sc*')[-1])
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
		diff_matrix = np.abs(a_matrix-gene_exp_matrix)

	if fitwhat == 'fc':
		a_matrix = functional_connectivity()
		gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s.npy'%(fitwhat,topological,distance))[-1]
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
		diff_matrix = np.abs(a_matrix-gene_exp_matrix)

	to_plot = [a_matrix,gene_exp_matrix]

	names = ['%s matrix'%(fitwhat),'gene coexpression matrix']
	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		if reduce_networks == True: membership[i] = reduce_dict[n.split('_')[2]]
		else:membership[i] = n.split('_')[2]
	
	for idx,matrix,name in zip(np.arange(len(to_plot)),to_plot,names):
		sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
		np.fill_diagonal(matrix,0.0)
		swap_indices = swap(matrix,membership)
		heatfig = sns.heatmap(matrix[swap_indices,:][:,swap_indices],yticklabels=[''],xticklabels=[''],square=True,rasterized=True,cbar=False,ax=subplots[idx])
		# Use matplotlib directly to emphasize known networks
		heatfig.set_title(name,fontdict={'fontsize':'large'})
		if idx == 0: heatfig.annotate('r=%s'%(nan_pearsonr(np.tril(a_matrix,-1).flatten(),np.tril(gene_exp_matrix,-1).flatten())[0]), xy=(1,1),**{'fontsize':'small'})
		for i,network in zip(np.arange(len(membership)),membership[swap_indices]):
			if network != membership[swap_indices][i - 1]:
				heatfig.figure.axes[idx].add_patch(patches.Rectangle((i+len(membership[membership==network]),len(membership)-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor='black',linewidth="1",angle=180))
	
	# idx = 2
	# matrix = diff_matrix
	# sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
	# np.fill_diagonal(matrix,0.0)
	# swap_indices = swap(matrix,membership)
	# heatfig = sns.heatmap(matrix[swap_indices,:][:,swap_indices],yticklabels=[''],xticklabels=[''],square=True,rasterized=True,cbar=False,ax=subplots[idx])
	# # Use matplotlib directly to emphasize known networks
	# heatfig.set_title('difference',fontdict={'fontsize':'large'})
	# # heatfig.annotate('r=%s'%(nan_pearsonr(np.tril(a_matrix,-1).flatten(),np.tril(gene_exp_matrix,-1).flatten())[0]), xy=(1,1),**{'fontsize':'small'})
	# for i,network in zip(np.arange(len(membership)),membership[swap_indices]):
	# 	if network != membership[swap_indices][i - 1]:
	# 		heatfig.figure.axes[idx].add_patch(patches.Rectangle((i+len(membership[membership==network]),len(membership)-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor='black',linewidth="1",angle=180))	

	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/%s.pdf'%(fitwhat))
	sns.plt.show()

def fit_matrix_multi(ignore_idx):
	sys.stdout.flush()
	global co_a_matrix
	global gene_exp
	temp_m = np.corrcoef(gene_exp[:,np.arange(gene_exp.shape[1])!=ignore_idx])
	np.fill_diagonal(temp_m,np.nan)
	return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]

def fit_matrix(matrix,topological=True,distance=True,network=None):
	try: result = np.load('/home/mbmbertolero/data/gene_expression/results/fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,network))
	except:
		global co_a_matrix
		global gene_exp
		if matrix == 'fc': co_a_matrix = functional_connectivity(topological,distance,network)
		if matrix == 'sc': co_a_matrix = structural_connectivity(topological,distance,network)
		np.fill_diagonal(co_a_matrix,0.0)
		co_a_matrix[np.isinf(co_a_matrix)] = np.nan
		cores= multiprocessing.cpu_count()-1
		pool = Pool(cores)
		result = pool.map(fit_matrix_multi,range(gene_exp.shape[1]))
		np.save('/home/mbmbertolero/data/gene_expression/results/fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,network),np.array(result))
	return np.array(result)

def fit_SA(indices_for_correlation):
	global co_a_matrix #grab the matrix we are working with
	global gene_exp #grab the "timeseries" of gene expression, well id by gene expression value shape
	#get correlation matrix for select genes
	temp_m = np.corrcoef(gene_exp[:,np.ix_(indices_for_correlation)][:,0,:])
	np.fill_diagonal(temp_m,np.nan)
	return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]

def SA_find_genes(matrix,topological=False,distance=True,network=None,n_genes=100,start_temp=.5,end_temp=.01,temp_step=.001,n_trys=100,cores=10):
	#set number of cores depending on which node we are on
	# cores= multiprocessing.cpu_count()-2
	
	# grab the coact/fc/sc matrix global variable, which we will assing the matrix we are anylzing to
	# we have to use global variable, as we want to use a lot of cores
	global co_a_matrix
	#load correlation matrix to fit genes coexp to, assign it to the global co_a_matrix variable
	if matrix == 'fc': co_a_matrix = functional_connectivity(topological,distance,network)
	if matrix == 'sc': co_a_matrix = structural_connectivity(topological,distance,network)
	np.fill_diagonal(co_a_matrix,np.nan) #remove diagonal

	# initally, start with the top n genes that increase fit
	# this was previously calcualted, it is just the r value between coexpression and the fc/sc/coact matrix
	# without that gene in the co-expression matrix, lower values means it probably helps with fit of coexpression
	gene_r_impact = 1 - fit_matrix(matrix,topological,distance,network)

	#calculate the fit on the top n genes that increase the fit, use as first estimate
	current_genes = np.arange(gene_exp.shape[1])[np.argsort(gene_r_impact)][-n_genes:]
	initial_m = np.corrcoef(gene_exp[:,np.ix_(current_genes)][:,0,:])
	current_r = nan_pearsonr(co_a_matrix.flatten(),initial_m.flatten())[0]
	#normalize increases/decreases to probabilities that sum to 1
	prs = gene_r_impact/float(np.sum(gene_r_impact))
	
	# set multiprocessing to have the number of cores requested
	pool = Pool(cores)
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
			remove_prs = 1-prs[current_genes]
			remove_prs = remove_prs/np.sum(remove_prs) #renormalize
			remove_genes = set(np.random.choice(current_genes,int(current_temp),False,remove_prs))
			# remove these genes
			for gene in remove_genes: temp_genes.remove(gene)
			# find genes to add, bias towards genes that increased fit
			while True: # we have to ensure it gets back to n_genes, since it can select genes aready in the set
				temp_genes.add(np.random.choice(gene_exp.shape[1],1,False,prs)[0])
				if len(temp_genes) == n_genes: break	
			# save it to the array we are going to send to the multiprocessing function
			potential_indices[i] = np.array(list(temp_genes))
		#send to fit function, which calculates fit of gene coexpression to whatever matrix was selected
		results = pool.map(fit_SA,potential_indices)
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
		if temp < end_temp: break
	np.save('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,network),np.array(save_results))

def nn_regress(node):
	global features
	global measure
	global layers
	global size
	print node
	sys.stdout.flush()
	model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=layers,alpha=1e-5,random_state=0)
	model.fit(features[np.arange(size)!=node],measure[np.arange(size)!=node])
	return model.predict(features[node].reshape(1, -1))

def nn_cater(node):
	global features
	global measure
	global layers
	global size
	print node
	sys.stdout.flush()
	model = MLPClassifier(solver='lbfgs',hidden_layer_sizes=layers,alpha=1e-5,random_state=0)
	model.fit(features[np.arange(size)!=node],measure[np.arange(size)!=node])
	return model.predict(features[node].reshape(1, -1))

def what_genes(cores):
	global features
	global measure
	global layers
	global layers_name
	global size
	sns.set_style('whitegrid')
	sns.set(font='Helvetica')
	ctype='fc'
	distance = True
	topological = False

	try:
		pc = np.load('/home/mbmbertolero/data/gene_expression/results/pc.npy')
		degree = np.load('/home/mbmbertolero/data/gene_expression/results/degree.npy')
		wcd = np.load('/home/mbmbertolero/data/gene_expression/results/wcd.npy')
	except:
		matrix = functional_connectivity(topological,distance,None)
		matrix = np.tril(matrix,-1)
		matrix = matrix + matrix.transpose()
		pc = []
		wcd = []
		degree = []
		for cost in np.arange(150,201)*0.001:
			g = brain_graphs.matrix_to_igraph(matrix.copy(),cost=cost,mst=True)
			g = brain_graphs.brain_graph(g.community_infomap(edge_weights='weight'))
			pc.append(g.pc)
			wcd.append(g.wmd)
			degree.append(g.community.graph.strength(weights='weight'))
		pc = np.nanmean(pc,axis=0)
		wcd = np.nanmean(wcd,axis=0)
		degree = np.nanmean(degree,axis=0)
		np.save('/home/mbmbertolero/data/gene_expression/results/pc.npy',pc)
		np.save('/home/mbmbertolero/data/gene_expression/results/degree.npy',degree)
		np.save('/home/mbmbertolero/data/gene_expression/results/wcd.npy',wcd)

	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]

	df = pd.DataFrame(columns=['gene','node','participation coefficient','within community strength','strength'])
	for node,name,pval,w,d in zip(range(400),membership,pc,wcd,degree):
		try:
			for gene in np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(ctype,topological,distance,node))[-1]:
				df= df.append(pd.DataFrame(np.array([[gene],[node],[pval],[w],[d]]).transpose(),columns=['gene','node','participation coefficient','within community strength','strength']))
		except:
			df= df.append(pd.DataFrame(np.array([[np.nan],[node],[pval],[w],[d]]).transpose(),columns=['gene','node','participation coefficient','within community strength','strength']))
	df.gene = df.gene.astype(float)
	df['participation coefficient'] = df['participation coefficient'].astype(float)
	df['strength'] = df['strength'].astype(float)
	df['within community strength'] = df['within community strength'].astype(float)

	unique = np.unique(df.gene[np.isnan(df.gene.values)==False])
	features = np.zeros((400,len(unique))).astype(bool)
	for node in range(400):
		g_array = unique
		g_b_array = unique.copy().astype(bool)
		g_b_array[:] = False
		for g in df.gene[df.node==node].values:
			g_b_array[g_array==g] = True
		features[node,:] = g_b_array
	features = features.astype(int)
	mask = np.sum(features,axis=1) == 100
	size = len(mask[mask==True])
	features = features[mask,:]

	reduce_dict = {'VisCent':'Visual','VisPeri':'Visual','SomMotA':'Motor','SomMotB':'Motor','DorsAttnA':'Dorsal Attention','DorsAttnB':'Dorsal Attention','SalVentAttnA':'Ventral Attention','SalVentAttnB':'Ventral Attention','Limbic':'Limbic','ContA':'Control','ContB':'Control','ContC':'Control','DefaultA':'Default','DefaultB':'Default','DefaultC':'Default','TempPar':'Default'}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]

	"""
	testing 1,2 1,2
	"""
	# for measure,name in zip([pc,wcd,degree,membership],['pc','wcd','degree','network']):
	# 	print name
	# 	measure = measure[mask]
	# 	results = []
	# 	for i in range(3):
	# 		m = np.zeros((len(measure))).astype(bool)
	# 		m[np.random.choice(range(len(measure)),300,False)] = True
	# 		if name == 'network': model = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(10,),alpha=1e-5,random_state=0)
	# 		else:model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(10,10,10,10,10,10,10,10,10,10,10,10,10,10),alpha=1e-5,random_state=0)
	# 		model.fit(features[m],measure[m])
	# 		if name =='network':
	# 			prediction = model.predict(features[m==False])
	# 			results.append(len(prediction[prediction==measure[m==False]]) / float(len(prediction)))
	# 		else:results.append(pearsonr(model.predict(features[m==False]),measure[m==False])[0])
	# 	print np.mean(results)
	# np.save('/home/mbmbertolero/data/gene_expression/results/nn%s%s.npy'%(layers_name,name),results)

	"""
	predict the nodal values of a node based on which genes maximize its fit to FC
	"""
	for measure,name in zip([pc,wcd,degree],['pc','wcd','degree']):
	# for measure,name in zip([pc],['pc']):
		measure = measure[mask]
		pool = Pool(cores)
		prediction = np.array(pool.map(nn_regress,range(size)))
		np.save('/home/mbmbertolero/data/gene_expression/results/nn%s_%s.npy'%(layers_name,name),prediction[:,0])

	"""
	predict the network memebership of the node based on which genes maximize its fit to FC
	"""
	membership = membership[mask]
	measure = membership
	pool = Pool(cores)
	prediction = np.array(pool.map(nn_cater,range(size)))
	np.save('/home/mbmbertolero/data/gene_expression/results/nn%s_network.npy'%(layers_name),prediction[:,0])

def plot_what_genes():
	sns.set_style('whitegrid')
	sns.set(font='Helvetica')
	ctype='fc'
	distance = True
	topological = False

	matrix = functional_connectivity(topological,distance,None)
	matrix = np.tril(matrix,-1)
	matrix = matrix + matrix.transpose()
	pc = np.load('/home/mbmbertolero/data/gene_expression/results/pc.npy')
	degree = np.load('/home/mbmbertolero/data/gene_expression/results/degree.npy')
	wcd = np.load('/home/mbmbertolero/data/gene_expression/results/wcd.npy')

	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]

	df = pd.DataFrame(columns=['gene','node','participation coefficient','within community strength','strength'])
	for node,name,pval,w,d in zip(range(400),membership,pc,wcd,degree):
		try:
			for gene in np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(ctype,topological,distance,node))[-1]:
				df= df.append(pd.DataFrame(np.array([[gene],[node],[pval],[w],[d]]).transpose(),columns=['gene','node','participation coefficient','within community strength','strength']))
		except:
			df= df.append(pd.DataFrame(np.array([[np.nan],[node],[pval],[w],[d]]).transpose(),columns=['gene','node','participation coefficient','within community strength','strength']))
	df.gene = df.gene.astype(float)
	df['participation coefficient'] = df['participation coefficient'].astype(float)
	df['strength'] = df['strength'].astype(float)
	df['within community strength'] = df['within community strength'].astype(float)

	"""
	predict the nodal values of a node based on which genes maximize its fit to FC
	"""

	unique = np.unique(df.gene[np.isnan(df.gene.values)==False])
	features = np.zeros((400,len(unique))).astype(bool)
	for node in range(400):
		g_array = unique
		g_b_array = unique.copy().astype(bool)
		g_b_array[:] = False
		for g in df.gene[df.node==node].values:
			g_b_array[g_array==g] = True
		features[node,:] = g_b_array
	features = features.astype(int)
	mask = np.sum(features,axis=1) == 100
	size = len(mask[mask==True])
	features = features[mask,:]

	reduce_dict = {'VisCent':'Visual','VisPeri':'Visual','SomMotA':'Motor','SomMotB':'Motor','DorsAttnA':'Dorsal Attention','DorsAttnB':'Dorsal Attention','SalVentAttnA':'Ventral Attention','SalVentAttnB':'Ventral Attention','Limbic':'Limbic','ContA':'Control','ContB':'Control','ContC':'Control','DefaultA':'Default','DefaultB':'Default','DefaultC':'Default','TempPar':'Default'}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	for i,n in enumerate(yeo_df):
		membership[i] = reduce_dict[n.split('_')[2]]
	
	membership = membership[mask]
	pc = pc[mask]
	degree = degree[mask]
	wcd = wcd[mask]

	result_df = pd.DataFrame(columns=['prediction_acc','prediction_type','layers','nodes'])
	for real,measure in zip([pc,wcd,degree,membership],['pc','wcd','degree','network']):
		for prediction in glob.glob('/home/mbmbertolero/data/gene_expression/results/*nn*%s*'%(measure)):
			players = prediction.split('/')[-1].split('.')[0].split('_')[1]
			nodes = prediction.split('/')[-1].split('.')[0].split('_')[2]
			try: nodes = int(nodes)
			except:
				nidx = 0
				while True:
					try:
						nidx = nidx + 1
						int(nodes[:nidx])
						continue
					except:break
				nodes = int(nodes[:nidx-1])

			prediction = np.load(prediction)
			if measure == 'network':
				r = len(real[real==prediction]) / float(len(real))
			else:
				r = pearsonr(prediction,real)[0]
			result_df = result_df.append(pd.DataFrame(np.array([[r],[measure],[players],[nodes]]).transpose(),columns=['prediction_acc','prediction_type','layers','nodes']))
	
	result_df.prediction_acc = result_df.prediction_acc.astype(float)
	df.nodes = df.nodes.astype(int)
	df.layers = df.layers.astype(int)
	for tp in np.unique(result_df.prediction_type.values):
		print tp,result_df[result_df.prediction_type==tp].layers.values[np.argmax(result_df.prediction_acc.values[result_df.prediction_type==tp])],result_df[result_df.prediction_type==tp].prediction_acc.values[np.argmax(result_df.prediction_acc.values[result_df.prediction_type==tp])]

	sns.set_style('white')
	sns.set(font='Helvetica')
	fig, subplots = sns.plt.subplots(2,2,figsize=(7.16535,7.16535))
	for idx,tp in enumerate(np.unique(result_df.prediction_type.values)):
		tdf = result_df[result_df.prediction_type==tp]
		tdf.prediction_acc[tdf.prediction_acc<0.0] = 0.0
		tdf.nodes = tdf.nodes.astype(int)
		tdf.layers = tdf.layers.astype(int)
		print tp,tdf.layers.values[np.argmax(tdf.prediction_acc.values)],tdf.nodes.values[np.argmax(tdf.prediction_acc.values)],np.max(tdf.prediction_acc.values)
		f = sns.stripplot(tdf.nodes.values,tdf.layers.values,hue=tdf.prediction_acc.values,palette="coolwarm",size=16,ax=subplots.flatten()[idx])
		f.set_ylabel('layers')
		f.set_xlabel('nodes')
		f.set_title(tp)
		f.legend_.remove()
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/nn_prediction_sweet.pdf')
	# sns.plt.show()
	sns.plt.close()


	sns.set(font='Helvetica')
	for measure,tp in zip([pc,wcd,degree],['pc','wcd','degree']):
		tdf = result_df[result_df.prediction_type==tp]
		tdf.nodes = tdf.nodes.astype(int)
		tdf.layers = tdf.layers.astype(int)
		idx = np.argmax(tdf.prediction_acc.values)
		nodes = tdf.nodes.values[idx]
		layers = tdf.layers.values[idx]
		prediction = np.load('/home/mbmbertolero/data/gene_expression/results/nn_%s_%s%s.npy'%(layers,nodes,tp))
		g = sns.jointplot(measure,prediction, kind="reg")
		sns.plt.ylabel('gene prediction')
		sns.plt.xlabel('%s' %(tp))
		sns.plt.tight_layout()
		sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/nn_prediction_%s.pdf'%(tp))
		# sns.plt.show()
		sns.plt.close()
	
	sns.set(font='Helvetica')
	tp = 'network'
	measure = membership
	tdf = result_df[result_df.prediction_type==tp]
	tdf.nodes = tdf.nodes.astype(int)
	tdf.layers = tdf.layers.astype(int)
	idx = np.argmax(tdf.prediction_acc.values)
	nodes = tdf.nodes.values[idx]
	layers = tdf.layers.values[idx]
	prediction = np.load('/home/mbmbertolero/data/gene_expression/results/nn_%s_%s%s.npy'%(layers,nodes,tp))
	bar_df = pd.DataFrame(columns=['system','recall'])
	for system in np.unique(measure):
		recall = len(membership[(membership==system) & (prediction == system)]) / float(len(membership[membership==system]))
		bar_df = bar_df.append(pd.DataFrame(np.array([[system],[recall]]).transpose(),columns=['system','recall']))
	bar_df.recall = bar_df.recall.astype(float)
	sns.barplot(y='recall',x='system',data=bar_df,orient='v')
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/nn_prediction_network.pdf')
	sns.plt.close()

def sge(whatrun):
	"""
	SGE stuff
	"""
	# fit_brain_map()
	# neurosynth()
	# os.system('qsub -V -l mem_free=24G -j y -o /home/mbmbertolero/data/allen_gene_expression/sge/ -e /home/mbmbertolero/data/allen_gene_expression/sge/ -N co_act gene_expression_analysis.py')
	# qsub -pe threaded 18 -binding linear:18  -V -l mem_free=5G -j y -o /home/mbmbertolero/data/gene_expression/sge/ -e /home/mbmbertolero/data/gene_expression/sge/ -N fitbrain gene_expression_analysis.py 

	if whatrun == 'make_fc':
		for subject in os.listdir('/home/despoB/connectome-data/'):
			os.system('qsub -V -l mem_free=34G -j y -o /home/mbmbertolero/data/gene_expression/sge/ -e /home/mbmbertolero/data/gene_expression/sge/ -N vfc gene_expression_analysis.py make_fc %s'%(subject))

	if whatrun == 'nodal':
		for node in range(400):
			os.system('qsub -pe threaded 10 -binding linear:10  -V -l mem_free=2.2G -j y -o /home/mbmbertolero/data/gene_expression/sge/ -e /home/mbmbertolero/data/gene_expression/sge/ -N n%s gene_expression_analysis.py %s %s' %(node,'nodal',node))

	if whatrun == 'layers':
		for nodes in [5,10,15,20,25,50,100,200,300]:
			for layers in range(1,16):
				if nodes * layers <= 1500:
					continue
				os.system('qsub -pe threaded 20 -binding linear:20  -V -l mem_free=20G -j y -o /home/mbmbertolero/data/gene_expression/sge/ -e /home/mbmbertolero/data/gene_expression/sge/ -N l%s_n%s gene_expression_analysis.py what_genes %s %s'%(layers,nodes,layers,nodes))

def run_all(matrix,topological,distance,network):
	fit_matrix(matrix,topological,distance,network)
	SA_find_genes(matrix,topological,distance,network,n_trys=50)

def plot_pc_by_fit(ctype,topological=False,distance=True):
	sns.set_style('whitegrid')
	sns.set(font='Helvetica')

	matrix = functional_connectivity(topological,distance,None)
	matrix = np.tril(matrix,-1)
	matrix = matrix + matrix.transpose()
	pc = []
	for cost in [.15,.16,.17,.18,.19,.2]:
		g = brain_graphs.matrix_to_igraph(matrix.copy(),cost=cost,mst=True)
		g = brain_graphs.brain_graph(g.community_infomap(edge_weights='weight'))
		pc.append(g.pc)
	pc = np.nanmean(pc,axis=0)

	reduce_dict = {'VisCent':0,'VisPeri':0,'SomMotA':1,'SomMotB':1,'DorsAttnA':2,'DorsAttnB':2,'SalVentAttnA':3,'SalVentAttnB':3,'Limbic':4,'ContA':5,'ContB':5,'ContC':5,'DefaultA':6,'DefaultB':6,'DefaultC':6,'TempPar':7}
	membership = np.zeros((400)).astype(str)
	yeo_df = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])['name'].values[1:]
	yeo_colors = pd.read_csv('/home/mbmbertolero/data/Alex400_MNI/Schaefer2016_400Parcels_17Networks_colors_23_05_16.txt',sep='\t',header=None,names=['name','R','G','B','0'])
	names = ['Visual','Motor','Dorsal Attention','Ventral Attention','Limbic','Control','Default','Temporal Parietal']



	for i,n in enumerate(yeo_df):
		membership[i] = n.split('_')[2]
	df = pd.DataFrame(columns=['fit','network','participation coefficient'])
	for node,name,pval in zip(range(400),membership,pc):
		try:
			a_matrix = functional_connectivity(topological,distance,node)
			gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(ctype,topological,distance,node))[-1]
			gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
			mask = np.sum([gene_exp_matrix,a_matrix],axis=0)
			mask[np.isnan(mask)==True] = False
			mask[mask!=False] = True
			mask = mask.astype(bool)
			r = nan_pearsonr(gene_exp_matrix[mask].flatten(),a_matrix[mask].flatten())[0]
			df= df.append(pd.DataFrame(np.array([[r],[name],[pval]]).transpose(),columns=['fit','network','participation coefficient']))
		except:
			df= df.append(pd.DataFrame(np.array([[np.nan],[name],[pval]]).transpose(),columns=['fit','network','participation coefficient']))
	df.fit = df.fit.astype(float)
	df['participation coefficient'] = df['participation coefficient'].astype(float)
	order = np.unique(membership)[np.argsort(df.groupby('network')['fit'].apply(np.nanmean))]
	
	colors = np.array([yeo_colors['R'],yeo_colors['G'],yeo_colors['B']]).transpose()[1:,][np.isnan(df.fit.values)==False] /256.
	
	def get_axis_limits(ax, scale=.9):
		# return -(ax.get_xlim()[1]-ax.get_xlim()[0])*.1, ax.get_ylim()[1]*scale
		return ax.get_xlim()[0], ax.get_ylim()[1] + (ax.get_ylim()[1]*.1)

	fig,subplots = sns.plt.subplots(3,figsize=(7.204724,14))
	yeo = sns.plt.imread('/home/mbmbertolero/data/gene_expression/yeo400.png')
	sns.set_style("white")
	s = subplots[0]
	s.imshow(yeo,origin='upper')
	s.set_xticklabels([])
	s.set_yticklabels([])

	sns.regplot(data=df,x='fit',y='participation coefficient',scatter_kws={'facecolors':colors},ax=subplots[1])

	network_colors = []
	for network in order:
		network_colors.append(np.mean(colors[membership[np.isnan(df.fit)==False]==network,:],axis=0))
	
	sns.barplot(data=df,x='network',y='fit',palette=network_colors,order=order,ax=subplots[2])
	subplots[2].set_ylim((np.min(df.fit),np.max(df.fit)))

	for label in subplots[2].get_xmajorticklabels():
		label.set_rotation(90)
	sns.despine()
	s.spines['bottom'].set_visible(False)
	s.spines['left'].set_visible(False)
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/figure1.pdf')
	sns.plt.show()

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

if len(sys.argv) > 1:
	if run_type == 'make_fc': subject_functional_connectivity(sys.argv[2])
	if run_type == 'nodal': SA_find_genes(matrix='fc',network=int(sys.argv[2]),cores=10)
	if run_type == 'what_genes':
		n_layers = int(sys.argv[2])
		n_nodes = int(sys.argv[3])
		layers = []
		for n in range(n_layers):
			layers.append(int(n_nodes))
		layers = tuple(layers)
		# layers_name = '_'
		# for l in layers:
		# 	layers_name = layers_name + '%s_' %(l)
		layers_name = '_%s_%s'%(n_layers,n_nodes)
		what_genes(cores=multiprocessing.cpu_count()-2)



# build gene network where edges are the similarity of expression across the brain; what role do gene play in this network that maximize fit to FC?

# run_all('fc',False,True,None)
# run_all('fc',False,True,'Control')
# run_all('fc',False,True,'Visual')
# run_all('fc',False,True,'Motor')
# run_all('fc',False,True,'Dorsal Attention')
# run_all('fc',False,True,'Ventral Attention')
# run_all('fc',False,True,'Temporal Parietal')
# run_all('fc',False,True,'Limbic')