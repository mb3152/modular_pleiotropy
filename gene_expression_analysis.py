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
from sklearn.linear_model import Ridge, RidgeClassifier

import glob
from itertools import combinations
import operator
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import patches
plt.rcParams['pdf.fonttype'] = 42
path = '/home/mbmbertolero/data/helvetica/Helvetica.ttf'
mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = 'Helvetica'
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
parser.add_argument('-m',action='store',dest='matrix',default='sc')
parser.add_argument('-layers',action='store',dest='n_layers',type=int,default=1)
parser.add_argument('-name',action='store',dest='name',type=str,default='participation coefficient')
parser.add_argument('-nodes',action='store',dest='n_nodes',type=int,default=5)
parser.add_argument('-n',action='store',dest='network')
parser.add_argument('-task',action='store',dest='task')
parser.add_argument('-s',action='store',dest='subject')
parser.add_argument('-topological',action='store',dest='topological',type=bool,default=False)
parser.add_argument('-distance',action='store',dest='distance',type=bool,default=True)
parser.add_argument('-prediction',action='store',dest='prediction',type=str,default='genetic')
r = parser.parse_args()
locals().update(r.__dict__)
try:network = int(network)
except:pass
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
			if random in m: continue
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
		np.save('/home/mbmbertolero/data/gene_expression/data/matrices/sc_matrix_new.npy',matrix)
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

def avg_graph_metrics(matrix):
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
		# g = brain_graphs.brain_graph(g.community_infomap(edge_weights='weight'))
		g = VertexClustering(g,membership=membership_ints)
		g = brain_graphs.brain_graph(g)
		pc.append(g.pc)
		wcd.append(g.wmd)
		degree.append(g.community.graph.strength(weights='weight'))
		between.append(g.community.graph.betweenness())
	pc = np.nanmean(pc,axis=0)
	wcd = np.nanmean(wcd,axis=0)
	degree = np.nanmean(degree,axis=0)
	between = np.nanmean(between,axis=0)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix),pc)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix),degree)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix),wcd)
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix),between)

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

def fit_matrix_multi(ignore_idx):
	print ignore_idx
	sys.stdout.flush()
	global co_a_matrix
	global gene_exp
	temp_m = np.corrcoef(gene_exp[:,np.arange(gene_exp.shape[1])!=ignore_idx])
	np.fill_diagonal(temp_m,np.nan)
	return nan_pearsonr(co_a_matrix.flatten(),temp_m.flatten())[0]

def fit_matrix(matrix,topological=True,distance=True,network=None):
	try: result = np.load('/home/mbmbertolero/data/gene_expression/results/fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,network))
	except:
		global gene_exp
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

def SA_find_genes(matrix,topological=False,distance=True,network=None,n_genes=100,start_temp=.5,end_temp=.01,temp_step=.001,n_trys=100,cores=15):
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

def plot_pc_by_fit(matrix,topological=False,distance=True):

	sns.set(context="paper",font='Helvetica',style='white')
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

	if matrix == 'fc': m = functional_connectivity(topological,distance,None)
	if matrix == 'sc': m = structural_connectivity(topological,distance,None)
	m = m + m.transpose()
	m = np.tril(m,-1)
	m = m + m.transpose()

	try:
		pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
		strength = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))
		wcs = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))
		between = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))
	except:
		pc = []
		wcs = []
		strength = []
		between = []
		for cost in np.linspace(0.01,0.10):
			g = brain_graphs.matrix_to_igraph(m.copy(),cost=cost,mst=True)
			# g = brain_graphs.brain_graph(g.community_infomap(edge_weights='weight'))
			g = VertexClustering(g,membership=membership_ints)
			g = brain_graphs.brain_graph(g)
			pc.append(g.pc)
			wcs.append(g.wmd)
			strength.append(g.community.graph.strength(weights='weight'))
			between.append(g.community.graph.betweenness())
		pc = np.nanmean(pc,axis=0)
		wcd = np.nanmean(wcd,axis=0)
		strength = np.nanmean(strength,axis=0)
		between = np.nanmean(between,axis=0)
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix),pc)
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix),strength)
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix),wcd)
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix),between)

	df = pd.DataFrame(columns=['fit','network','participation coefficient','within community strength','strength','betweenness'])
	for node,name,pc_val,wcs_val,strength_val,b_val in zip(range(400),membership,pc,wcs,strength,between):
		if node in ignore_nodes: continue
		try:gene_exp_matrix = np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,node))[-1]
		except:continue
		gene_exp_matrix = np.corrcoef(gene_exp[:,gene_exp_matrix])
		r = nan_pearsonr(m[node,:].flatten(),gene_exp_matrix[node,:].flatten())[0]
		df= df.append(pd.DataFrame(np.array([[r],[name],[pc_val],[wcs_val],[strength_val],[b_val]]).transpose(),columns=['fit','network','participation coefficient','within community strength','strength','betweenness']))
	df.fit = df.fit.astype(float)
	df['participation coefficient'] = df['participation coefficient'].astype(float)
	df['strength'] = df['strength'].astype(float)
	df['within community strength'] = df['within community strength'].astype(float)
	df['betweenness'] = df['betweenness'].astype(float)
	df.to_csv('/home/mbmbertolero/data/gene_expression/results/%s_fits_df.csv'%(matrix))
	order = np.unique(membership)[np.argsort(df.groupby('network')['fit'].apply(np.nanmean))]
	
	network_colors = []
	for network in order:
		network_colors.append(np.mean(colors[membership==network,:],axis=0))

	def get_axis_limits(ax, scale=.9):
		# return -(ax.get_xlim()[1]-ax.get_xlim()[0])*.1, ax.get_ylim()[1]*scale
		return ax.get_xlim()[0], ax.get_ylim()[1] + (ax.get_ylim()[1]*.1)

	fig,subplots = sns.plt.subplots(3,figsize=(7.204724,14))
	yeo = sns.plt.imread('/home/mbmbertolero/gene_expression/yeo400.png')
	s = subplots[0]
	s.imshow(yeo,origin='upper')
	s.set_xticklabels([])
	s.set_yticklabels([])
	sns.plt.draw()
	sns.regplot(data=df,x='fit',y='participation coefficient',scatter_kws={'facecolors':colors},ax=subplots[1])
	r = str(pearsonr(df['participation coefficient'],df.fit)[0])[:6]
	p = str(pearsonr(df['participation coefficient'],df.fit)[1])[:6]
	sns.barplot(data=df,x='network',y='fit',palette=network_colors,order=order,ax=subplots[2])
	subplots[2].set_ylim(.2,.7)
	for label in subplots[2].get_xmajorticklabels():
		label.set_rotation(90)
	sns.despine()
	s.spines['bottom'].set_visible(False)
	s.spines['left'].set_visible(False)
	sns.plt.tight_layout()
	subplots[1].text(np.max(df.fit)*.9,np.max(df['participation coefficient'])*.9,'r = %s\np= %s'%(r,p))
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/network_fits_and_pccorr_%s.pdf'%(matrix))
	sns.plt.show()

def compare_genetic_fits():

	sns.set(context="paper",font='Helvetica',style='white')

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
	sc_fits = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/sc_fits_df.csv')
	sc_fits['network'] = 'structural connectivity'
	fc_fits = pd.read_csv('/home/mbmbertolero/data/gene_expression/results/fc_fits_df.csv')
	fc_fits['network'] = 'functional connectivity'
	df = fc_fits.append(sc_fits)
	#plot all the PC fits from sc and fc
	sns.regplot(data=df,x='fit',y='participation coefficient',scatter_kws={'facecolors':colors})
	r = str(np.around(pearsonr(df['participation coefficient'],df.fit)[0],4))
	p = str(np.around(pearsonr(df['participation coefficient'],df.fit)[1],4))
	if p == '0.0': p = '<1-e5'
	sns.plt.text(np.max(df.fit)*.9,np.max(df['participation coefficient'])*.9,'r = %s\np= %s'%(r,p))
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/fc_and_sc_pc_fits.pdf')
	sns.plt.show()

	#plot all the fits
	x,y = sc_fits['fit'], fc_fits['fit']
	g = sns.regplot(data=df,x=x,y=y,scatter_kws={'facecolors':colors})
	g.set_ylabel('fc fit')
	g.set_xlabel('sc fit')
	r = pearsonr(x,y)[0]
	p = pearsonr(x,y)[1]
	if r > 0: xy=(.1, .87)
	else: xy=(.7, .87)
	g.annotate("r={:.3f}\np={:.3f}".format(r,p),xy=xy, xycoords=g.transAxes)
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/fc_and_sc_fits.pdf')
	sns.plt.show()

	fig, subplots = sns.plt.subplots(2,2,figsize=(7.16535,7.16535))
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
	sns.plt.savefig('/home/mbmbertolero/gene_expression/figures/fc_and_sc_metric_corr.pdf')
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

def role_prediction(matrix,prediction,distance=True,topological=False,cores=4):
	global features
	global measure
	global layers
	global layers_name
	global size

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

	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))
	degree = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))
	wcd = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))
	between = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))

	if prediction == 'genetic':
		df = pd.DataFrame(columns=['gene','node','participation coefficient','within community strength','strength','betweenness','network'])
		for node,name,pval,w,d,b,m in zip(range(400),membership,pc,wcd,degree,between,membership):
			if node in ignore_nodes: continue
			for gene in np.load('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_%s_%s_%s.npy'%(matrix,topological,distance,node))[-1]:
				df= df.append(pd.DataFrame(np.array([[gene],[node],[pval],[w],[d],[b],[m]]).transpose(),columns=['gene','node','participation coefficient','within community strength','strength','betweenness','network']))
		df.gene = df.gene.astype(float)
		df['participation coefficient'] = df['participation coefficient'].astype(float)
		df['strength'] = df['strength'].astype(float)
		df['within community strength'] = df['within community strength'].astype(float)
		df['betweenness'] = df['betweenness'].astype(float)
		df.node = df.node.values.astype(int)
		unique = np.unique(df.gene[np.isnan(df.gene.values)==False])
		features = np.zeros((400,len(unique))).astype(bool)
		for node in df.node.values:
			g_array = unique
			g_b_array = unique.copy().astype(bool)
			g_b_array[:] = False
			for g in df.gene[df.node==node].values:
				g_b_array[g_array==g] = True
			features[node,:] = g_b_array
		features = features.astype(int)
		mask = np.ones((400)).astype(bool)
		mask[ignore_nodes] = False
		membership = membership[mask]
		features = features[mask]

	if prediction == 'structural_connectivity':

		df = pd.DataFrame(columns=['node','participation coefficient','within community strength','strength','betweenness','network'])
		for node,name,pval,w,d,b,m in zip(range(400),membership,pc,wcd,degree,between,membership):
			df= df.append(pd.DataFrame(np.array([[node],[pval],[w],[d],[b],[m]]).transpose(),columns=['node','participation coefficient','within community strength','strength','betweenness','network']))
		df['participation coefficient'] = df['participation coefficient'].values.astype(float)
		features = []
		spc = np.load('/home/mbmbertolero/data/gene_expression/results/sc_pc.npy')
		sdegree = np.load('/home/mbmbertolero/data/gene_expression/results/sc_strength.npy')
		swcd = np.load('/home/mbmbertolero/data/gene_expression/results/sc_wcd.npy')
		sbetween = np.load('/home/mbmbertolero/data/gene_expression/results/sc_between.npy')
		# sedges = structural_connectivity(topological,distance,None)
		# np.fill_diagonal(sedges,np.nan)
		for node in range(400):
			f = []
			f.append(spc[node])
			# f.append(sdegree[node])
			# f.append(swcd[node])
			# f.append(sbetween[node])
			# for edge in list(sedges[node][np.isnan(sedges[node])==False]):
				# f.append(edge)
			features.append(np.array(f))
		features = np.array(features)
	
	size = features.shape[0]
	"""
	predict the nodal values of a node based on which genes maximize its fit to FC
	"""
	for name in ['participation coefficient','strength','within community strength','betweenness']:
		measure = np.array(df.groupby('node').mean()[name].values)
		pool = Pool(16)
		prediction_array = np.array(pool.map(regress,range(size)))[:,0]
		del pool
		print pearsonr(prediction_array,measure)
		np.save('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s.npy'%(prediction,matrix,name.replace(' ','_')),prediction_array)
	"""
	predict the network memebership of the node based on which genes maximize its fit to FC
	"""
	measure = membership
	pool = Pool(16)
	prediction_array = np.array(pool.map(cater,range(size)))[:,0]
	np.save('/home/mbmbertolero/data/gene_expression/results/%s_%s_network.npy'%(prediction,matrix),prediction_array)
	print 'RESULT: ' + str(len(prediction_array[prediction==membership]))

def plot_role_prediction(matrix = 'fc'):
	sns.set(context="notebook",font='Helvetica',style='white')
	
	distance = True
	topological = False
	prediction = 'genetic'
	reduce_membership = False

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

	mask = np.ones((400)).astype(bool)
	mask[ignore_nodes] = False

	pc = np.load('/home/mbmbertolero/data/gene_expression/results/%s_pc.npy'%(matrix))[mask]
	pc_p = np.load('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s.npy'%(prediction,matrix,'participation_coefficient'))
	strength = np.load('/home/mbmbertolero/data/gene_expression/results/%s_strength.npy'%(matrix))[mask]
	strength_p = np.load('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s.npy'%(prediction,matrix,'strength'))
	wcd = np.load('/home/mbmbertolero/data/gene_expression/results/%s_wcd.npy'%(matrix))[mask]
	wcd_p = np.load('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s.npy'%(prediction,matrix,'within_community_strength'))
	between = np.load('/home/mbmbertolero/data/gene_expression/results/%s_between.npy'%(matrix))[mask]
	between_p = np.load('/home/mbmbertolero/data/gene_expression/results/%s_%s_%s.npy'%(prediction,matrix,'betweenness'))


	df = pd.DataFrame(columns=['node','nodal metric','observed','prediction'])
	for i in range(len(pc)):
		df = df.append(pd.DataFrame(np.array([[i],['participation coefficient'],[pc[i]],[pc_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))
		df = df.append(pd.DataFrame(np.array([[i],['strength'],[strength[i]],[strength_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))
		df = df.append(pd.DataFrame(np.array([[i],['within community strength'],[wcd[i]],[wcd_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))
		df = df.append(pd.DataFrame(np.array([[i],['betweenness'],[between[i]],[between_p[i]]]).transpose(),columns=['node','nodal metric','observed','prediction']))

	df.observed = df.observed.values.astype(float)
	df.prediction = df.prediction.values.astype(float)

	g = sns.lmplot(x='observed',y='prediction',col='nodal metric',data=df,sharex=False,sharey=False,col_wrap=2,palette="muted",hue='nodal metric',size=(7.204724/2))
	for i,m in enumerate(g.hue_names):
		p = pearsonr(df['prediction'][df['nodal metric']==m].values,df['observed'][df['nodal metric']==m].values)
		r,p = p[0],p[1]
		if r > 0: xy=(.1, .87)
		else: xy=(.7, .87)
		g.axes[i].annotate("r={:.3f}\np={:.3f}".format(r,p),xy=xy, xycoords=g.axes[i].transAxes)
	sns.plt.tight_layout()
	sns.plt.savefig('/home/mbmbertolero/data/gene_expression/figures/predict_%s_with_%s.pdf'%(matrix,prediction))
	sns.plt.show()	

def task_performance(subjects,task):
	df = pd.read_csv('//home/mbmbertolero//S900_Release_Subjects_Demographics.csv')
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

def get_subjects(task,homedir='/home/mbmbertolero/hcp_performance/'):
	from functools import reduce
	subjects = []
	task_subjects = np.loadtxt('/home/mbmbertolero/hcp/scripts/tfMRI/subject_lists/subject_runnum_list/all_new_subject_list_sort_post_motion.txt').astype(int).astype(str)
	rest_subjects = np.loadtxt('/home/mbmbertolero/hcp/scripts/rfMRI/subject_lists/subject_runnum_list/all_new_subject_list_sort_post_motion.txt').astype(int).astype(str)
	bedpost_subjects = []
	for f in glob.glob('/home/mbmbertolero/data/gene_expression/sc_matrices/**/matrix.npy'):
		bedpost_subjects.append(f.split('/')[-2])

	all_subjects =  reduce(np.intersect1d,[task_subjects,rest_subjects,bedpost_subjects])
	for subject in all_subjects:
		task_files = np.loadtxt('/home/mbmbertolero/hcp/scripts/tfMRI/subject_lists/allsub_MSM_fMRI_list/%s_tfMRI_list_post_motion.txt'%(int(subject)),dtype=str)
		rest_files = np.loadtxt('/home/mbmbertolero/hcp/scripts/rfMRI/subject_lists/allsub_MSM_ICA_FIX_fMRI_list/%s_rfMRI_list_post_motion.txt'%(int(subject)),dtype=str)
		if rest_files.shape[0] != 4: continue
		if np.isnan(task_performance([int(subject)],task))[0]: continue
		subjects.append(int(subject))
	subjects.sort()
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
			files = glob.glob('/home/mbmbertolero/hcp/matrices/yeo/*%s*REST*.txt' %(subject))
			for f in files:
				if 'random' in f: continue
				f = np.loadtxt(f)
				np.fill_diagonal(f,0.0)
				f[np.isnan(f)] = 0.0
				f = np.arctanh(f)
				m.append(f)
			m = np.nanmean(m,axis=0)
			m = np.tril(m,-1) + np.tril(m,-1).transpose()
		else: 
			m = np.load('/home/mbmbertolero/data/gene_expression/sc_matrices/%s/matrix.npy'%(subject))
			m = m + m.transpose()
			m = np.tril(m,-1)
			m = m + m.transpose()
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

def super_edge_predict_new(t):
	fit_mask = np.ones((subject_pcs.shape[0])).astype(bool)
	fit_mask[t] = False
	if use_matrix == True:
		flat_matrices = np.zeros((subject_pcs.shape[0],len(np.tril_indices(264,-1)[0])))
		for s in range(subject_pcs.shape[0]):
			m = matrices[s]
			flat_matrices[s] = m[np.tril_indices(264,-1)]
		
		rest_perf_edge_corr = generate_correlation_map(task_perf[fit_mask].reshape(1,-1),flat_matrices[fit_mask].transpose())[0]
		
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

	# pvals = np.array([rest_pc,rest_wmd,rest_perf_edge_scores,subject_mods]).transpose()
	pvals = np.array([rest_pc,rest_wmd,subject_mods]).transpose()
	neurons = (3,3)
		
	train = np.ones(len(pvals)).astype(bool)
	train[t] = False
	model = MLPRegressor(solver='lbfgs',hidden_layer_sizes=neurons,alpha=1e-5,random_state=t)
	model.fit(pvals[train],task_perf[train])
	result = model.predict(pvals[t].reshape(1, -1))[0]
	return result

def performance_across_tasks(matrix):
	global subject_pcs
	global subject_wmds
	global subject_mods
	global task_perf
	global matrices
	global use_matrix
	use_matrix = True
	tasks=['WM','RELATIONAL','LANGUAGE','SOCIAL','EMOTION']
	loo_columns= ['Task','Predicted Performance','Performance']
	loo_df = pd.DataFrame(columns = loo_columns)
	for task in tasks:
		subjects = get_subjects(task)
		static_results = get_graphs(subjects,matrix)
		subject_pcs = static_results['subject_pcs'].copy()
		subject_wmds = static_results['subject_wmds']
		matrices = static_results['matrices']
		subject_mods = static_results['subject_mods']


		task_perf = task_performance(subjects,task)

		# if control_motion == True:
		# 	task_perf = sm.GLM(task_perf,sm.add_constant(subject_motion)).fit().resid_response
		# 	assert np.isclose(0.0,pearsonr(task_perf,subject_motion)[0]) == True

		"""
		prediction / cross validation
		"""
		pool = Pool(16)
		nodal_prediction = pool.map(super_edge_predict_new,range(len(task_perf)))
		del pool
		result = pearsonr(np.array(nodal_prediction).reshape(-1),task_perf)
		print 'Prediction of Performance: ', result
		# sys.stdout.flush()
		# loo_array = []
		# for i in range(len(nodal_prediction)):
		# 	loo_array.append([task,nodal_prediction[i],task_perf[i]])
		# loo_df = pd.concat([loo_df,pd.DataFrame(loo_array,columns=loo_columns)],axis=0)

def sge(run_type,matrix):
	"""
	SGE stuff
	"""
	if run_type == 'graph_metrics':
		for task in tasks:
			subjects = get_subjects(task)
			for s in subjects:
				os.system("qsub -N gm_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r graph_metrics -s %s -task %s "%(s,s,task))

	if run_type == 'find_genes':
		for node in range(400):
			if os.path.exists('/home/mbmbertolero/data/gene_expression/results/SA_fit_all_%s_False_True_%s.npy'%(matrix,node)): continue
			os.system('qsub -binding linear:4 -pe unihost 4 -N fg_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis.py -r find_genes -m %s -n %s'%(node,matrix,node))
	if run_type == 'sweep':
		ln = []
		for layers in range(1,16):
			for nodes in [5,10,15,20,25,50,100,200,300]:
				ln.append([nodes,layers])	
		ln = np.array(ln)
		for nodes,layers in ln[np.argsort(np.multiply(ln.transpose()[0],ln.transpose()[1]))]:
			print nodes,layers
			for matrix in ['fc','sc']:
				os.system("qsub -binding linear:4 -pe unihost 4 -N pdt_%s_%s -j y -b y -o /home/mbmbertolero/sge/ -e /home/mbmbertolero/sge/ /home/mbmbertolero/gene_expression/gene_expression_analysis.py -prediction genetic -r predict_role -m %s -layers %s -nodes %s "%(layers,nodes,matrix,layers,nodes))

def run_all(matrix,topological,distance,network):
	fit_matrix(matrix,topological,distance,network)
	SA_find_genes(matrix,topological,distance,network,n_trys=50)

ignore_nodes = np.where(np.nansum(gene_exp,axis=1)==0)[0]

if run_type == 'find_genes':  SA_find_genes(matrix=matrix,network=network,cores=4)
if run_type == 'predict_role': role_prediction(matrix,prediction,distance=True,topological=False,cores=4)
if run_type == 'graph_metrics': 
	graph_metrics(subject,task,'sc')
	graph_metrics(subject,task,'fc')






