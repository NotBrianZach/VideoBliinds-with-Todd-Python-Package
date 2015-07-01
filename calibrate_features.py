import pylab as py
import numpy as np
import glob
import os
import re
import sklearn
import scipy
import scipy as sp
import scipy.io
import scipy.optimize as opt
from sklearn.externals import joblib

from sklearn import svm, datasets, ensemble

a = 50.58
b = 1
c = 0.08451
d = 0

def f(x, a, b, c, d):
    if np.abs(d) > 30:
	    return 100 * a/(1. + np.exp(-c * (x-d))) + b
    return a/(1. + np.exp(-c * (x-d))) + b

def f2(x, a, b):
    return a * np.log(1 + b*x)

def sorted_nicely(l): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

def load_movie():
	lvl = np.arange(2, 17)
	fps = np.array(['25',
	'50',
	'25',
	'50',
	'25',
	'25',
	'25',
	'50',
	'25',
	'25',
	])
	nms = np.array(['bs',
	'mc',
	'pa',
	'pr',
	'rb',
	'rh',
	'sf',
	'sh',
	'st',
	'tr'])
	fpath='/home/kingston/repositories/video_proc/example_code/MOVIE/opt/calibration/outputs/'
	score_dict = {}
	i = 0
	for nm in nms:
		j = 0
		for ds in lvl: 
			#bs10_25fps.yuv_movie.txt 
			fname = fpath + nm + str(ds) + '_' + fps[i] + 'fps.yuv_movie.txt'
			fi = open(fname, 'r')
			line = float(fi.readline())
			fi.close()
			seq = nm + str(ds)	
			score_dict[seq] = line 
			j+=1
		i+=1
	return score_dict

def load_vblind():
	lvl = np.arange(1, 17)
	fps = np.array(['25',
	'50',
	'25',
	'50',
	'25',
	'25',
	'25',
	'50',
	'25',
	'25',
	])
	nms = np.array(['bs',
	'mc',
	'pa',
	'pr',
	'rb',
	'rh',
	'sf',
	'sh',
	'st',
	'tr'])
	fpath='/home/kingston/code/video_bliinds/output/'
	score_dict = {}
	i = 0
	for nm in nms:
		j = 0
		for ds in lvl: 
			#bs10_25fps.yuv_movie.txt 
			fname = fpath + nm + str(ds) + '_' + fps[i] + '_feats.txt'
			fi = open(fname, 'r')
			feats = fi.readline()
			fi.close()
			feats = np.array(feats.replace('\n', '')[:-1].split(' '))
			feats = feats.astype(np.float)
			seq = nm + str(ds)
	#		print feats
			score_dict[seq] = feats
			j+=1
		i+=1
	return score_dict

def load_mse():
	lvl = np.arange(2, 17)
	nms = np.array(['bs',
	'mc',
	'pa',
	'pr',
	'rb',
	'rh',
	'sf',
	'sh',
	'st',
	'tr'])
	score_dict = {}
	fpath = '/home/kingston/code/movie_MSE_PSNR/MSE.mat'
	mat = scipy.io.loadmat(fpath)
	dat = mat['MSE_g']
	i = 0
	for nm in nms:
		j = 1
		for ds in lvl: 
			seq = nm + str(ds)	
			score_dict[seq] = dat[i, j]
			j+=1
		i+=1
	return score_dict

def load_ssim():
	lvl = np.arange(2, 17)
	nms = np.array(['bs',
	'mc',
	'pa',
	'pr',
	'rb',
	'rh',
	'sf',
	'sh',
	'st',
	'tr'])
	score_dict = {}
	fpath = '/home/kingston/code/movie_ssim/mssim.mat'
	mat = scipy.io.loadmat(fpath)
	dat = mat['mssim_g']
	i = 0
	for nm in nms:
		j = 1
		for ds in lvl: 
			seq = nm + str(ds)	
			score_dict[seq] = dat[i, j]
			j+=1
		i+=1
	return score_dict

def load_strred():
	lvl = np.arange(2, 17)

	nms = np.array(['pa', 'rb', 'rh', 'tr', 'st', 'sf', 'bs', 'sh', 'mc', 'pr'])
	score_dict = {}
	fpath = '/home/kingston/code/newstrred/strred.mat'
	mat = scipy.io.loadmat(fpath)
	dat = mat['strred']
	i = 0
	for nm in nms:
		j = 1
		for ds in lvl: 
			seq = nm + str(ds)	
			score_dict[seq] = dat[i, j]
			j+=1
		i+=1
	return score_dict

distortion_lvl = np.arange(2, 17)
names = np.array(['pa',
'rb',
'rh',
'tr',
'st',
'sf',
'bs',
'sh',
'mc',
'pr'])

#build the DMOS lookup table
DMOS_lookup={}
seqs = []
for nm in names:
	for ds in distortion_lvl: 
		seqs.append(nm + str(ds))	

for nm in names:
	seqs.append(nm + str(1))	
seqs = np.array(seqs)

fi = open('/home/kingston/databases/live_vqdb/live_video_quality_data.txt','r')
i = 0
for item in fi:
	score = item.split('\t')[0]
	DMOS_lookup[seqs[i]] = np.float(score)
	i += 1

for nm in names:
	DMOS_lookup[seqs[i]] = 0
	i += 1

#load the strred output
ssim = load_ssim()
mse = load_mse()
movie = load_movie()
strred = load_strred()
videobliinds = load_vblind()

#now organize scores with ground truth
gt_scores = []
strred_scores = []
ssim_scores = []
mse_scores = []
movie_scores = []
vblind_feats = []
for item in seqs:
#	strred_scores.append(strred[item])
#	ssim_scores.append(ssim[item])
#	movie_scores.append(movie[item])
#	mse_scores.append(mse[item])
	vblind_feats.append(videobliinds[item])
	gt_scores.append(DMOS_lookup[item])
#strred = np.array(strred_scores)
#movie = np.array(movie_scores)
#ssim = np.array(ssim_scores)
#mse = np.array(mse_scores)
vblind = np.array(vblind_feats)
gt_scores = np.array(gt_scores)
#print np.shape(vblind)
#exit(0)

print "mse"
#rho, pval = scipy.stats.pearsonr(mse, gt_scores)
#print rho, pval
#srho, spval = scipy.stats.spearmanr(mse, gt_scores)
#print srho, spval

print "ssim"
#ssim = 1-ssim
#rho, pval = scipy.stats.pearsonr(ssim, gt_scores)
#print rho, pval
#srho, spval = scipy.stats.spearmanr(ssim, gt_scores)
#print srho, spval

print "strred"
#rho, pval = scipy.stats.pearsonr(strred, gt_scores)
#print rho, pval
#srho, spval = scipy.stats.spearmanr(strred, gt_scores)
#print srho, spval

print "movie"
#rho, pval = scipy.stats.pearsonr(movie, gt_scores)
#print rho, pval
#srho, spval = scipy.stats.spearmanr(movie, gt_scores)
#print srho, spval


#fit things
#(a_st, b_st), _ = opt.curve_fit(f2, strred, gt_scores, (a, b))

#(a_mov, b_mov), _ = opt.curve_fit(f2, movie, gt_scores, (a, b))

#(a_ssim, b_ssim), _ = opt.curve_fit(f2, ssim, gt_scores, (a, b))

#print a_mov, b_mov

#print np.shape(gt_scores)
#print np.shape(vblind)
#exit(0)
#fit svm for vblinds

#set features to normalized

#Cs = np.array([2**1, 2**2, 2**5, 2**8, 2**9, 2**10, 2**12, 2**14, 2**16])
#gammas = np.array([1e-6, 1e-05, 1e-04, 1e-03, 1e-02])
#	for gamma_t in gammas:
#C_t = 16384
#gamma_t = 1e-05
#mean = np.mean(vblind, axis=0)
#std = np.std(vblind, axis=0)
#y_mean = np.mean(gt_scores)
#y_std = np.std(gt_scores)
#y_t = (gt_scores.copy() - y_mean)/y_std
#feats = (vblind.copy() - mean)/std

#clf = svm.SVR(C=C_t, gamma=gamma_t, kernel="rbf")#gamma=1e-04
#clf.fit(feats, y_t)
#gt_scores /= 100.0
#vblind_val = clf.predict(feats)
rang = np.array([1024])#, 1100, 2)
#rang = np.arange(0.1, 2, 0.1)

import rpy2
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

robj.packages.importr("kernlab")
#kernlab = importr("kernlab")
#robj.r.load("MPDR.training_size2000.RData")
#model = robj.globalenv["model"]
#outcome = robj.r.predict(model)

#svm_model = ksvm(x=features_train, y=mos_train, type="nu-svr", kernel="vanilladot",kpar="automatic", nu=0.45, C=1.75, cross=2)
#features_test = as.matrix(read.table("features_test.txt"))
#predicted_mos=predict(svm_model,features_test)
#svm_model = robj.r.ksvm(x=vblind, y=gt_scores, type="nu-svr", kernel="vanilladot",kpar="automatic", nu=0.45, C=1.75, cross=2)


#exit(0)

if 0:
	clf = svm.NuSVR(nu=0.35, C=1024, kernel="rbf", gamma=4.45e-04)
	y_train = gt_scores#[idx].copy()
	#y_test = gt_scores[idx_t].copy()
	train = vblind
	y_mu = np.mean(y_train)
	y_sd = np.std(y_train)
	mu = np.mean(train, axis=0)
	sd = np.std(train, axis=0)
	y_train -= y_mu
	y_train /= y_sd
	train -=mu
	train /=sd
	#test = vblind[idx_t, :].copy()

	clf.fit(train, y_train)

	#save trained model
	joblib.dump(clf, 'video_bliinds_model.pkl', compress=9)
	joblib.dump(y_mu, 'video_bliinds_model_label_mu.pkl', compress=9)
	joblib.dump(y_sd, 'video_bliinds_model_label_sd.pkl', compress=9)
	joblib.dump(mu, 'video_bliinds_model_mu.pkl', compress=9)
	joblib.dump(sd, 'video_bliinds_model_sd.pkl', compress=9)

	exit(0)

#calibration curve
for C_t in rang:
	coefs = []
	gts = []
	scores =[]
	scoresp =[]
	for i in xrange(10):
		clf = svm.NuSVR(nu=0.35, C=C_t, kernel="rbf", gamma=4.45e-04)
		idx = np.hstack((np.arange(0, i*16), np.arange(i*16+16, len(gt_scores))))
		idx_t = np.arange(i*16, i*16+16)

		y_train = gt_scores[idx].copy()
		y_test = gt_scores[idx_t].copy()
		train = vblind[idx, :].copy()
		test = vblind[idx_t, :].copy()

		y_mu = np.mean(y_train)
		y_sd = np.std(y_train)
		y_test -= np.mean(y_train)
		y_test /= np.std(y_train)
		y_train -= np.mean(y_train)
		y_train /= np.std(y_train)

		test -= np.mean(train, axis=0)
		test /= np.std(train, axis=0)
		train -= np.mean(train, axis=0)
		train /= np.std(train, axis=0)

		clf.fit(train, y_train)

		val = clf.predict(test)
		val *= y_sd
		val += y_mu
		if coefs == []: 
			coefs = val
			gts = gt_scores[idx_t]
		else: 
			coefs = np.hstack((coefs, val))
			gts = np.hstack((gts, gt_scores[idx_t]))
		scores.append(scipy.stats.spearmanr(val, gt_scores[idx_t])[0])
		scoresp.append(scipy.stats.pearsonr(val, gt_scores[idx_t])[0])

#remove 0s (pristine videos)
coefs = coefs[gts!=0]
gts = gts[gts!=0]

#map coefs to gts using logistic
(a_vblind, b_vblind), _ = opt.curve_fit(f2, coefs, gts, (a, b))
print a_vblind, b_vblind

srho, spval = scipy.stats.spearmanr(coefs, gts)
rho, pval = scipy.stats.pearsonr(coefs, gts)
print C_t, srho, rho#, spval

print np.median(scores), np.median(scoresp)

range = np.arange(1, 100, 0.1)
#fit logistic curve to the strred and movie

y_p = f2(range, a_vblind, b_vblind)

py.figure()
py.plot(range, y_p)
py.scatter(coefs, gts)
#py.xlim([0, np.max(ssim)])
py.ylim([0, 100])
py.xlabel('Predicted DMOS')
py.ylabel('True DMOS')
py.tight_layout()
py.savefig('logistic_fit_vbliind.png')
py.show()


exit(0)
#if 1:
for C_t in rang:
	coefs = []
	gts = []
	scores =[]
	scoresp =[]
	for i in xrange(9):
		for j in xrange(i+1, 10):
			#clf = svm.SVR(C=C_t, kernel="linear")#gamma=1e-04
			#clf = svm.NuSVR(nu=0.08, C=C_t, kernel="linear")#linear")#gamma=1e-04
			clf = svm.NuSVR(nu=0.35, C=C_t, kernel="rbf", gamma=4.45e-04)
			idx = np.hstack((np.arange(0, i*16), np.arange(i*16+16, j*16), np.arange(j*16+16, len(gt_scores))))
			idx_t = np.hstack((np.arange(i*16, i*16+16), np.arange(j*16, j*16+16)))

			y_train = gt_scores[idx].copy()
			y_test = gt_scores[idx_t].copy()
			train = vblind[idx, :].copy()
			test = vblind[idx_t, :].copy()
			#print svm_model

			y_test -= np.mean(y_train)
			y_test /= np.std(y_train)
			y_train -= np.mean(y_train)
			y_train /= np.std(y_train)

			mi = np.min(y_train, axis=0)
			ma = np.max(y_train, axis=0) - mi
			#y_test -= mi
			#y_test /= ma
			#y_train -= mi
			#y_train /= ma
	#			print np.shape(vblind)
	#			exit(0)


			#test -= mi#np.mean(train, axis=0)
			#test /= ma#np.std(train, axis=0)
			#test -= 0.5
			#test *= 2
			#train -= mi#np.mean(train, axis=0)
			#train /= ma#np.std(train, axis=0)
			#train -= 0.5
			#train *= 2

			#mi = np.min(train, axis=0)
			#ma = np.max(train, axis=0) - mi
			#test -= mi
			#test /= ma
			#train -= mi
			#train /= ma
			test -= np.mean(train, axis=0)
			test /= np.std(train, axis=0)
			train -= np.mean(train, axis=0)
			train /= np.std(train, axis=0)

	#		svm_model = robj.r.ksvm(x=train, y=y_train, type="nu-svr", kernel="vanilladot",kpar="automatic", nu=0.45, C=C_t, cross=2)
	#		val =robj.r.predict(svm_model,test)
	#		val = np.array(val).ravel()

			clf.fit(train, y_train)

			val = clf.predict(test)
			if coefs == []: 
				coefs = val
				gts = gt_scores[idx_t]
			else: 
				coefs = np.hstack((coefs, val))
				gts = np.hstack((gts, gt_scores[idx_t]))
			scores.append(scipy.stats.spearmanr(val, gt_scores[idx_t])[0])
			scoresp.append(scipy.stats.pearsonr(val, gt_scores[idx_t])[0])

	srho, spval = scipy.stats.spearmanr(coefs, gts)
	rho, pval = scipy.stats.pearsonr(coefs, gts)
	print C_t, srho, rho#, spval

	print np.median(scores), np.median(scoresp)

#check correlation
exit(0)
print "video bliinds"
rho, pval = scipy.stats.pearsonr(coefs, gt_scores)
print rho, pval
exit(0)

#produce logistic function
(a_vbl, b_vbl, c_vbl, d_vbl), _ = opt.curve_fit(f, coefs, gts, (a, b, c, d))

#exit(0)
range = np.arange(-10, 10, 0.00001)
#fit logistic curve to the strred and movie

y_p = f(range, a_vbl, b_vbl, c_vbl, d_vbl)

py.figure()
py.scatter(vblind_val, gt_scores)
py.plot(range, y_p)
#py.xlim([0, np.max(ssim)])
py.ylim([0, 100])
py.xlabel('Predicted DMOS')
py.ylabel('True DMOS')
py.tight_layout()
py.savefig('logistic_fit.png')
py.show()

#use SVM with videoblinds
