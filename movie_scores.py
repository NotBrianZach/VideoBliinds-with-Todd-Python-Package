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

a_movie = 11.256803759
b_movie = 146929.116058

def f2(x, a, b):
    return a * np.log(1 + b*x)

def sorted_nicely(l): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

stems = np.array([
'bf_r1',
'fc_r1',
'sd_r1',
'rb_r1',
'ss_r1',
'po_r1',
'la_r1',
'dv_r1',
'tk_r1',
'hc_r1',
'bf_w1',
'fc_w1',
'sd_w1',
'rb_w1',
'ss_w1',
'po_w1',
'la_w1',
'dv_w1',
'tk_w1',
'hc_w1',
'bf_fr4',
'fc_fr4',
'sd_fr4',
'rb_fr4',
'ss_fr4',
'po_fr4',
'la_fr4',
'dv_fr4',
'tk_fr4',
'hc_fr4',
])

#read in the scores
fpath='/home/kingston/repositories/video_proc/example_code/MOVIE/opt/results/outputs/'
opath='/home/kingston/repositories/video_proc/quality/movie/'

for stem in stems:
	tmovie_files = sorted_nicely(glob.glob(fpath + stem + "_tmovie.frame*.txt"))
	smovie_files = sorted_nicely(glob.glob(fpath + stem + "_smovie.frame*.txt"))
	for i in xrange(len(tmovie_files)):
		bname = stem + "_score_" + str(i+17) + ".txt"
		oname = opath + bname

		fi = open(tmovie_files[i], 'r')
		tmov = np.float(fi.readline())
		fi.close()
		fi = open(smovie_files[i], 'r')
		smov = np.float(fi.readline())
		fi.close()

		movie = smov * np.sqrt(tmov)

		#prediction
		y_pred = f2(movie, a_movie, b_movie)

		fi = open(oname, 'w')
		fi.write(str(y_pred))
		fi.close()
