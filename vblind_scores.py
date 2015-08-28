#import pylab as py
#import numpy as np
import glob
import os
import re
import sklearn
#import scipy
#import scipy as sp
#import scipy.io
#import scipy.optimize as opt
import sys
from sklearn.externals import joblib


#from sklearn import svm, datasets, ensemble

filePath = os.path.dirname(os.path.abspath(__file__))
a_vblind = 67.14762
b_vblind = 0.02275

def f2(x, a, b):
    return a * np.log(1 + b*x)

def sorted_nicely(l): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

#load model stuff
#model = joblib.load(filePath +"/" + 'video_bliinds_model.pkl')
#mu = joblib.load(filePath + "/" + 'video_bliinds_model_mu.pkl')
#sd = joblib.load(filePath + "/" + 'video_bliinds_model_sd.pkl')
#y_mu = joblib.load(filePath + "/" +'video_bliinds_model_label_mu.pkl')
#y_sd = joblib.load(filePath + "/" + 'video_bliinds_model_label_sd.pkl')
model = joblib.load('video_bliinds_model.pkl')
mu = joblib.load('video_bliinds_model_mu.pkl')
sd = joblib.load('video_bliinds_model_sd.pkl')
y_mu = joblib.load('video_bliinds_model_label_mu.pkl')
y_sd = joblib.load('video_bliinds_model_label_sd.pkl')

#read in the scores
#fpath='/home/kingston/code/video_bliinds/output2/'
#opath='/home/kingston/repositories/video_proc/quality/video_blinds/'
#fpath='/Users/brian/Downloads/VideoBLIINDS_Code_MicheleSaad/'
#opath='/Users/brian/'
fpath='~/vbliindTrainingOutput/'
fpath=os.path.expanduser(fpath)
opath='~/vbliindTrainingOutput/'
opath=os.path.expanduser(opath)
#print sys.argv[1]#should be dir path to file and also unique run number
#TODO split on spaces?sys.argv[1].split("")
#print sys.argv[1][13:]#should be dir path to file

print fpath
items = glob.glob(fpath + '*.txt')
print items
#features_file = open('~/vbliindTrainingOutput/features_test.txt','r')
#print features_file
items = sorted_nicely(items)
for t in items:
    bname = os.path.basename(t)
    #bname = bname.replace('feats', 'score')
    bname = bname.replace('features_test', 'vscore')
    oname = opath + bname
    
    fi = open(t, 'r')
    feats = fi.readline()
    fi.close()
    feats = np.array(feats.replace('\n', '')[:-1].split(' '))
    feats = feats.astype(np.float)
    print 'feats'
    print feats
    print 'mu'
    print mu
    
    #prediction
    feats -= mu
    feats /= sd
    y_pred = model.predict(feats)
    y_pred *= y_sd
    y_pred += y_mu
    y_pred = f2(y_pred, a_vblind, b_vblind)
    
    fi = open(oname, 'w')
    fi.write(str(y_pred[0]))
    fi.close()
