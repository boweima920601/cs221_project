import numpy as np
import scipy
import sklearn
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import collections


files = {'english' : [], 'hindi' : [], 'mandarin' : [], 'russian' : []}
nation_dict = {'english' : 0, 'hindi' : 1, 'mandarin' : 2, 'russian' : 3}
nations = nation_dict.keys()

path = './wave'

for i in os.listdir(path):
	parser_file = os.path.join(path, i)
	for nation in nations:
		if os.path.isfile(parser_file) and nation in i:
			files[nation].append(parser_file)

X = np.empty((0, 13), dtype = np.float32)
Y = np.empty((0, 1), dtype = np.int32)
for nation in nations:
	for wav_file in files[nation]:
		(rate,sig) = wav.read(wav_file)	
		mfcc_features = mfcc(sig, rate)
		X = np.concatenate((X, mfcc_features), axis = 0)
		y = np.ones((mfcc_features.shape[0], 1)) * nation_dict[nation]
		print(y.shape)
		Y = np.concatenate((Y, y), axis = 0)	

print(X.shape)
print(Y.shape)



