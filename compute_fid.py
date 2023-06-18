#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import statistics
import random

# Paths
dataset="CIFAR10"
model=["EviD-GAN"] 
stats_path = 'ps\\fid_stats_cifar10_train.npz' #"ps\\fid_stats_stl10_unlabel32.npz"   #fid_stats_cifar10_train.npz  cifar100_stats.npz# 
result=[]
result2=[]
record_num =0
record_train="EVID\\New_Result_{}_{}_{}.txt".format(dataset,model[0],record_num)
while os.path.exists(record_train):
    record_num += 1
    record_train ="EVID\\New_Result_{}_{}_{}.txt".format(dataset,model[0],record_num) 
    
fp= open(record_train,"w+")    
try:
    for j in model:
        for i in range(49): 
            image_path = "D:\\g_sample\\{}\\{}\\G_epoch_{}\\1\\".format(j,dataset,(i*2000)+10000) # set path to some generated images 
            inception_path = fid.check_or_download_inception(None) # download inception network

            # loads all images into memory (this might require a lot of RAM!)
            image_list = glob.glob(os.path.join(image_path, '*.png'))[:12500]
            images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

            # load precalculated training set statistics
            f = np.load(stats_path)
            mu_real, sigma_real = f['mu'][:], f['sigma'][:]
            f.close()

            fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            result.append(fid_value)
            a="{} - {}*** FID: {}".format(j,(i*2000)+10000,fid_value)
            result2.append(a)
            print("{} - {}*** FID: {}".format(j,(i*2000)+10000,fid_value))
        fp.write("\r\n{} {} {} {} {} *** {}".format(j,min(result),max(result),statistics.mean(result),statistics.stdev(result),str(result)))
        fp.write("\r\n {}".format(str(result2)))
except:  
    print("Error occurs")
    fp.write("\r\n{} {} {} {} {} *** {}".format(j,min(result),max(result),statistics.mean(result),statistics.stdev(result),str(result)))
    fp.write("\r\n {}".format(str(result2)))
finally:
    fp.close()
print("END")
        