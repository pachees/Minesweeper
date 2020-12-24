# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 09:44:41 2018

[Deprecated] Trainer based in TF 1.2 library for testing: 
    Get in touch if you need a solver implementing reinforcement learning
    
(And yes of course there are much better ways to write the code)

@author: pachees
"""

import numpy as np
import grid_handler as gh
import tensorflow as tf
import datetime

class training:
    
    def gridr(rmajor, cmajor, nmastergrids, nmirrors):

        Gridsin, Gridsout = gh.main(rmajor, cmajor, nmastergrids, nmirrors)

        batchsize = nmastergrids * nmirrors 
                
        rmajor = rmajor
        cmajor = cmajor
        
        flatin = np.zeros((batchsize, rmajor*cmajor),  dtype=float)
        flatout = np.zeros((batchsize, rmajor*cmajor),  dtype=int)
        c = 0
        
        for i in range(nmastergrids):
            for j in range(nmirrors):
                flatin[c] = Gridsin[i][j].flatten()
                flatout[c] = Gridsout[i][j].flatten()
                c += 1
                
        return flatin, flatout
    
    def classifier(rmajor, cmajor, nmastergrids, nmirrors, hidlayersdef, outclasses, sets, epochs, batchsize, learningrate):
        tf.reset_default_graph()
        
        n_cells = rmajor*cmajor
        n_hidlayers = len(hidlayersdef)
        
        hidlayersdef.insert(0, n_cells)
        hidlayersdef.insert(n_hidlayers, outclasses)
        
        
        n_layers = n_hidlayers+2
        networks = [[0] * (n_layers-1)]*n_cells
        
        outputtriad = [0]*n_cells
        
        costs = [0]*n_cells
        optimizers = [0]*n_cells
        
        x = tf.placeholder(dtype = tf.float64,  name = 'Xwala',  shape = [None, n_cells])
        y = tf.placeholder(dtype = tf.int64,  name = 'Yout',  shape = [None,  n_cells])
        
        
        for i in range(n_cells):
            
            networks[i][0] = x
            for j in range(1, n_hidlayers+1):
                networks[i][j] = tf.contrib.layers.fully_connected(networks[i][j-1],  hidlayersdef[j],  activation_fn=tf.nn.relu)
            outputtriad[i] = tf.contrib.layers.fully_connected(networks[i][n_hidlayers],  outclasses,  activation_fn=tf.nn.sigmoid)
            
            
            costs[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, 
                                                               labels = y[:, i], 
                                                               logits = outputtriad[i])

            optimizers[i] = tf.train.AdamOptimizer(learningrate).minimize(costs[i])
        
        
        init = tf.global_variables_initializer()
        
        xtesth, ytesth = training.gridr(rmajor, cmajor, nmastergrids, nmirrors)
        
        saver = tf.train.Saver()
        now = datetime.datetime.now()
        now = str(now)
        
        with tf.Session() as sess:
            sess.run(init)
            
            for k in range(sets):
                xh, yh = training.gridr(rmajor, cmajor, nmastergrids, nmirrors)
                xh = xh/9.0
                
                for step in range(epochs):
                    shuffle_indices = np.random.permutation(np.arange(len(yh)))
                    xh = xh[shuffle_indices]
                    yh = yh[shuffle_indices] 
                    
                    sess.run(optimizers, feed_dict = {x:xh, y:yh})
                    
                    if step % 5 == 0:
                        print("Set: {},  Epoch: {}".format(k, step))
                    
                outlayer = np.array(sess.run(outputtriad,  feed_dict={x: xtesth}))
                
                safehitvalue = np.zeros((n_cells, xtesth.shape[0]))

                for i in range(n_cells):
                    for j in range(xtesth.shape[0]):
                        safehitvalue[i][j] = outlayer[i][j][2]
                
                idx = np.argmax(safehitvalue, 0)
                
                rightsc = 0
                for i in range(ytesth.shape[0]):
                    if ytesth[i][idx[i]] == 1 :
                        rightsc += 1
                rpc = rightsc /ytesth.shape[0]
                print(rpc)
                
                filename = "model-"+str(rmajor)+"-"+str(cmajor)+"-"+str(nmastergrids)+"-"+str(nmirrors)+"-"+str(step)+"-"+str(k)+".ckpt"
                filepath = "/home/"+now+"/"+filename
                save_path = saver.save(sess,  filepath)
                print(save_path)
                
        sess.close()
        
rmajor = 10
cmajor = 10
nmastergrids = 50
nmirrors = 20

hidlayersdef = [8, 4]
outclasses = 3

sets = 20
epochs = 100 
batchsize = 4
learningrate = 0.001

training.classifier(rmajor, cmajor, 
                    nmastergrids, nmirrors, 
                    hidlayersdef,outclasses,sets,
                    epochs, batchsize, learningrate)