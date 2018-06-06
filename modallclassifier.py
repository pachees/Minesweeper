#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 09:44:41 2018

@author: awesotope
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 07:16:07 2018

@author: awesotope
"""
"""
Created on Sun May 27 17:05:53 2018

hidden = 2
mine = 1
open = 0


@author: awesotope
"""


import numpy as np
import tensorflow as tf
from scipy import sparse
import datetime
"""
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
import matplotlib

"""


class mastergridcreator:    

    #nmines < r*c 
    #returns indeces (minesidx as [r1,c1])
    def generateindices(self,r,c,nmines,rmajor,cmajor):   
        minesidx = []
        flag=0
    
        for i in range(nmines):
            flag = 1
            while(flag == 1):
                minesrtemp = np.random.randint(0,r)
                minesctemp = np.random.randint(0,c)
                x = [minesrtemp,minesctemp]
                if x not in minesidx:
                    minesidx.append(x)
                    flag = 0
        
        return minesidx
    #minesidx = [[1,2],[2,9],[5,3]] for eg
    
    #Creates the actual matrix and allots cell values
    def createbasematrix(self,r,c,minesidx,rmajor,cmajor): 
    
        V =[]
        Vidxr = []
        Vidxc = []
        
        for i in range(r):
            for j in range(c):
                
                cval = 0                
                
                if [i,j] in minesidx:
                    cval = 9 
                else:
                    
                    try:                       
                        if [i-1,j] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:                       
                        if [i+1,j] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:
                        if [i,j+1] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:
                        if [i,j-1] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:
                        if [i-1,j-1] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:
                        if [i-1,j+1] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:
                        if [i+1,j-1] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                    
                    try:
                        if [i+1,j+1] in minesidx:
                            cval = cval + 1
                    except ExplicitException:
                        pass
                   
                V.append(cval)
                Vidxr.append(i)
                Vidxc.append(j)
                        
        gridinp = np.array(sparse.coo_matrix((V,(Vidxr,Vidxc)),shape=(rmajor,cmajor)).todense())
        return gridinp
        
    #equivalent outgrid
    def baseoutgrid(self,rmajor,cmajor,minesidx):
        
        gridout = np.full((rmajor,cmajor),0,int)
        
        for e in minesidx:
            gridout[e[0]][e[1]] = 1
        
        return gridout
                
        
    # returns a vector of size r*c, which represents a grid with nmines mines
    def main(self,r,c,nmines,rmajor,cmajor):          
        
        obj = mastergridcreator()
        """
        if nmines > (r * c) : 
            print("Too many mines kisi minesweeper ki aukaat nahi hai cant do")
        else:
            minesidx,minesr,minesc = obj.generateindices(r,c,nmines)
            grid = obj.createbasematrix(r,c,minesidx)
        """
        minesidx = obj.generateindices(r,c,nmines,rmajor,cmajor)
        gridinp = obj.createbasematrix(r,c,minesidx,rmajor,cmajor)
        gridout = obj.baseoutgrid(rmajor,cmajor,minesidx)
        
        return minesidx,gridinp,gridout



#returns a numpy array full of mirrors based on input grid
        
class mirrorgridsmaker:
    
    
    def hiddenindeces(self,nmirrors,r,c,minesidx,nmines):
        hiddenindxarr = []
        nhidden = np.empty(nmirrors,int)
        
        for i in range(nmirrors): 
            nhidden[i] = np.random.randint(1, r*c - nmines) #number of hidden cells are found nmirrors number of times
            					     #always leaves at least one visible mine
            hiddenindxtemp = []
            for j in range(nhidden[i]): #indeces of hidden cells are put into hiddenindxarr 
                flag = 1
                while(flag == 1):
                    rtemp = np.random.randint(0,r)
                    ctemp = np.random.randint(0,c)
                    x = [rtemp,ctemp]
                    if (x not in minesidx) and (x not in hiddenindxtemp):
                        hiddenindxtemp.append(x)
                        flag = 0
            hiddenindxarr.append(hiddenindxtemp)
            
        return nhidden,hiddenindxarr

    
    def generatemirrors(self,basegridin,basegridout,nmirrors,hiddenindxarr,rmajor,cmajor):
        
        mirrorsin = np.full((nmirrors,rmajor,cmajor),basegridin,int)
        mirrorsout = np.full((nmirrors,rmajor,cmajor),basegridout,int)

        for i in range(nmirrors):
            for e in hiddenindxarr[i]:
                mirrorsin[i][e[0]][e[1]] = 9
                mirrorsout[i][e[0]][e[1]] = 2
        
        return mirrorsin,mirrorsout
                
    
    def main(self,basegridin,basegridout,nmirrors,r,c,minesidx,nmines,rmajor,cmajor):
        
        obj = mirrorgridsmaker()
        
        nhidden,hiddenindxarr = obj.hiddenindeces(nmirrors,r,c,minesidx,nmines)
        mirrorsin,mirrorsout = obj.generatemirrors(basegridin,basegridout,nmirrors,hiddenindxarr,rmajor,cmajor)     
        return mirrorsin,mirrorsout,nhidden,hiddenindxarr

class Baapclass:
   
    
    def main(rmajor = 20, cmajor = 20, nmastergrids = 50, nmirrors = 20):
        
        
        #maxpossnmines = rmajor*cmajor - 1 #maximum possible number of mines in any of these examples
        
        #Minesidxs = np.empty((nmastergrids,maxpossnmines,2),dtype=int)
        #Hiddenidxs = np.empty((nmastergrids,nmirrors,maxpossnmines,2),dtype=int) #maxposshiddenidxs = maxpossnmines
        
        Gridsinpnp = np.empty((nmastergrids,nmirrors,rmajor,cmajor),dtype=int) 
        Gridsoutnp = np.empty((nmastergrids,nmirrors,rmajor,cmajor),dtype=int)
                
        Mastergridsin = np.empty((nmastergrids,rmajor,cmajor),int)
        Mastergridsout = np.empty((nmastergrids,rmajor,cmajor),int)
        
        Nmines = np.empty(nmastergrids,dtype=int)
        Nhidden = np.empty((nmastergrids,nmirrors),dtype=int)
        #Gridsrc = np.array([np.random.randint(2,rmajor+1,(nmastergrids)),np.random.randint(2,cmajor+1,(nmastergrids))])
        Gridsrc = np.array([np.full(nmastergrids,rmajor),np.full(nmastergrids,cmajor)])
        
        objmastergridcreator = mastergridcreator()
        objmirrorgridsmaker = mirrorgridsmaker()
        
        for b in range(nmastergrids):
            
            Nmines[b] = np.random.randint(1, Gridsrc[0][b]*Gridsrc[1][b]-1) #leaves at least 2 non mine
								     #necessary, since at least one will be hidden
								     #and at least one remaining visible

            minesidx,Mastergridsin[b],Mastergridsout[b] = objmastergridcreator.main(Gridsrc[0][b],Gridsrc[1][b],Nmines[b],rmajor,cmajor)
            Gridsinpnp[b],Gridsoutnp[b],Nhidden[b],hiddenindxarr = objmirrorgridsmaker.main(Mastergridsin[b],Mastergridsout[b],nmirrors,Gridsrc[0][b],Gridsrc[1][b],minesidx,Nmines[b],rmajor,cmajor)
                
            """
            ###### RESIZING BLOCK ##########
            for i in range(rmajor*cmajor-1 - Nmines[b]):
                minesidx.append([-5,-5])
            for i in range(nmirrors):
                for j in range(rmajor*cmajor-1 - Nhidden[b][i]):
                    hiddenindxarr[i].append([-5,-5])
            #################################                                                           r,c,nmines,rmajor,cmajor

            Minesidxs[b] = np.array(minesidx)
            Hiddenidxs[b] = np.array(hiddenindxarr)
            """
        
        return Gridsinpnp,Gridsoutnp



class training:
    
    def gridr(rmajor,cmajor,nmastergrids,nmirrors):

        Gridsin,Gridsout = Baapclass.main(rmajor,cmajor,nmastergrids,nmirrors)

        batchsize = nmastergrids * nmirrors 
                
        rmajor = rmajor
        cmajor = cmajor
        
        flatin = np.zeros((batchsize,rmajor*cmajor), dtype=float)
        flatout = np.zeros((batchsize,rmajor*cmajor), dtype=int)
        c = 0
        
        for i in range(nmastergrids):
            for j in range(nmirrors):
                flatin[c] = Gridsin[i][j].flatten()
                flatout[c] = Gridsout[i][j].flatten()
                c += 1
                
        return flatin,flatout
    
    def classifier(rmajor,cmajor,nmastergrids,nmirrors,hidlayersdef,outclasses,sets,epochs,batchsize,learningrate):
        tf.reset_default_graph()
        
        n_cells = rmajor*cmajor
        n_hidlayers = len(hidlayersdef)
        
        hidlayersdef.insert(0,n_cells)
        hidlayersdef.insert(n_hidlayers,outclasses)
        
        
        n_layers = n_hidlayers+2
        networks = [[0] * (n_layers-1)]*n_cells
        
        outputtriad = [0]*n_cells
        
        costs = [0]*n_cells
        optimizers = [0]*n_cells
        
        x = tf.placeholder(dtype = tf.float64, name = 'Xwala', shape = [None,n_cells])
        y = tf.placeholder(dtype = tf.int64, name = 'Yout', shape = [None, n_cells])
        
        
        for i in range(n_cells):
            
            networks[i][0] = x
            for j in range(1,n_hidlayers+1):
                networks[i][j] = tf.contrib.layers.fully_connected(networks[i][j-1], hidlayersdef[j], activation_fn=tf.nn.relu)
            outputtriad[i] = tf.contrib.layers.fully_connected(networks[i][n_hidlayers], outclasses, activation_fn=tf.nn.sigmoid)
            
            
            costs[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None,
                                                               labels = y[:,i],
                                                               logits = outputtriad[i])

            optimizers[i] = tf.train.AdamOptimizer(learningrate).minimize(costs[i])
        
        
        init = tf.global_variables_initializer()
        
        xtesth,ytesth = training.gridr(rmajor,cmajor,nmastergrids,nmirrors)
        
        saver = tf.train.Saver()
        now = datetime.datetime.now()
        now = str(now)
        with tf.Session() as sess:
            sess.run(init)
            
            for k in range(sets):
                xh,yh = training.gridr(rmajor,cmajor,nmastergrids,nmirrors)
                xh = xh/9.0
                
                for step in range(epochs):
                    shuffle_indices = np.random.permutation(np.arange(len(yh)))
                    xh = xh[shuffle_indices]
                    yh = yh[shuffle_indices] 
                    
                    sess.run(optimizers,feed_dict = {x:xh,y:yh})
                    
                    if step % 5 == 0:
                        print("Set: {}, Epoch: {}".format(k,step))
                    
                outlayer = np.array(sess.run(outputtriad, feed_dict={x: xtesth}))
                
                safehitvalue = np.zeros((n_cells,xtesth.shape[0]))

                for i in range(n_cells):
                    for j in range(xtesth.shape[0]):
                        safehitvalue[i][j] = outlayer[i][j][2]
                
                idx = np.argmax(safehitvalue,0)
                
                rightsc = 0
                for i in range(ytesth.shape[0]):
                    if ytesth[i][idx[i]] == 1 :
                        rightsc += 1
                rpc = rightsc /ytesth.shape[0]
                print(rpc)
                
                filename = "model-"+str(rmajor)+"-"+str(cmajor)+"-"+str(nmastergrids)+"-"+str(nmirrors)+"-"+str(step)+"-"+str(k)+".ckpt"
                filepath = "/home/awesotope/dev/python/Minesweeper/Minesweeper/Allclassifier/Main/Run"+now+"/"+filename
                save_path = saver.save(sess, filepath)
                print(save_path)
                
        sess.close()
        
rmajor = 10
cmajor = 10
nmastergrids = 50
nmirrors = 20

hidlayersdef = [8,4]
outclasses = 3

sets = 20
epochs = 100 
batchsize = 4
learningrate = 0.001

training.classifier(rmajor,cmajor,nmastergrids,nmirrors,hidlayersdef,outclasses,sets,epochs,batchsize,learningrate)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
