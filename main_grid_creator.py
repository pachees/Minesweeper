# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:05:53 2018

Creates Minesweeper grids

hidden = 2
mine = 1
open = 0


@author: pachees
"""


import numpy as np
from scipy import sparse
#from Exception import ExplicitException

class main_grid_creator:    

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
    def baseoutgrid(self, rmajor, cmajor, minesidx):
        
        gridout = np.full((rmajor, cmajor), 0, int)
        
        for e in minesidx:
            gridout[e[0]][e[1]] = 1
        
        return gridout
                
        
    # returns a vector of size r*c, which represents a grid with nmines mines
    def main(self, r, c, nmines, rmajor, cmajor):          
        
        obj = main_grid_creator()
        
        """
        if nmines > (r * c) : 
            print("Too many mines kisi minesweeper ki aukaat nahi hai cant do")
        else:
            minesidx,minesr,minesc = obj.generateindices(r,c,nmines)
            grid = obj.createbasematrix(r,c,minesidx)
        """
        
        minesidx = obj.generateindices(r, c, nmines, rmajor, cmajor)
        gridinp = obj.createbasematrix(r, c, minesidx, rmajor, cmajor)
        gridout = obj.baseoutgrid(rmajor, cmajor, minesidx)
        
        return minesidx, gridinp, gridout
    
    

