# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 07:16:07 2018

Returns a numpy array full of mirrors based on input grid

@author: pachees
"""

import numpy as np

        
class mirrorgridsmaker:
    
    
    def hiddenindeces(self, nmirrors, r, c, minesidx, nmines):
        hiddenindxarr = []
        nhidden = np.empty(nmirrors, int)
        
        for i in range(nmirrors): 
            nhidden[i] = np.random.randint(1,  r*c - nmines) #number of hidden cells are found nmirrors number of times
            					     #always leaves at least one visible mine
            hiddenindxtemp = []
            for j in range(nhidden[i]): #indeces of hidden cells are put into hiddenindxarr 
                flag = 1
                while(flag == 1):
                    rtemp = np.random.randint(0, r)
                    ctemp = np.random.randint(0, c)
                    x = [rtemp, ctemp]
                    if (x not in minesidx) and (x not in hiddenindxtemp):
                        hiddenindxtemp.append(x)
                        flag = 0
            hiddenindxarr.append(hiddenindxtemp)
            
        return nhidden, hiddenindxarr

    
    def generatemirrors(self, basegridin, basegridout, nmirrors, hiddenindxarr, rmajor, cmajor):
        
        mirrorsin = np.full((nmirrors, rmajor, cmajor), basegridin, int)
        mirrorsout = np.full((nmirrors, rmajor, cmajor), basegridout, int)

        for i in range(nmirrors):
            for e in hiddenindxarr[i]:
                mirrorsin[i][e[0]][e[1]] = 9
                mirrorsout[i][e[0]][e[1]] = 2
        
        return mirrorsin, mirrorsout
                
    
    def main(self, basegridin, basegridout, nmirrors, r, c, minesidx, nmines, rmajor, cmajor):
        
        obj = mirrorgridsmaker()
        
        nhidden, hiddenindxarr = obj.hiddenindeces(nmirrors, r, c, minesidx, nmines)
        mirrorsin, mirrorsout = obj.generatemirrors(basegridin, basegridout, nmirrors, hiddenindxarr, rmajor, cmajor)     
        return mirrorsin, mirrorsout, nhidden, hiddenindxarr
    
    