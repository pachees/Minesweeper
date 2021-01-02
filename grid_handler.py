# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 20:22:20 2020

@author: risha
"""

import main_grid_creator as mgc
import mirror_grids_maker as mgm
import numpy as np


class grid_handler:
   
    
    def main(rmajor = 20, cmajor = 20, nmastergrids = 50, nmirrors = 20):
        
        """Documentation pending"""
	
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
        
        objmastergridcreator = mgc.mastergridcreator()
        objmirrorgridsmaker = mgm.mirrorgridsmaker()
        
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
