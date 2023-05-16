'''
Created on Dec 3, 2018

@author: Erdong Wang
Voltage, cent meter, second

c dpr=e Er c dt--->pr=pr0+dpr
c dpz=e Ez c dt--->pz=pz0+dpz
gamma=(1-(c pr**2+c pz**2)/e0**2)**0.5
e=gamma e0
betar=c pr/e
betaz=c pz/e
vr=betar c
vz=betaz c

v1.1 np.arange-->linspace
'''
import numpy as np
from numpy import genfromtxt, dtype
import scipy.interpolate as itp #interp2d ,RectBivariateSpline
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from sympy.physics.mechanics.particle import Particle
from matplotlib import ticker, cm
import copy
from pprint import pprint

datafilepath = "../data/SHC/"
c=3e10 #cm/s

e0=0.511e6 #eV
charge_e0=-1
p0=938e6
charge_p0=1
h2=p0*2
charge_h2_1=1
charge_h2_2=2

def sfotreatment(filename):   
    sforaw = fileopen(filename)
    cha_string = ['-------------------------------------------------------------------------------']
    a = searchstring(sforaw, cha_string)
    segmentnum = int(sforaw[a[1] - 1][-1])
    seg = {}
    for elem in range(1, segmentnum + 1):  # reading the data from all the segment
        seg[elem] = pd.DataFrame(sforaw[a[elem + 1] + 5:a[elem + 1 + 1]], columns=sforaw[a[elem + 1] + 3])
        seg[elem]=seg[elem].apply(pd.to_numeric)
    # seg_panel=pd.Panel(seg)   
    #print(seg)
    return seg, segmentnum

def sf7treatment(filename):
    # sf7raw=genfromtxt(datafilepath+filename,delimiter='\t', dtype=None)
    sf7raw = fileopen(filename)   
    df = pd.DataFrame(sf7raw[33:-4], columns=sf7raw[31])  # from 33 to -4 is 6 colomn field map
    df=df.apply(pd.to_numeric)
    
    rinc= int(sf7raw[29][4])
    zinc=int(sf7raw[29][5])
    splitmin=sf7raw[27][2].split(',')
    rmin=float(splitmin[0][1:])
    zmin=float(splitmin[1][:-1])
    splitmax=sf7raw[-5]
    print(splitmax)
    rmax=float(splitmax[0])
    zmax=float(splitmax[1])
   
    return df, rmin,rmax,rinc,zmin,zmax,zinc

def fileopen(filename):  
    with open(datafilepath + filename) as raw:
        rawfile = [line.split() for line in raw]
    raw.close() 
    return rawfile

def searchstring(rawline, cha_string):
    n = 0
    a = []
    for elem in rawline:
        if elem == cha_string:
            a.append(n)
        n += 1
    return(a)

    
def geneparticle(segdatasfo,segmentnum,voltage,gradient,rmin,rmax,zmin,zmax,charge,mass,plot):
    particlesdot=pd.DataFrame(columns=segdatasfo[1].columns.values) #particlesdot, the orignal dataframe read from sfo file
    
    for elem in range(1,segmentnum+1):#segmentnum+1
        if(charge>0):
            seggene=segdatasfo[elem].loc[(abs(segdatasfo[elem]['V'])<voltage) & (segdatasfo[elem]['|E|']>gradient) & (segdatasfo[elem]['R']<15) & (segdatasfo[elem]['Z']>0)]
        elif(charge<0):        
            #seggene=segdatasfo[elem].loc[(segdatasfo[elem]['V']<voltage) & (segdatasfo[elem]['|E|']>gradient)] #calculate electron from entire ball
            seggene=segdatasfo[elem].loc[(segdatasfo[elem]['V']<voltage) & (segdatasfo[elem]['|E|']>gradient) & (segdatasfo[elem]['R']>2.2) & (segdatasfo[elem]['Z']>9)] #calculate electron from entire ball

        particlesdot=pd.concat([particlesdot,seggene], ignore_index=True)
    #print("partocles",particledot)
    particlesdotps0=particlesdot[['R','Z']]
    inizero=np.zeros(shape=(len(particlesdot.index),3))#generate zero column
    inivelo=pd.DataFrame(inizero,columns=['Vr','Vz','E']) # set up the sero column with the header~
    particlesdotps0=particlesdotps0.join(inivelo) #initial particles phase space with {R   Z   Vr   Vz    E} take the R,Z from sfo dataframe
    ddt=3*dt
    cpr=charge*particlesdot['Er'].values*c*ddt
    cpz=charge*particlesdot['Ez'].values*c*ddt
    gamma=(1+(cpr**2+cpz**2)/mass**2)**0.5
    #print(particlesdot,particlesdotps0)
    particlesdotps=copy.deepcopy(particlesdotps0)
    particlesdotps['E']=gamma*mass
    particlesdotps['Vr']=cpr*c/particlesdotps['E']
    particlesdotps['Vz']=cpz*c/particlesdotps['E']
    particlesdotps['R']=particlesdotps0['R']+particlesdotps['Vr']*ddt
    particlesdotps['Z']=particlesdotps0['Z']+particlesdotps['Vz']*ddt
    #print(particlesdotps) 
    if plot:
        if charge<0:
            plt.xlim(-rmax,rmax)
            plt.ylim(zmin,zmax)
            plt.xticks(np.arange(0,rmax,5))
            plt.yticks(np.arange(zmin,zmax,5))
            plt.plot(particlesdot['R'],particlesdot['Z'],'bo',markersize=0.2)
            plt.gca().set_aspect("equal")
            plt.savefig(savname+"_ini_par.jpg",dpi=450)
        elif charge >0:
            plt.xlim(-rmax,rmax)
            plt.ylim(zmin,zmax)
            plt.xticks(np.arange(0,rmax,5))
            plt.yticks(np.arange(zmin,zmax,5))
            plt.plot(particlesdot['R'],particlesdot['Z'],'ro',markersize=0.2)
            plt.gca().set_aspect("equal")
            plt.savefig(savname+"_ini_ion.jpg",dpi=450)

        #plt.show()
    
    return particlesdotps    


def fieldinterp(field,rmin,rmax,rinc,zmin,zmax,zinc,plot,segdatasfo, segmentnum):
    #print(field)
    print(rmin,rmax,rinc,zmin,zmax,zinc)
    #rarray=np.arange(rmin,rmax+(rmax-rmin)/rinc,(rmax-rmin)/rinc) #generate a ascend rarray
    #zarray=np.arange(zmin,zmax,(zmax-zmin)/zinc) #zmax+(zmax-zmin)/zinc
    
    
     
    
    rarray=np.linspace(rmin,rmax,rinc+1)
    zarray=np.linspace(zmin,zmax,zinc+1)
    
    print(len(rarray),len(zarray))
    matrixer=field['Er'].values.reshape(len(zarray),len(rarray)) #a 2D shape field map, Er
    matrixez=field['Ez'].values.reshape(len(zarray),len(rarray))# Ez
    matrixv=field['V'].values.reshape(len(zarray),len(rarray))# Voltage
    matrixe=field['|E|'].values.reshape(len(zarray),len(rarray))# absolute E field

    #print(matrixez.shape)
    fer=itp.RectBivariateSpline(zarray,rarray,matrixer) # the map shape is z,r
    fez=itp.RectBivariateSpline(zarray,rarray,matrixez) #do not use interp2d which is extremly slow
    fv=itp.RectBivariateSpline(zarray,rarray,matrixe)
    if 0:
        fig,ax=plt.subplots()
        levels=np.linspace(0,100000,80)
        cs=ax.contourf(rarray,zarray,matrixe,levels=levels,cmap=cm.jet)# 
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        for elem in range(1,segmentnum+1):
            ax.plot(segdatasfo[elem]['R'],segdatasfo[elem]['Z'],'y-',linewidth=0.5 )
        cbar=fig.colorbar(cs,ticks=np.linspace(0,100000,11))
        cbar.set_label("V/cm")
        
        fig.gca().set_aspect("equal")
        fig.savefig(savname+"_fielde.jpg",dpi=1800)
     
    if plot:
        max_field=330000
        fig,ax=plt.subplots()
        levels=np.linspace(0,max_field,80)
        cs=ax.contourf(rarray,zarray,matrixe,levels=levels,cmap=cm.jet)# 
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        for elem in range(1,segmentnum+1):
            ax.plot(segdatasfo[elem]['R'],segdatasfo[elem]['Z'],'y-',linewidth=0.5 )
        cbar=fig.colorbar(cs,ticks=np.linspace(0,max_field,11))
        cbar.set_label("V/cm")
        ax.set_xlim([0,18]) #final [0,10], real [0,15], local[0,7.5], shc[0,18]
        ax.set_ylim([-15,25]) #final [0,18],read [5,30],local[6.2,18],shc[-15,25]
        fig.gca().set_aspect("equal")
        fig.savefig(savname+"_fielde_local.jpg",dpi=1800)      
        #plt.contourf(rarray,zarray,matrixe,50) #if log scale, then add locator=ticker.LogLocator()
        #plt.legend(loc='right')
        #plt.gca().set_aspect("equal")
        #plt.savefig(savname+"_fielde.jpg",dpi=1800)
        #plt.show()
    #print(fer.ev([-22.3,-22.3,1],[4.3,11.2,1]))
    return fer,fez,fv
    
def trans(inipar, fez,fer,fv,charge,mass):
    dcpr=charge*fer.ev(inipar['Z'].values,inipar['R'].values)*c*dt
    dcpz=charge*fez.ev(inipar['Z'].values,inipar['R'].values)*c*dt
    cpr=inipar['Vr']*inipar['E']/c
    cpz=inipar['Vz']*inipar['E']/c
    gamma=(1+((cpr+dcpr)**2+(cpz+dcpz)**2)/mass**2)**0./5
    #print(particlesdot,particlesdotps0)
    endpar=copy.deepcopy(inipar)
    endpar['E']=gamma*mass
    endpar['Vr']=(cpr+dcpr)*c/endpar['E']
    endpar['Vz']=(cpz+dcpz)*c/endpar['E']
    endpar['R']=inipar['R']+inipar['Vr']*dt
    endpar['Z']=inipar['Z']+inipar['Vz']*dt
    
    # add on: to remove the particles achieved the boundary
    #endparf=endpar.loc[(((fer.ev(inipar['Z'].values,inipar['R'].values))**2+(fez.ev(inipar['Z'].values,inipar['R'].values))**2)/((fer.ev(endpar['Z'].values,endpar['R'].values))**2+(fez.ev(endpar['Z'].values,endpar['R'].values))**2))<2]
    endparf=endpar.loc[(((fv.ev(inipar['Z'].values,inipar['R'].values))/(fv.ev(endpar['Z'].values,endpar['R'].values)))<1.5)]

    return endparf
       


def motion(inipar,fer,fez,fv,tout,ttotal,charge,mass):    
   
    transpar=copy.deepcopy(inipar)
    trans_cum={}
    trans_cum[0.0]=transpar
    for tcum in np.arange(dt,ttotal,dt):
        
       #print(tcum," ",tout," ",tcum%tout)
        transpar=trans(transpar, fez, fer,fv,charge,mass)
        if tcum%tout==0:
            trans_cum[tcum]=transpar
        tcum+=dt   
    outpar=pd.concat(trans_cum)
    return outpar


def resultdraw(segdatasfo, segmentnum, particle,rmin,rmax,zmin,zmax,charge):
    plt.figure(figsize=(10,8))
    plt.xlim(-rmax,rmax)
    plt.ylim(zmin,zmax)
    plt.xticks(np.arange(0,rmax,5))
    plt.yticks(np.arange(zmin,zmax,5))
        #plt.plot(segdata[1]['R'],segdata[1]['Z'],'-')
        
    for elem in range(1,segmentnum+1):
        plt.plot(segdatasfo[elem]['R'],segdatasfo[elem]['Z'],'m-')
    
    if charge<0:
        if particle is not 0:
            plt.plot(particle['R'].values,particle['Z'].values,'bo',markersize=0.2)#,c=particle['E'].values-e0,'o',markersize=0.1)
            plt.gca().set_aspect("equal")
            plt.savefig(savname+"_beam.jpg",dpi=450)
            plt.show()
    elif charge>0:
        if particle is not 0:
            plt.plot(particle['R'].values,particle['Z'].values,'ro',markersize=0.2)#,c=particle['E'].values-e0,'o',markersize=0.1)
            plt.gca().set_aspect("equal")
            plt.savefig(savname+"_ion.jpg",dpi=450)
            plt.show()
          
    return 0

        
        
    
    
    
def test():
   
    segdata, segmentnum = sfotreatment(filename+".SFO")
    field,rmin,rmax,rinc,zmin,zmax,zinc=sf7treatment(filename+".SF7")

    particlesdotps=geneparticle(segdata, segmentnum,voltage,grad,rmin,rmax,zmin,zmax,spcs_charge,spcs,plot) #segdata,segmentnum, voltage,gradient
    # print(particlesdotps)
    fer,fez,fv=fieldinterp(field,rmin,rmax,rinc,zmin,zmax,zinc,plot,segdata, segmentnum )
    #print(fer(-3.028,7.715),fez(-3.028,7.715),'\n',fer(4.8492,6.33),fez(4.8492,6.33),'\n',fer(0.04,6.5335),fez(0.04,6.5335))


    outpar=motion(particlesdotps, fer, fez,fv,tout,ttotal,spcs_charge,spcs)
    outpar.to_csv("temp")
    #print(particlesdotps['E']-511000,outpar['E']-511000)
    resultdraw(segdata, segmentnum,outpar,rmin,rmax,zmin,zmax,spcs_charge)

if __name__ == '__main__':
    plot=0
    spcs=e0 # e0,p0,h2
    spcs_charge=charge_e0 #charge_e0, charge_p0, charge_h2_1, charge_h2_2
    if spcs_charge<0:
        grad=3000 #v/cm
        voltage=-500000 #V
        dt=3e-12 #
        tout=10*dt
        ttotal=1000*dt
    if spcs_charge>0:
        grad=1
        dt=1e-11
        tout=100*dt
        ttotal=10000*dt
        voltage=1   
        
    filename='SHC'
    savname=filename+'_'+str(int(grad/1000))
    
    test()