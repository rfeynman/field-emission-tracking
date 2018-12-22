'''
Created on Dec 3, 2018

@author: wange
Voltage, cent meter, second

c dpr=e Er c dt--->pr=pr0+dpr
c dpz=e Ez c dt--->pz=pz0+dpz
gamma=(1-(c pr**2+c pz**2)/e0**2)**0.5
e=gamma e0
betar=c pr/e
betaz=c pz/e
vr=betar c
vz=betaz c

'''
import numpy as np
from numpy import genfromtxt, dtype
import scipy.interpolate as itp #interp2d ,RectBivariateSpline
import math
import pandas as pd
import matplotlib.pyplot as plt
from sympy.physics.mechanics.particle import Particle
from matplotlib import ticker, cm
import copy

datafilepath = "../data/high/"
dt=3e-12 #s
c=3e10 #cm/s
e0=0.511e6 #eV
grad=117000
filename='jlab_noshed'
savname=filename+'_'+str(int(grad/1000))

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
    splitmax=sf7raw[28][2].split(',')
    rmax=float(splitmax[0][1:])
    zmax=float(splitmax[1][:-1])
    
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

    
def geneparticle(segdatasfo,segmentnum,voltage,gradient,plot):
    particlesdot=pd.DataFrame(columns=segdatasfo[1].columns.values) #particlesdot, the orignal dataframe read from sfo file
    
    for elem in range(1,segmentnum+1):#segmentnum+1
        seggene=segdatasfo[elem].ix[(segdatasfo[elem]['V']<voltage) & (segdatasfo[elem]['|E|']>gradient)]
        particlesdot=pd.concat([particlesdot,seggene], ignore_index=True)
    #print("partocles",particledot)
    particlesdotps0=particlesdot[['R','Z']]
    inizero=np.zeros(shape=(len(particlesdot.index),3))#generate zero column
    inivelo=pd.DataFrame(inizero,columns=['Vr','Vz','E']) # set up the sero column with the header~
    particlesdotps0=particlesdotps0.join(inivelo) #initial particles phase space with {R   Z   Vr   Vz    E} take the R,Z from sfo dataframe
    ddt=3*dt
    cpr=-particlesdot['Er'].values*c*ddt
    cpz=-particlesdot['Ez'].values*c*ddt
    gamma=(1+(cpr**2+cpz**2)/e0**2)**0.5
    #print(particlesdot,particlesdotps0)
    particlesdotps=copy.deepcopy(particlesdotps0)
    particlesdotps['E']=gamma*e0
    particlesdotps['Vr']=cpr*c/particlesdotps['E']
    particlesdotps['Vz']=cpz*c/particlesdotps['E']
    particlesdotps['R']=particlesdotps0['R']+particlesdotps['Vr']*ddt
    particlesdotps['Z']=particlesdotps0['Z']+particlesdotps['Vz']*ddt
    #print(particlesdotps) 
    if plot:
        plt.xlim(-25,25)
        plt.ylim(-25,30)
        plt.xticks(np.arange(0,25,5))
        plt.yticks(np.arange(-25,30,5))
        plt.plot(particlesdot['R'],particlesdot['Z'],'bo',markersize=0.2)
        plt.gca().set_aspect("equal")
        plt.savefig(savname+"_ini_par.jpg",dpi=450)
        #plt.show()
    
    return particlesdotps    


def fieldinterp(field,rmin,rmax,rinc,zmin,zmax,zinc,plot):
    #print(field)
    rarray=np.arange(rmin,rmax+(rmax-rmin)/rinc,(rmax-rmin)/rinc) #generate a ascend rarray
    zarray=np.arange(zmin,zmax+(zmax-zmin)/zinc,(zmax-zmin)/zinc)
    matrixer=field['Er'].values.reshape(len(zarray),len(rarray)) #a 2D shape field map, Er
    matrixez=field['Ez'].values.reshape(len(zarray),len(rarray))# Ez
    matrixv=field['V'].values.reshape(len(zarray),len(rarray))# Voltage
    matrixe=field['|E|'].values.reshape(len(zarray),len(rarray))# absolute E field

    #print(matrixez.shape)
    fer=itp.RectBivariateSpline(zarray,rarray,matrixer) # the map shape is z,r
    fez=itp.RectBivariateSpline(zarray,rarray,matrixez) #do not use interp2d which is extremly slow
    fv=itp.RectBivariateSpline(zarray,rarray,matrixe)
    if plot:
        plt.contourf(rarray,zarray,matrixe,50 ) #if log scale, then add locator=ticker.LogLocator()
        plt.gca().set_aspect("equal")
        plt.savefig(savname+"_fielde.jpg",dpi=1800)
        #plt.show()
    #print(fer.ev([-22.3,-22.3,1],[4.3,11.2,1]))
    return fer,fez,fv
    
def trans(inipar, fez,fer,fv):
    dcpr=-fer.ev(inipar['Z'].values,inipar['R'].values)*c*dt
    dcpz=-fez.ev(inipar['Z'].values,inipar['R'].values)*c*dt
    cpr=inipar['Vr']*inipar['E']/c
    cpz=inipar['Vz']*inipar['E']/c
    gamma=(1+((cpr+dcpr)**2+(cpz+dcpz)**2)/e0**2)**0.5
    #print(particlesdot,particlesdotps0)
    endpar=copy.deepcopy(inipar)
    endpar['E']=gamma*e0
    endpar['Vr']=(cpr+dcpr)*c/endpar['E']
    endpar['Vz']=(cpz+dcpz)*c/endpar['E']
    endpar['R']=inipar['R']+inipar['Vr']*dt
    endpar['Z']=inipar['Z']+inipar['Vz']*dt
    
    # add on: to remove the particles achieved the boundary
    #endparf=endpar.loc[(((fer.ev(inipar['Z'].values,inipar['R'].values))**2+(fez.ev(inipar['Z'].values,inipar['R'].values))**2)/((fer.ev(endpar['Z'].values,endpar['R'].values))**2+(fez.ev(endpar['Z'].values,endpar['R'].values))**2))<2]
    endparf=endpar.loc[(((fv.ev(inipar['Z'].values,inipar['R'].values))/(fv.ev(endpar['Z'].values,endpar['R'].values)))<1.5)]

    return endparf
       
    
def motion(inipar,fer,fez,fv,tout,ttotal):    
   
    transpar=copy.deepcopy(inipar)
    trans_cum={}
    trans_cum[0.0]=transpar
    for tcum in np.arange(dt,ttotal,dt):
        
        transpar=trans(transpar, fez, fer,fv)
        if tcum%tout==0:
            trans_cum[tcum]=transpar
        tcum+=dt   
    outpar=pd.concat(trans_cum)
    return outpar


def resultdraw(segdatasfo, segmentnum, particle):
    plt.figure(figsize=(10,8))
    plt.xlim(-30,30)
    plt.ylim(-30,30)
    plt.xticks(np.arange(0,25,5))
    plt.yticks(np.arange(-25,30,5))
        #plt.plot(segdata[1]['R'],segdata[1]['Z'],'-')
        
    for elem in range(1,segmentnum+1):
        plt.plot(segdatasfo[elem]['R'],segdatasfo[elem]['Z'],'m-')
    
    if particle is not 0:
        plt.plot(particle['R'].values,particle['Z'].values,'bo',markersize=0.2)#,c=particle['E'].values-e0,'o',markersize=0.1)
        
    plt.gca().set_aspect("equal")
    plt.savefig(savname+"_beam.jpg",dpi=450)
    #plt.show()      
    return 0

        
        
    
    
    
def test():
    plot=1
    segdata, segmentnum = sfotreatment(filename+".SFO")
    particlesdotps=geneparticle(segdata, segmentnum,-200000,grad,plot) #segdata,segmentnum, voltage,gradient
    field,rmin,rmax,rinc,zmin,zmax,zinc=sf7treatment(filename+".SF7")
    fer,fez,fv=fieldinterp(field,rmin,rmax,rinc,zmin,zmax,zinc,plot)
    #print(fer(-3.028,7.715),fez(-3.028,7.715),'\n',fer(4.8492,6.33),fez(4.8492,6.33),'\n',fer(0.04,6.5335),fez(0.04,6.5335))


    tout=20*dt
    ttotal=300*dt
    outpar=motion(particlesdotps, fer, fez,fv,tout,ttotal)
    print(particlesdotps['E']-511000,outpar['E']-511000)
    resultdraw(segdata, segmentnum,outpar)

if __name__ == '__main__':
    test()