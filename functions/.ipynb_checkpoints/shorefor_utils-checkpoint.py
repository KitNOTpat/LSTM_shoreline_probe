#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:52:59 2018

@author: z5167465
"""

import math
import numpy as np
import scipy
import scipy.signal

# def fallvelocity(D,Tw):
    
#     #D = grain size (m)
#     #Tw = temperature in degrees C
#     #w returned in m/s
    
#     D=D*100
#     ROWs=2.75 #Density of sand (Mark 2.75 kRISTEN 2.65)
#     g=981     #Gravity n cm**2/s
    
#     T=np.array([5 ,10, 15, 20, 25])
#     v   =np.array([0.0157, 0.0135, 0.0119, 0.0105, 0.0095])
#     ROW =np.array([1.028, 1.027, 1.026, 1.025, 1.024])
    
#     vw=np.interp(Tw,T,v)
#     ROWw=np.interp(Tw,T,ROW)    
    
#     A=((ROWs-ROWw)*g*(D**3))/(ROWw*(vw**2))
    
#     if A < 39:
#         w=((ROWs-ROWw)*g*(D**2))/(18*ROWw*vw)
#     else:
#         if A < 10**4:   
#        	  w=((((ROWs-ROWw)*g/ROWw)**0.7)*(D**1.1))/(6*(vw**0.4))
#         else:
#             w=np.sqrt(((ROWs-ROWw)*g*D)/(0.91*ROWw))
            
#     w=w/100 #convert to SI (m/s)
    
#     return w

def fallvelocity(D,Tw):
    
    #D = grain size (m)
    #Tw = temperature in degrees C
    #w returned in m/s
    
    D=D*100
    ROWs=2.75 #Density of sand 
    g=981     #Gravity n cm**2/s
    	
    T=np.array([5 ,10, 15, 20, 25])
    v   =np.array([0.0157, 0.0135, 0.0119, 0.0105, 0.0095])
    ROW =np.array([1.028, 1.027, 1.026, 1.025, 1.024])
    	
    vw=np.interp(Tw,T,v)
    ROWw=np.interp(Tw,T,ROW)    
    	
    A=((ROWs-ROWw)*g*(D**3))/(ROWw*(vw**2))

    if  A < 39:
        w = ((ROWs-ROWw)*g*(D**2))/(18*ROWw*vw)
    	
    elif (A < 10**4) & (A > 39):

     	w= ((((ROWs-ROWw)*g/ROWw)**0.7)*(D**1.1))/(6*(vw**0.4))   
    elif A > 10**4:
     	w= np.sqrt(((ROWs-ROWw)*g*D)/(0.91*ROWw))
    w=w/100 #convert to SI (m/s)

    return w


def brierrs(xm,x,xb,dx):
    #Brier Skils Function
    
    #xm - measured data
    #x - model
    #xb benchmark data
    #dx - measurement error
    
    BSS = 1 - np.mean(((np.absolute(x-xm))-dx)**2)/np.mean((xb-xm)**2)
    
    return BSS

def akaikeIC(xm,x,k):
    #Akaike's Information Criteria determine's the relative appropriate of a
    #model. See Akaike, 1974 or Kuriyama, 2012.
    #xm - measured data
    #x - model data
    #k - the number of free parameters
    n=len(xm)
    resid=xm-x
    AIC=n*(np.log10(2*np.pi)+1)+n*np.log10(np.var(resid))+2*k
    
    return AIC

def calcHoFromHs(Hs,T,h):
    
    #Function to reverse shoal wave height to deep water equivalent.
    #Assumes no refraction. Used in R2 (Stockdon et al 06) where Ho is needed.
    #Under variable bathy, using an inshore wave height and reverse shoaling may 
    #be more accurate than using the offshore one registered at wave gauge. 
    #Hs= significant wave height in water depth h
    #T = wave period
    
    #Adapted from Kristen 09
    
    pi=math.pi
    gamma=0.78
    g=9.81
    Cgo=0.25*g*T/pi
    k=dispsol2(h, 1/T)
    L=2*pi/k
    C=L/T
    Cg = 0.5*(1+ 4*pi/L/np.sinh(4*pi*h/L))*C
    Ho=Hs*np.sqrt(Cg/Cgo)
    
    return Ho

def calcHoShoal(H,T,h1):
    
    #Function to reverse shoal wave height to deep water equivalent
    
    #Adapted from Kristen 2010
    
    pi=math.pi
    g=9.81
    Cgo=1/4*g*T/pi
    Cg=calcCg(h1,T)
    Ks=np.sqrt(Cgo/Cg)
    Ho=H/Ks    
    return Ho


def calcCg(h,T):
    
    g=9.81
    y=4.03*h/(T**2)
    kd2=y**2+y/(1+(0.666*y)+(0.355*y**2)+(0.161*y**3)+(0.0632*y**4)+(0.0218*y**5)+(0.00564*y**6))
    kh=np.sqrt(kd2)
    Cg=g*T/(2*math.pi)*(np.tanh(kh))*(0.5*(1+2*kh/np.sinh(2*kh)))
    
    return Cg

#to be changed using patos code
    



def WS85FilterConv(omega,D,phi,dt):
    
#code to calculate the omegaMean value based on Wright and Short 1985 paper.
#inputs 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#omega = time series of dimensionless fall velocity
#dt = time step in Omega, hrs
#D = number of days used in back filtering - set this to 2*phi
#phi = number of days when beach memory is 10%
#The method utilises convolution theorem to apply the filter. It is much
#faster than using loops! However, the methodology does not give good
#results for the last phi data points which must be calculated using the
#slow looping method
# Adapted from Mark davidson 11/7/12
     
    # phi=phi.astype(int)
    # D=D.astype(int)
    
    #print('Computing equilibrium omega value ... WS85 convolution')
    
    dt=dt/24 # dt in days #changed KSNov17
    D=np.round(D/dt).astype(int) #D in hours
    phi=np.round(phi/dt)  #phi in hours
    meanOmega=np.mean(omega)
    omega=omega-meanOmega
    
    #Define filter back to front for convolution
    ii=np.arange(0,D-1)
    padding=np.zeros(D-1)
    filterCoeff=10**(-np.abs(ii)/phi)
    
    filterCoeff=np.hstack((padding,filterCoeff))
    window=filterCoeff/np.sum(filterCoeff)

    
    #perform convolution
    omegaFiltered=scipy.signal.convolve(omega,window, mode='same')
    
    #Finally add on mean
    omegaFiltered=omegaFiltered+meanOmega
    
    return omegaFiltered
    

    


    

    