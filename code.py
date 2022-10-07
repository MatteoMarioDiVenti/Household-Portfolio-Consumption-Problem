# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:59:14 2021

@author: mmd2218
"""

import numpy as np
import time 

#number N of points
N=1000
#parameters
#theta is long run variance in Heston Model
theta=0.122251
#mean revert rate
kappa=4.996804
#correlation assume 0<=rho<=1
rho=0.5
#parameter in the diff eq solution; the square of it needs to be strictly less 2*k*theta aka Feller condition; it is vol of vol
e=0.849266
#Gamma is parameter in the utility function; it represents risk aversion
g=2
#max volatility 
vmax=1.5
#min volatility 
v1=0.001
#volatility discrete increments
dv=(vmax-v1)/(N)    
#time increments
dt=0.0001
#lamdba is the excess expected return to return volatility.
l=0.3
#The risk-free rate of return
r=0.01
#is the discount factor for the utility;  it determines the fraction of wealth to spend and impatience of investor 
delta=0.01
#creates the volatility vector
v=np.arange(v1,vmax+dv/2,dv)


#modified parameters
kprime=kappa-(g-1)*l*rho*e/g
thetaprime=theta*kappa/kprime 

#mu prime
M=kprime*thetaprime-kprime*v
#test for Mu sign for later use
Mplus=M>0
Mminus=M<0
#
sigmaprime=(e**2)*v
#create diagonals   
supdiag=Mplus[1:]*M[1:]/dv+0.5*sigmaprime[1:]/(dv**2)
subdiag=-M[:N]*Mminus[:N]/dv+0.5*sigmaprime[:N]/(dv**2)
#create the matrix S
S=np.diag(supdiag,+1)+np.diag(subdiag,-1) 
supdiag=np.append(supdiag,0)
subdiag=np.append(0,subdiag)
maindiag=-supdiag-subdiag
S=S+np.diag(maindiag,0)
#inverted matrix for later
inv=np.linalg.inv(np.eye(N+1)-S*dt)

#create initial H guess
#want to go from 100 to 0.1 in N+1 steps
H=np.arange(100,0.1-(99.9/N),-99.9/N)
#H=np.ones([N+1])/delta
#set up testing for iteration
diff=100
test=0.0001
iteration=0
#the main iteration
while diff>test:    
 #calculate H difference equation
 Hlong=np.append(H,0)
 Hlong=np.append(0,Hlong)
 Hprime=(Hlong[1:]-Hlong[:N+2])/dv
 #create K for each iteration
 #use vector form for computational speed
 HH=(Hprime[1:]*Mplus+Hprime[:N+1]*Mminus)/H
 K= delta/g+(1-1/g)*(r+0.5*g*v*(l**2-g**2*e**2*(1-rho**2)*(HH**2)))
 #store the last H
 Hprev=H 
 K=1-K*dt
 RHS=K*H+dt
 #calculate new H in three steps
 H=inv.dot(RHS)
 #Format the new H to also have zeros at ends
 #check the diff between the two
 diff= max(abs(H-Hprev))
 #just to see the number of the current iteration as it goes
 iteration+=1
 print(iteration)
print('end')
phi=(l/g)-rho*e*(Hprime[:-1]/H)
intertemp=-rho*e*(Hprime[:-1]/H)


import matplotlib.pyplot as plt

plt.plot(v[1:],H[1:],'b')
plt.xlabel('volatility')
plt.ylabel('H(vt)')
plt.plot(v[1:],phi[1:]-intertemp[1:],'g')
plt.xlabel('volatility')
plt.ylabel('Myopic part')
plt.plot(v[1:],intertemp[1:],'y')
plt.xlabel('volatility')
plt.ylabel('Intertemporal part')
plt.plot(v[1:],phi[1:],'r')
plt.xlabel('volatility')
plt.ylabel('Phi(vt)')
