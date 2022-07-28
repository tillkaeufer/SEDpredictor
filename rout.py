import numpy as np


def adjust_rout(R1in,Rout,Rtaper,gtaper,e1psilon,M1disk):
    Np=2000
    
    #Constants
    muH=2.329971445420226E-024
    Msun=1.988922500000000E+033
    AU=14959787000000.0
    
    #Convertion
    R1in=R1in*AU
    Rout=Rout*AU
    Rtaper=Rtaper*AU
    M1disk=M1disk*Msun
    
    if gtaper==None:
        gtaper=min(2,e1psilon)
    rtmp=np.zeros(Np+1)
    Ntmp=np.zeros(Np+1)
    mass=np.zeros(Np+1)
    for i in range(Np+1):
        rtmp[i] = R1in+(2*Rout-R1in)*float(i)/float(Np)
        Ntmp[i] = rtmp[i]**(-e1psilon)* np.exp(-(rtmp[i]/Rtaper)**(2.0-gtaper))
    mass[0] = 0.0
    for i in range(1,Np+1):
        dr = rtmp[i]-rtmp[i-1] 
        f1 = 4.0*np.pi*rtmp[i-1]* Ntmp[i-1]
        f2 = 4.0*np.pi*rtmp[i]  * Ntmp[i]
        mass[i] = mass[i-1] + muH*0.5*(f1+f2)*dr
    fac  = M1disk/mass[Np]
    Ntmp = fac * Ntmp
    mass = fac * mass

    for i in range(1,Np-1):
        if ((Ntmp[i]<10**20) and (mass[i] > 0.95*M1disk)):
            break
    Rout  = rtmp[i]
    return Rout/AU