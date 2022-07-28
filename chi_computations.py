import numpy as np

def chi_window(Fmod,Fphot,sphot,lphot):
    #--------- compute chi in spectral windows --------
    window = [0.3,1.0,3.0,8.0,15.0,30.0,100.0,300.0,1000.0,3000.0,1.E+99]
    iwin = 0
    Nchi = 0
    chi2 = 0.0
    Nwin = 0
    chi2win = 0.0 
    Ndat=len(Fphot)
    for i in range(0,Ndat):
        dchi =  np.log(Fmod[i]/Fphot[i])/(sphot[i]/Fphot[i]) #is this really log or log10
        if lphot[i]>window[iwin]:     # window finished
            if iwin>0:
#                print('%3d %3d %7.3f' % (iwin+1,Nwin,np.sqrt(chi2win/Nwin)))
                chi2 += chi2win/Nwin 
                Nchi += 1
            #else:
            #    print()

            iwin += 1
            Nwin = 0
            chi2win = 0.0

        chi2win += dchi**2            # add dchi to window-chi
        Nwin += 1
#        print('%13.5f %10.6e %10.6e %6.2f%c %7.2f' % (lphot[i],Fphot[i],Fmod[i],sphot[i]/Fphot[i]*100.0,"%",dchi))

    #print(Nchi+1,Nwin,np.sqrt(chi2win/Nwin))
    if Nwin>0:
        chi2 += chi2win/Nwin          # last window
        Nchi += 1

    chi = np.sqrt(chi2/Nchi)        # total chi
    #print(' ==>  chi = %8.4f' % (chi))
    return chi

def chi_squared(Fmod,Fphot,sphot,lphot):
    #--------- compute chi properly --------

    Ndat=len(Fphot)
    chi2=0.0
    for i in range(0,Ndat):
        diff=Fphot[i]-Fmod[i]
        chi2+=(diff/sphot[i])**2

    chi = np.sqrt(chi2)        # total chi

    return chi2

def log_like(Fmod,Fphot,sphot,lphot):
    #--------- compute loglike properly --------
    
    chi2=chi_squared(Fmod,Fphot,sphot,lphot)
    const=np.sum(np.log(2*np.pi*(sphot)**2))
    loglikelihood =  -0.5 * (chi2 +const) #does sig_obs+sig_model account 

    return loglikelihood

def chi_squared_reduced(Fmod,Fphot,sphot,lphot):
    #--------- compute chi reduced properly --------

    Ndat=len(Fphot)
    chi2=0.0
    for i in range(0,Ndat):
        diff=Fphot[i]-Fmod[i]
        chi2+=(diff/sphot[i])**2
    chi2=chi2/Ndat
    chi = np.sqrt(chi2)        # total chi

    return chi2
