# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:14:35 2022

@author: tfk1
"""
import numpy as np
import matplotlib.pyplot as plt
import glob



class density_plot:
    def __init__(self,para_dict,N_rad=199,N_ver=99):
        '''
        dictionary with the all relevant parameters 
        '''
        self.parameter_dict=para_dict
        '''
        2-D array with the density structure
        '''
        self.density_structure=np.zeros((N_rad+1,N_ver+1))
        '''
        list with the surface density by radius
        '''
        self.surface_den=np.ones(N_rad)
        '''
        list with the radial coordiantes
        '''
        self.radii=None
        self.N_rad=N_rad
        '''
        list with the height coordinates (z/r)
        '''
        self.z_list=None
        self.N_ver=N_ver
        '''
        single or two zone models and counter to change inner and outer zone
        '''
        self.two_zone=False
        self.inner_zone=True

        '''
        physical constants
        '''
        self.const={'muH':2.329971445420226E-024,
                    'Msun':1.988922500000000E+033,
                    'AU':14959787000000.0}
        
    def adjust_rout(self,debug=False):
        #number of points for adjusting the radius
        Np=2000
        
        '''
        get relevant parameters from dict
        '''
        R1in=self.parameter_dict['Rin']
        Rtaper=self.parameter_dict['Rtaper']
        Rout=4*Rtaper
        e1psilon=self.parameter_dict['epsilon']
        if 'gtaper' in self.parameter_dict:
            gtaper=self.parameter_dict['gtaper']
        else:
            gtaper=min(2,e1psilon)
        M1disk=self.parameter_dict['Mdisk']

        
        #gettting the constants
        AU=self.const['AU']
        Msun=self.const['Msun']
        muH=self.const['muH']

        #Convertion from AU to cm and Msun to kg
        R1in=R1in*AU
        Rout=Rout*AU
        Rtaper=Rtaper*AU
        M1disk=M1disk*Msun
        
        #setting radial lists of points
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
        
        #saving Rout
        self.parameter_dict['Rout']=Rout/AU

        
    def get_density_structure(self):
        if 'Rout' not in self.parameter_dict:
            #print('Adjusting Rout')
            self.adjust_rout(self)
        
        #check if single or two-zone file
        if 'R2in' in self.parameter_dict:
            self.two_zone=True
        else:
            self.two_zone=False
        
        #setup the radial and vertical points 
        if self.two_zone:
            min_rad=self.parameter_dict['R2in']
        else:
            min_rad=self.parameter_dict['Rin']
        max_rad=self.parameter_dict['Rout']
        
        self.radii=np.logspace(np.log10(min_rad),np.log10(max_rad),num=self.N_rad+1)
        self.z_list=np.arange(0,2,2/(self.N_ver+1))
        #print(np.shape(self.z_list))
        
        self.get_radial_structure(self)
        if self.two_zone:
            inner_density=self.surface_den
            self.get_radial_structure(self)
            outer_density=self.surface_den
            tot_structure=np.ones((self.N_rad+1))
            tot_structure[:len(inner_density)]=inner_density
            tot_structure[-len(outer_density):]=outer_density
            self.surface_den=tot_structure
        self.vertical_structure(self)
    
    def get_radial_structure(self,debug=False):
        new_rtmp=True
        debug=False
        two_zone=self.two_zone
        '''
        get relevant parameters from dict
        '''
        if two_zone and self.inner_zone:
            R1in=self.parameter_dict['R2in']
            Rout=self.parameter_dict['R2out']
            Rtaper=0
            gtaper=0
            e1psilon=self.parameter_dict['e2psilon']
            
            M1disk=self.parameter_dict['M2disk']
            self.inner_zone=False
        else:
            R1in=self.parameter_dict['Rin']
            Rtaper=self.parameter_dict['Rtaper']
            Rout=self.parameter_dict['Rout']
            e1psilon=self.parameter_dict['epsilon']
            if 'gtaper' in self.parameter_dict:
                gtaper=self.parameter_dict['gtaper']
            else:
                gtaper=min(2,e1psilon)
            M1disk=self.parameter_dict['Mdisk']

        
        #gettting the constants
        AU=self.const['AU']
        Msun=self.const['Msun']
        muH=self.const['muH']

        rtmp=self.radii
        if new_rtmp:
            idx_rtmp=np.where((np.round(np.array(rtmp),3)>=np.round(R1in,3)) & (np.round(np.array(rtmp),3)<=np.round(Rout,3)))[0]
            rtmp=rtmp[idx_rtmp]
        if debug:
            print(R1in,Rtaper,Rout,M1disk)
        
        if debug:
            print(np.min(rtmp*AU),R1in)
            print(np.max(rtmp*AU),Rout)
           
        Np=len(rtmp)-1

        Ntmp=np.zeros(Np+1)
        mass=np.zeros(Np+1)
        if debug:
            print(e1psilon,Rtaper,gtaper)
            print(np.shape(rtmp))
        for i in range(Np+1):
            #rtmp[i] = R1in+(Rout-R1in)*float(i)/float(Np)
            if Rtaper!=0:
                Ntmp[i] = (rtmp[i]*AU)**(-e1psilon)* np.exp(-(rtmp[i]/Rtaper)**(2.0-gtaper))
            else:
                Ntmp[i] = (rtmp[i]*AU)**(-e1psilon)
        if debug:
            print(Ntmp)
        mass[0] = 0.0

        for i in range(1,Np+1):
            dr = (rtmp[i]-rtmp[i-1])*AU 
            f1 = 4.0*np.pi*rtmp[i-1]*AU* Ntmp[i-1]
            f2 = 4.0*np.pi*rtmp[i]*AU  * Ntmp[i]
            mass[i] = mass[i-1] + muH*0.5*(f1+f2)*dr
        if debug:
            print('tot_mass',mass[Np])
            print('parameter mass',M1disk*Msun)
        fac  = M1disk*Msun/mass[Np]
        Ntmp = fac * Ntmp
        mass = fac * mass
        if debug:
            print('new mass', mass)
            print(fac)
 
        self.surface_den=Ntmp
    def vertical_structure(self,debug=False):
        two_zone=self.two_zone
        debug=False
        if two_zone:
            r_2=1
            H2=self.parameter_dict['MCFOST_H2']
            beta2=self.parameter_dict['MCFOST_B2']
            r2out=self.parameter_dict['R2out']
       
        r_0=100
        H0=self.parameter_dict['MCFOST_H0']
        beta=self.parameter_dict['MCFOST_BETA']
        rin=self.parameter_dict['Rin']

        z_list=self.z_list
        r_list=self.radii
        surf=self.surface_den
        
        AU=self.const['AU']
        if debug:
            print('Shape surf',np.shape(surf))
        for j in range(len(r_list)):
            stop=False
            r=r_list[j]
            N_surf=surf[j]
            if debug:
                print('R')
                print(r)
            density_list=np.zeros_like(z_list)
            if two_zone:
                if r>rin:
                    Hr=H0*(r/r_0)**(beta)
                elif r<r2out:
                    Hr=H2*(r/r_2)**(beta2)
                else:
                    stop=True
                    self.density_structure[j]=1
            else:
                Hr=H0*(r/r_0)**(beta)

            if not stop:
                if debug:
                    print('H')
                    print(Hr/r)
                f1=2*N_surf/np.sqrt(2*np.pi)/(Hr*AU)
                for i in range(len(z_list)):
                    z=z_list[i]*r
                    f2=np.exp(-(z/Hr)**2/2)
                    nHtot=f1*f2
                    if debug:
                        print(nHtot)
                    density_list[i] = max(1.0,nHtot)
                self.density_structure[j]=density_list
    def plot_fig_columndensity(self):
        fig, ax = plt.subplots(figsize=(9,6))
        dens=np.log10(self.surface_den)
        radii=self.radii
        ax.plot(radii,dens,label='$\\Sigma_{\\rm gas}$')# / (1.4\\rm amu)$')
        ax.plot(radii,dens-2,label='$\\Sigma_{\\rm dust}$')# / (1.4\\rm amu)$') # HAVE TO CHECK WHY PETER USED THIS
        ax.set_xscale('log')
        ax.set_xlabel('$r$ [AU]')
        ax.set_ylabel('$\\log_{10} N_{\\rm <H>} \\rm [cm^{-2}]$')
        ax.set_xlim([np.min(radii),np.max(radii)])
        ax.set_ylim(ymin=20)
        ax.legend()
        return fig
    def plot_fig_density(self,zmax=1.0,cmin=4.0):
        dens=np.log10(self.density_structure)
        radii=self.radii
        z_list=self.z_list
        
        min_val=cmin
        max_val=np.max(dens)
        steps=22
        levels=np.arange(min_val,max_val*1.01,(max_val-min_val)/steps)
        fig, ax = plt.subplots(figsize=(9,6))
        plotax=ax.contourf(radii,z_list,dens.T,levels=levels,cmap='viridis',extend='both')
        ax.set_xscale('log')
#        ax.tick_params(direction='in' ,color='white', labelcolor='black')
        ax.set_ylabel('$z/r$')
        ax.set_xlabel('$r$ [AU]')
        ax.set_ylim([0,zmax])
        
        fig.colorbar(plotax,label='$\\log_{10} n_{\\rm <H>} \\rm [cm^{-3}]$')
        return fig
    
    
# hrd utils

def load_siesstracks(debug=False):
    files=glob.glob('./data/Siesstracks/*.hrd')
    files.sort()
    L=[]
    T=[]
    ages=[]
    m=[]
    max_T,min_T=0,100000
    max_L,min_L=0,10
    for i in range(len(files)):
        if debug: print(files[i],np.shape(np.loadtxt(files[i])[:,2]))
        idx=np.max(np.where(np.loadtxt(files[i])[:,1]==1))+1
        L.append(np.loadtxt(files[i])[:idx,2]) #L
        T.append(np.loadtxt(files[i])[:idx,6]) #T
        ages.append(np.loadtxt(files[i])[:idx,-1])
        m.append(np.loadtxt(files[i])[0,-2])
        if debug: print(len(ages[i]),len(T[i]),len(L[i]))
        max_t,min_t = np.max(T[i]),np.min(T[i])
        if max_t>max_T:
            max_T=max_t
        if min_t< min_T:
            min_T=min_t
        max_l,min_l = np.max(L[i]),np.min(L[i])
        if max_l>max_L:
            max_L=max_l
        if min_l< min_L:
            min_L=min_l
    myr=10**6
    chrome_list=[0.01*myr,0.1*myr,1*myr,10*myr,20*myr]#2*myr,3*myr,4*myr,5*myr,6*myr,7*myr,8*myr,9*myr,,50*myr]#,100*myr]

    isochrome=np.zeros((len(chrome_list),len(files),2))
    for j in range(len(chrome_list)):
        for i in range(len(ages)):
            v = (np.abs(ages[i] - chrome_list[j])).argmin()

            isochrome[j,i,0]=T[i][v]
            isochrome[j,i,1]=L[i][v]
    isochrome=np.sort(isochrome,axis=1)
    return L,T,m,max_T,max_L,min_T,min_L,isochrome,chrome_list

def plot_hrd(L,T,m,max_T,max_L,min_T,min_L,isochrome,chrome_list,lim_o_T,lim_o_L,T_low,L_low,T_up_mod,L_up_mod,lim_y_T,lim_y_L,temp,lum):
    
    fig=plt.figure(figsize=(9,7),constrained_layout=True)
    ax = fig.add_subplot(111)

    take_label=[2,4,6,8,12,22,26,31]

    for i in range(len(T)):

        if i in take_label:

            ax.plot(T[i],L[i],c='black',alpha=0.7)#,c=cm(1.0*i/len(files)),alpha=0.5)
            if i == 2:
                ax.annotate(str(m[i])+ r' $M_{sun}$',xy=(T[i][0],L[i][0]),xytext=(T[i][0]*1,L[i][0]*1.2))
            else:
                ax.annotate(str(m[i])+ r' $M_{sun}$',xy=(T[i][0],L[i][0]),xytext=(T[i][0]*1.0,L[i][0]*1.0))
        else:

            ax.plot(T[i],L[i],c='grey',alpha=0.7)#,c=cm(1.0*i/len(files)),alpha=0.5)
    ax.set_xlim([max_T*1.1,min_T*0.9])
    ax.set_ylim([min_L*0.9,max_L*1.1])

    ax.set_xlabel(r'$T_{eff} \ [K]$')
    ax.set_ylabel(r'$L \ [L_{sun}]$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    xtickslocs=[3*10**3,5*10**3,10**4,2*10**4]
    xtickslocs_names=[r'$3\cdot10^3$',r'$5\cdot10^3$',r'$10^4$',r'$2\cdot10^4$']
    #print(xtickslocs,xtickslocs_names)
    ax.set_xticks(ticks=xtickslocs)#
    ax.set_xticklabels(labels=xtickslocs_names)
    #ax.xaxis.set_major_locator(mticker.FixedLocator(xtickslocs))
    for v in range(len(chrome_list)):
        ax.plot(isochrome[v,:,0],isochrome[v,:,1],linestyle='--',alpha=0.7,color='grey')
        ax.annotate(f'{chrome_list[v]/10**6} Myr',xy=(isochrome[v,0,0],isochrome[v,0,1]),xytext=(isochrome[v,0,0]*0.97,isochrome[v,0,1]*0.97))
    
    #add boundries
    ax.plot(lim_o_T[11:-28],lim_o_L[11:-28],linestyle='solid',color='tab:red',lw=3)
    ax.plot(lim_y_T[11:-28],lim_y_L[11:-28],linestyle='solid',color='tab:red',lw=3)
    
    ax.plot(T_up_mod[146:-2],L_up_mod[146:-2],linestyle='solid',color='tab:red',lw=3)
    ax.plot(T_low[38:-225],L_low[38:-225],linestyle='solid',color='tab:red',lw=3,label='Limits')
    ax.scatter(temp,lum,marker='.',s=200.0,color='tab:blue',label='Star')
    ax.legend(loc='lower left')
    return fig