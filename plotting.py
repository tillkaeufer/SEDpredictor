# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:14:35 2022

@author: tfk1
"""
import numpy as np
import matplotlib.pyplot as plt




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
        self.surface_den=np.zeros(N_rad)
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
            print('Adjusting Rout')
            self.adjust_rout(self)
        
        #setup the radial and vertical points 
        min_rad=self.parameter_dict['Rin']
        max_rad=self.parameter_dict['Rout']
        
        self.radii=np.logspace(np.log10(min_rad),np.log10(max_rad),num=self.N_rad+1)
        self.z_list=np.arange(0,2,2/(self.N_ver+1))
        #print(np.shape(self.z_list))
        
        self.get_radial_structure(self)
        self.vertical_structure(self)
    
    def get_radial_structure(self,debug=False):
        new_rtmp=True
        debug=False
        '''
        get relevant parameters from dict
        '''
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
        debug=False
        r_0=100
        z_list=self.z_list
        H0=self.parameter_dict['MCFOST_H0']
        r_list=self.radii
        beta=self.parameter_dict['MCFOST_BETA']
        surf=self.surface_den
        
        AU=self.const['AU']
        if debug:
            print('Shape surf',np.shape(surf))
        for j in range(len(r_list)):
            r=r_list[j]
            N_surf=surf[j]
            if debug:
                print('R')
                print(r)
            density_list=np.zeros_like(z_list)
            Hr=H0*(r/r_0)**(beta)
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
                density_list[i] = max(0.0,nHtot)
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