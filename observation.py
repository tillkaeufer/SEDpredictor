

from PyAstronomy import pyasl
import numpy as np

def load_observations(folder,data_file,dereddening_data=False):

    with open(f'{folder}/extinct.dat') as f:
        lines=f.readlines()
        for i in range(len(lines)):
            if 'E(B-V)' in lines[i]:
                e_bv=float(lines[i+1])
            if 'R_V' in lines[i]:
                R_V=float(lines[i+1])
    print(f'E(B-V)={e_bv}')
    print(f'R_V={R_V}')

    data_array=[]
    name_array=[]
    if data_file=='SEDobs.dat':
        with open(f'{folder}/SEDobs.dat') as f:
            lines=f.readlines()
            header=lines[1].split()
            if header[0]=='lam[mic]' and header[1]=='flux[Jy]' and header[2]=='sigma[Jy]' and header[3]=='rem':
                for i in range(2,len(lines)):
                    sp_line=lines[i].split()
                    if sp_line==[]:
                        print('Empty line')
                    elif sp_line[3]=='ok':
                        lam=float(sp_line[0])
                        flux=float(sp_line[1])
                        flux_sig=float(sp_line[2])
                        name=sp_line[4]
                        if flux_sig!=0:
                            data_array.append([lam,flux,flux_sig])
                            name_array.append(name)
            else:
                print('Different Header')
                print(header)
    if data_file=='SED_to_fit.dat':
        with open(f'{folder}/SED_to_fit.dat') as f:
            lines=f.readlines()
            for i in range(0,len(lines)):
                sp_line=lines[i].split()
                if sp_line==[]:
                    print('Empty line')
                else:
                    lam=float(sp_line[0])
                    flux=float(sp_line[1])
                    flux_sig=float(sp_line[2])
                    name=sp_line[3]
                    data_array.append([lam,flux,flux_sig])
                    name_array.append(name)

    
    data_array=np.asarray(data_array)
    print(f'Number of datapoints: {len(data_array)}')
    #convertion of units
    nu=2.99792458*10**14/data_array[:,0]
    data_array[:,1]=data_array[:,1]*10**(-23)*nu
    data_array[:,2]=data_array[:,2]*10**(-23)*nu
    if dereddening_data:
        # Deredden the spectrum
        fluxUnred = pyasl.unred(data_array[:,0]*10**4, data_array[:,1], ebv=e_bv, R_V=R_V)
    else:
        fluxUnred=data_array[:,1]
    return data_array[:,0],fluxUnred,data_array[:,2],name_array,e_bv,R_V # do we have to change sigma at the dereddening???

