import numpy as np

def load_all_star_lims():
    data_path='./data/star_data'
    lim_o_T=np.load(f'{data_path}/lim_o_T.npy')
    lim_o_L=np.load(f'{data_path}/lim_o_L.npy')
    T_low=np.load(f'{data_path}/T_low.npy')
    L_low=np.load(f'{data_path}/L_low.npy')
    T_up_mod=np.load(f'{data_path}/T_up_mod.npy')
    L_up_mod=np.load(f'{data_path}/L_up_mod.npy')
    lim_y_T=np.load(f'{data_path}/lim_y_T.npy')
    lim_y_L=np.load(f'{data_path}/lim_y_L.npy')
    return lim_o_T,lim_o_L,T_low,L_low,T_up_mod,L_up_mod,lim_y_T,lim_y_L

lim_o_T,lim_o_L,T_low,L_low,T_up_mod,L_up_mod,lim_y_T,lim_y_L=load_all_star_lims()

def normalizing(ax,ay,point):
    minx=np.min(ax)
    ptpx=np.ptp(ax)
    ax=(ax-minx)/ptpx
    
    miny=np.min(ay)
    ptpy=np.ptp(ay)
    ay=(ay-miny)/ptpy
    
    px=(point[0]-minx)/ptpx
    py=(point[1]-miny)/ptpy
    point=(px,py)
    return(ax,ay,point)

def find_nearest(arrayx,arrayy, point): #point in log
    ax = np.asarray(np.log10(arrayx))
    ay= np.asarray(np.log10(arrayy))
    ax,ay,point=normalizing(ax,ay,point)
    dist=[((point[0]-ax[i])**2+(point[1]-ay[i])**2) for i in range(len(ax))]
    idx=np.argmin(dist)
    return idx,arrayx[idx],arrayy[idx]

def in_or_out(arrayx,arrayy,point):
    idx,p1,p2=find_nearest(arrayx,arrayy,point)
    if arrayx[0]==lim_y_T[0] or arrayx[0]==lim_o_T[0]:
        try:
            v1=(np.log10(arrayx[idx+1])-np.log10(arrayx[idx]),np.log10(arrayy[idx+1])-np.log10(arrayy[idx]))
            v2=(np.log10(arrayx[idx+1])-point[0],np.log10(arrayy[idx+1])-point[1])
            xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product
            xp2=+1
        except:
            xp=-1
            v1_2=(np.log10(arrayx[idx-1])-np.log10(arrayx[idx]),np.log10(arrayy[idx-1])-np.log10(arrayy[idx]))
            v2_2=(np.log10(arrayx[idx-1])-point[0],np.log10(arrayy[idx-1])-point[1])
            xp2 = v1_2[0]*v2_2[1] - v1_2[1]*v2_2[0]  # Cross product
    else:
        v1=(np.log10(arrayx[idx+1])-np.log10(arrayx[idx]),np.log10(arrayy[idx+1])-np.log10(arrayy[idx]))
        v2=(np.log10(arrayx[idx+1])-point[0],np.log10(arrayy[idx+1])-point[1])
        xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

        v1_2=(np.log10(arrayx[idx-1])-np.log10(arrayx[idx]),np.log10(arrayy[idx-1])-np.log10(arrayy[idx]))
        v2_2=(np.log10(arrayx[idx-1])-point[0],np.log10(arrayy[idx-1])-point[1])
        xp2 = v1_2[0]*v2_2[1] - v1_2[1]*v2_2[0]  # Cross product
    
    #for visualization:
    #xp2=1
    if xp > 0 or xp2<0:
        return True
    else:
        return False
    
def check_if_in(p0,p1):
    in_hrd=False
    if not in_or_out(lim_o_T,lim_o_L,(p0,p1)):
        if not in_or_out(T_low,L_low,(p0,p1)):
            if in_or_out(T_up_mod,L_up_mod,(p0,p1)):
                if in_or_out(lim_y_T,lim_y_L,(p0,p1)):
                    in_hrd=True
    return in_hrd

def new_radius_lims(teff,lstar):
    '''
    We use the sublimation temperature limits that we have in the training grid to limit the possible rins/ r2ins.
    Since all stellar parameters are fixed we can simply calculte the new limits before the MultiNest run and adjust the prior
    '''
    
    hplanck = 6.62607554E-27        # Planck's constant
    bk      = 1.38065812E-16        # Boltzmann's constant
    cl      = 2.99792458E+10        # speed of light
    pi=np.pi
    cPl1    = 2.0*hplanck*cl**2
    cPl2    = hplanck*cl/bk
    sig_SB  = cPl1/cPl2**4*pi**5/15.0
    Lsun    = 3.8260E+33            # solar luminosity
    Msun    = 1.9889225E+33         # solar mass
    Rsun    = 6.9599000E+10         # solar radius
    Rsun_to_AU=214.939
    tsub_max=1677
    tsub_min=100
    
    #calculating_rstar
    
    rstar = np.sqrt(lstar*Lsun/(4.0*pi*sig_SB*teff**4)) /Rsun /Rsun_to_AU
    
    
    rmin=np.sqrt(1-0.2)* (teff/tsub_max)**2 * rstar
    rmax=np.sqrt(1-0.2)* (teff/tsub_min)**2 * rstar
    
    return rmin, rmax


def check_if_valid_prediction(para_dict,two_zone=False):
    #dust size checks and pahs
    error_string=''
    if para_dict['amin']>=para_dict['amax']:    
        error_string=error_string+'Minimum dust size is larger than maximum dust size\n'
    if two_zone:
        if para_dict['amin']>=para_dict['a2max']:    
            error_string=error_string+'Minimum dust size is larger than maximum dust size of inner zone\n'
        if para_dict['a2max']>para_dict['amax'] or para_dict['a2max']<para_dict['amax']*10**(-4):    
            error_string=error_string+'Maximum dust size of inner zone must be between maximum dust size of outer zone and 10^(-4) that value\n'
        if para_dict['f2PAH']>para_dict['fPAH'] or para_dict['f2PAH']<para_dict['fPAH']*10**(-3.5):    
            error_string=error_string+'PAH amount of inner zone must be between PAH amount of outer zone and 10^(-3.5) that value\n'
        
    #radial structure

    #if para_dict['Rin']>=para_dict['Rout']:    
    #    error_string=error_string+'Negative radial extent\n'
    rmin,rmax=new_radius_lims(para_dict['Teff'],para_dict['Lstar'])
    if two_zone:
        if para_dict['R2in']>rmax or para_dict['R2in']<rmin:    
            error_string=error_string+'Inner most radius is not sampled (too close or to far for this particular star)\n'
            error_string=error_string+'smallest possible radius: '+str(rmin)+' largest possible radius: '+str(rmax)+'\n'
    else:
        if para_dict['Rin']>rmax or para_dict['Rin']<rmin:    
            error_string=error_string+'Inner most radius is not sampled (too close or to far for this particular star)\n'
            error_string=error_string+'smallest possible radius: '+str(rmin)+' largest possible radius: '+str(rmax)+'\n'
            
    if two_zone:
        if para_dict['R2in']>=para_dict['R2out']:    
            error_string=error_string+'Inner zone has negative radial extent\n'
        if para_dict['R2out']>para_dict['Rin']:    
            error_string=error_string+'Overlapping zones\n'
        if para_dict['R2out']<para_dict['Rin']*10**(-3):    
            error_string=error_string+'Too large gap (R2out < Rin *10^-3)\n'
        
    # mass of disk
    
    if two_zone:
        if para_dict['M2disk']>para_dict['Mdisk']*10**(-0.7) or para_dict['M2disk']<para_dict['Mdisk']*10**(-7):    
            error_string=error_string+'Mass of inner zone must be between 10^(-0.7)* mass of outer zone and 10^(-7)*mass of outer zone\n'
    #mass of disk to mass of star
    
    
    return error_string
    