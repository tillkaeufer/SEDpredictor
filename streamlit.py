from time import time
timing=False
fast_plotting=False
if timing:
    start_tot=time()
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import altair as alt
import pandas as pd

import matplotlib.pyplot as plt
# jupyters notebook Befehl zum direkten Anzeigen von Matplotlib Diagrammen
plt.rcParams['figure.figsize'] = (9, 6)
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
colormap={0:'red',1:'green'}
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True 
plt.rcParams['xtick.labelsize'] = plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['xtick.major.width'] = plt.rcParams['ytick.major.width'] = 1.6
plt.rcParams['font.size'] = 12


from scipy import interpolate

#loading observations
from PyAstronomy import pyasl
from observation import load_observations
from chi_computations import chi_window,chi_squared,chi_squared_reduced
from dict_file import slider_dict_single, log_dict_single, slider_dict_two, log_dict_two 
from plotting import density_plot, load_siesstracks, plot_hrd
#from para_transform import para_to_parameterin
#from rout import adjust_rout
from check_if_possible import load_all_star_lims,normalizing,find_nearest,in_or_out,check_if_in, new_radius_lims,check_if_valid_prediction

import streamlit as st
para_dict={}

#image icon
im = Image.open("icon.png")

                                
st.set_page_config(
    layout="wide",
    page_title='SEDpredictor',
    page_icon=im,
    initial_sidebar_state='expanded',
    menu_items={'Get help': 'https://tillkaeufer.github.io/sedpredictor',
                'Report a bug': 'mailto:till.kaeufer@oeaw.ac.at',
                'About': 'SED predictions of protoplanetary disks by a neural networks. This app is made by Till Kaeufer.'}
    

)

NN_name='single_46_opti' # what network to use
star_name='star_m-only_3' #what network for mass prediction of star
path_data='./data' #where is the downloaded data

input_file=False

knn_switch=False # using Knn
Knn_factor=2 #parameter to adjust the uncertainties

observe=False
residual=False
chi_on=False
chi_mode=  'DIANA' #'squared'   #'squared_reduced'#     
loglike=False
dereddeningdata=False
folder_observation='./Example_observation/DNTau'
file_name='SED_to_fit.dat' 
write_parameterin=False
calc_mdisk=False
    
if timing:
    start=time()
#load NN

def para_to_parameterin(input_string):
    para_dict={}
    lines=input_string.split('\n')
    for line in lines:
        print(line)
        split_line=line.split()
        value=float(split_line[0])
        print(value,split_line)
        parameter=split_line[1]
        para_dict[parameter]=value
    with open('TemplateParameter.in','r') as f_temp:
        lines=f_temp.readlines()
    new_lines=''
    for line in lines:
        if '!' in line:
            idx=line.index('!')
            #print(idx)
            if 'dist' in line:
                value=para_dict['Dist[pc]']
                new_line=f'%12.5e {line[idx:]}' %value
            else:
                for key in para_dict.keys():
                    if key in line:
                        if key=='Rout':
                            value=4*para_dict['Rtaper']
                            new_line=f'%12.5e {line[idx:]}' %value

                        else:
                            value=para_dict[key]
                            new_line=f'%12.5e {line[idx:]}' %value
                            break
                    else:
                        new_line=line
        else:
            if 'Mg0.7Fe0.3SiO3[s]' in line:
                idx=line.index('Mg')
                new_line=f'%12.5e {line[idx:]}' %value
            if 'amC-Zubko[s]' in line:
                idx=line.index('amC-Zubko[s]')
                new_line=f'%12.5e {line[idx:]}' %value
            else:
                new_line=line
        #print(new_line)
        new_lines=new_lines+new_line     
    return new_lines


@st.cache(suppress_st_warning=True,allow_output_mutation=True)  
def load_data(path_data,NN_name, delete_derived_paras=True,two_zone=False):
    scaler=joblib.load(f'{path_data}/scaler/{NN_name}_para_scaler.save')
    y_scaler=joblib.load(f'{path_data}/scaler/{NN_name}_sed_scaler.save')
    model_saved=load_model(f'{path_data}/NeuralNets/{NN_name}.h5')
    if not two_zone:
        header_start=np.load(f'{path_data}/header.npy')
        header_start=np.concatenate((header_start,['incl']),axis=0)
    else:
        header_start=np.load(f'{path_data}/header_two_zone.npy')
        

    if delete_derived_paras:
        list_derived=['Mstar', 'amC-Zubko[s]', 'Rout']
        new_header_1=[]
        i_list=[]
        for i in range(len(header_start)):
            if header_start[i] not in list_derived:
                new_header_1.append(header_start[i])
                i_list.append(i)
        header=np.asarray(new_header_1)
    
    txt=str()
    with open(f'{path_data}/wavelength.out','r') as f:
        lines=f.readlines()
    for line in lines[1:]:
        
        txt=txt+line.strip()+' '  
    txt=txt[1:-2].split()
    wavelength=np.array(txt,'float64')
    return  scaler, y_scaler, model_saved, header, wavelength


#scaler,y_scaler, model_saved,header, wavelength=load_data(path_data=path_data,NN_name=NN_name)

@st.cache(suppress_st_warning=True,allow_output_mutation=True)  
def load_nn_and_scaler_star(path_data,star_name):    
    model_star=load_model(f'{path_data}/StarNets/{star_name}.h5')
    input_scaler=joblib.load(f'{path_data}/scaler/{star_name}_input_scaler.save')
    output_scaler=joblib.load(f'{path_data}/scaler/{star_name}_output_scaler.save')
    return model_star,input_scaler,output_scaler
model_star,input_scaler,output_scaler=load_nn_and_scaler_star(path_data,star_name)

def calculate_mstar(Teff,Lstar):
    trans_star=input_scaler.transform(np.expand_dims([Teff,Lstar],axis=0))
    pred_star=model_star(trans_star)
    log_mass=output_scaler.inverse_transform(pred_star)[0,0]
    
    return log_mass

@st.cache(suppress_st_warning=True,allow_output_mutation=True)  
def angle_to_mcfost_val(angle):
    rad=angle*np.pi/180
    cosx=np.cos(rad)
    mcfost_incl=9-(cosx-0.05)/0.1    
    return mcfost_incl


def transform_parameter(paraname,val,header,scaler):
    dummy=np.zeros((1,len(header)))
##    if name in name_log:
#        val=np.log10(val)
    if paraname=='incl':
        val=float(val)
        val=angle_to_mcfost_val(val)
    if paraname in header:
        pos=np.where(header==paraname)[0][0]
        dummy[0,pos]=val
        val=scaler.transform(dummy)[0,pos]
        return val,pos

@st.cache(suppress_st_warning=True,allow_output_mutation=True)  
def change_dist(dist,data):
    new_data=data*(100/dist)**2
    return new_data

@st.cache(suppress_st_warning=True,allow_output_mutation=True)      
def reddening( lam,flux, e_bv, R_V):
        # lam in mu m 
        fluxRed = pyasl.unred(lam*10**4, flux, ebv=-e_bv, R_V=R_V)
        return fluxRed

@st.cache(suppress_st_warning=True,allow_output_mutation=True)  
def spline(lam,nuflux,new_lam):

    #interpolation on a double logarithmic scale
    s=interpolate.InterpolatedUnivariateSpline(np.log10(lam),np.log10(nuflux))
    interp=10**(s(np.log10(new_lam)))
#    return interp #returning the nex SED
    return interp #for visualisation 
if timing:
    end=time()
    loading_time=end-start
    
    
L,T,m,max_T,max_L,min_T,min_L,isochrome,chrome_list=load_siesstracks()

lim_o_T,lim_o_L,T_low,L_low,T_up_mod,L_up_mod,lim_y_T,lim_y_L=load_all_star_lims()
    
def main():
    st.title('SEDs of protoplanetary disk models')
    st.subheader('by [Till Kaeufer](https://tillkaeufer.github.io/)')

    st.markdown(
    """
    
    <br><br/>
    This tool predictions SEDs of protoplanetary disk using neural networks.
    <br/>
    If this tool is useful to your work, please cite (Kaeufer et al., submitted)
    """
    , unsafe_allow_html=True)
#    should_tell_me_more = st.button('Tell me more')
    should_tell_me_more =False
    st.warning('This website is just in the making, so be careful.')
    
    st.markdown(
    """
    
    The [info page](https://tillkaeufer.github.io/sedpredictor) provides information on how to use this tool and examples for input files.
    
    """
    , unsafe_allow_html=True)
    
    if should_tell_me_more:
        print('Not yet working')
        
        
    else:
        questions = {
        'Complexity': ['Single zone', 'Two-zone'], #, 'Two-zone' 
     #   'Two-zone flavor': ['discontinues', 'continues','smooth'],
        'Input version':['Slider','Text only']}
  
        st.sidebar.markdown('# Create your disk')


        for question, answers in questions.items():
            valid_to_select = True
            #if question!='Two-zone flavor' or complexity=='Two-zone':

            st.sidebar.markdown("### " + question.replace('-', ' ').capitalize() + '?')
            if valid_to_select:
                        if question=='Complexity':
                            complexity = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')
                        else:
                            if question=='Input version':
                                input_version = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')

                            else:
                                selected_answer = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')
        
        #loading the networks and scalers for single or two-zone models
        
        if complexity=='Two-zone':
            two_zone=True
            
            slider_dict=slider_dict_two
            log_dict=log_dict_two

            NN_name='two_38_small_batch' # what network to use
        else:
            two_zone=False
            slider_dict=slider_dict_single
            log_dict=log_dict_single
            
            NN_name='single_46_opti' # what network to use
        scaler,y_scaler, model_saved,header, wavelength=load_data(path_data=path_data,NN_name=NN_name,two_zone=two_zone)
       
            
                                  

        st.markdown('---')
        st.markdown("""### Different settings.""")
        fast_plotting=st.checkbox('Simplified faster plotting',value=False)
        observe=st.checkbox('Compare to observation',value=False)
        if observe:
            residual=st.checkbox('Plot the residual',value=True)
            obj=st.selectbox('Select an object', ['49Cet','AATau','ABAur','BPTau', 'CITau', 'CYTau','CQTau',
                                        'DNTau','DFTau','DMTau','DOTau','FTTau','GMAur','HD100546',
                                        'HD135344B','HD142666','HD163296','HD169142','HD95881','HD97048',
                                        'LkCa15','MWC480','PDS66','RECX15','RULup','RYLup','TWCha',
                                        'TWHya','UScoJ1604-2130','V1149Sco'])
            folder_observation='./Example_observation/'+str(obj)
            isobs_file=st.file_uploader('Or upload your own observations',type=['txt','dat'])
            if isobs_file is not None:
                lines=isobs_file.getvalue().splitlines()
                data_array=[]
                for i in range(0,len(lines)):
                    sp_line=lines[i].split()
                    if sp_line==[]:
                        print('Empty line')
                    else:
                        lam=float(sp_line[0])
                        flux=float(sp_line[1])
                        flux_sig=float(sp_line[2])
                        data_array.append([lam,flux,flux_sig])
                        

        
                data_array=np.asarray(data_array)
                #print(f'Number of datapoints: {len(data_array)}')
                #convertion of units
                nu=2.99792458*10**14/data_array[:,0]
                data_array[:,1]=data_array[:,1]*10**(-23)*nu
                data_array[:,2]=data_array[:,2]*10**(-23)*nu
        
                fluxUnred=data_array[:,1]
                lam_obs,flux_obs,sig_obs= data_array[:,0],fluxUnred,data_array[:,2] # do we have to change sigma at the dereddening???
            else:
                file_name='SED_to_fit.dat' 
                lam_obs,flux_obs,sig_obs,filer_names,e_bvstart,R_Vstart=load_observations(folder_observation,file_name,dereddening_data=False)
                
            chi_on=st.checkbox('Write the Chi-value',value=False)
            #st.write(chi_on)
            if chi_on:
                chi_mode=st.selectbox('Chi-mode', ['squared','squared_reduced','DIANA'])

        else:
            residual=False
            chi_on=False


            
        e_bvstart=0.1
        R_Vstart=3.1
        dist_start=100
        
        use_parafile=st.checkbox('Use your own parameter file',value=False)
 
        for key in log_dict:
            slider_dict[key]['scale']=log_dict[key]
            
        for key in slider_dict:
            if slider_dict[key]['scale']=='log':
                if 'log' not in slider_dict[key]['label']:
                    #print(slider_dict[key]['label']+': fine')
                #else:
                    slider_dict[key]['label']='$log('+slider_dict[key]['label'][1:-1]+')$'
                    low=slider_dict[key]['lims'][0]
                    high=slider_dict[key]['lims'][1]
                    slider_dict[key]['lims']=[np.log10(low),np.log10(high)]            
        dist_start=100
        if use_parafile:
            ispara_file=st.file_uploader('Parameter file',type='txt')
            if ispara_file is not None:
                lines=ispara_file.getvalue().splitlines()
                #here I have to work on!!
                for line in lines:
#                    st.write(line)
                    split_line=line.split()
                    value=float(split_line[0])
                    parameter=split_line[1].decode('ascii')
                   # print(parameter,value)
                 
                    if parameter in slider_dict.keys():
                        if slider_dict[parameter]['scale']=='log':
                            value=np.log10(value)
                        slider_dict[parameter]['x0']=value
                        
                    else:
                        if parameter=='Dist[pc]':
                            dist_start=value
                        if parameter=='E(B-V)':
                            e_bvstart=value
                            
                        if parameter=='R(V)':
                            R_Vstart=value  
        
        
        create_density_plot=st.checkbox('Plot 2D density',value=True)
        
        plot_column_dens=st.checkbox(label='Plot column density',value=False)
        
        
        #print(slider_dict)
        if timing:
            start=time()

        
        st.sidebar.markdown("## " + 'Parameters'.capitalize())


        #fig,ax = plt.figure(figsize=(9,9))
        if observe and residual:
            fig, fig_ax = plt.subplots(figsize=(12,13.5))
            gs = fig.add_gridspec(3, 1)
            ax = fig.add_subplot(gs[0:2, :])
        else:
            fig, ax = plt.subplots(figsize=(12,9))

        st.sidebar.markdown('---')


        #distance
        st.sidebar.write('Distance [pc]')
        dist_start=float(st.sidebar.text_input(label='',value=dist_start,key='dist'))
        

        st.sidebar.markdown('---')

        st.sidebar.write('Reddening')
        
        st.sidebar.write('$E_{BV}$')
        e_bvstart=float(st.sidebar.text_input(label='',value=e_bvstart,key='ebv'))
        
    
        st.sidebar.write('$R_V$')
        R_Vstart=float(st.sidebar.text_input(label='',value=R_Vstart,key='rv'))
        
    
        st.sidebar.markdown('---')


        plt.subplots_adjust(left=0.2, bottom=0.41, top=0.95)

        features=np.zeros((1,len(header)))
        if timing:
            start2=time()
        c=0
        for key in header:
            #print(key)name=slider_dict[key]['label']
            mini,maxi=slider_dict[key]['lims']
            paraname=slider_dict[key]['label']
            try:
                value=float(slider_dict[key]['x0'])
            except:
                value=float((maxi+mini)/2)
            if input_version=='Slider':
                col1,col2=st.sidebar.columns([1,1])
                with col1:
                    
                    st.sidebar.write(paraname)
                with col2:
                    middle=st.sidebar.slider('',min_value=float(mini),max_value=float(maxi),value=value)
                if 'log' in paraname:
                    
                    round_n=6 
                    while np.round(10**middle,round_n)==0:
                        round_n+=3
                       
                    middle=st.sidebar.text_input(label='',value=float(np.round(10**middle,round_n)),key=c)#
                    middle=np.log10(float(middle))
                    c+=1       
    
                else:
                    middle=st.sidebar.text_input(label='',value=float(middle),key=c) #name
                    c+=1
                #print(c)
                st.sidebar.markdown('---')
            else:
                if 'log' in paraname:
                    name_withoutlog='$'+paraname[10:-2]+'$'
                    
                    st.sidebar.write(name_withoutlog)
                else:
                    st.sidebar.write(paraname)
                if 'log' in paraname:
                    
                    middle=st.sidebar.text_input(label='',value=float(np.round(10**value,4)),key=c)
                    middle=np.log10(float(middle))
                else:
                    middle=st.sidebar.text_input(label='',value=float(value),key=c)
                c+=1
                
            if timing:
                end=time()
                sidebar_time=end-start

            if calc_mdisk:
                if key=='Mdisk':
                    init_mass=10**middle
            
            val_trans, pos=transform_parameter(key,middle,header,scaler)
            features[0,pos]=val_trans
            if key=='amC-Zubko[s]':
                middle_sio=0.75-middle
                val_trans_sio, pos_sio=transform_parameter('Mg0.7Fe0.3SiO3[s]',middle_sio)
                features[0,pos_sio]=val_trans_sio
             
            #print(val_trans)  
        #print(features)
        
        
        plot_hrd_check=st.checkbox('Plot HRD',value=False)
        
        #slider to adjust the x and y axis
        lam_min_start,lam_max_start=np.min(wavelength),10**3
        flux_min_start,flux_max_start=10**-12,10**-7
        
        adjust_limits=False
        adjust_limits=st.checkbox('Adjust plot limits',value=False)

        
        if observe:
            lam_min_start,lam_max_start=np.min(lam_obs)*0.9,np.max(lam_obs)*1.1
            flux_min_start,flux_max_start=10**(int(np.min(np.log10(flux_obs))-1)), 10**(int(np.max(np.log10(flux_obs))+1))
        if adjust_limits:
            lam_min=float(st.text_input(label='Minimal wavelength',value=lam_min_start,key='lam_min'))
            lam_max=float(st.text_input(label='Maximal wavelength',value=lam_max_start,key='lam_max'))
            flux_min=float(st.text_input(label='Minimal SED value',value=flux_min_start,key='flux_min'))
            flux_max=float(st.text_input(label='Maximal SED value',value=flux_max_start,key='flux_max'))
            if create_density_plot:
                zmax=float(st.text_input(label='Density plot: maximum height (0-2)',value=1.0,key='dens_zmax'))
                
                cmin=float(st.text_input(label='Density plot: minimum density',value=4,key='dens_cmin'))
            
        else:
            lam_min=lam_min_start
            lam_max=lam_max_start
            flux_min=flux_min_start
            flux_max=flux_max_start
            zmax=1.0
            cmin=4.0
               
        st.markdown('---')
        
        #create dictionary of the input parametermeters for checking if possible and for plotting
        exp_features=scaler.inverse_transform(features)
        print(np.shape(exp_features),np.shape(header))
        dict_para={}
        for i in range(len(header)):
            if log_dict[header[i]]=='log':
                exp_features[:,i]=10**(exp_features[:,i])
            dict_para[header[i]]=exp_features[:,i][0]
        print(dict_para)
        
        
        #checking if possible
        
        if not check_if_in(p0=np.log10(dict_para['Teff']),p1=np.log10(dict_para['Lstar'])):
            st.error('Star not in the range the NN was trained on!!', icon="🚨")
            plot_hrd_check=True

        
        error_string=check_if_valid_prediction(dict_para,two_zone=two_zone)
        #TODO in check if valid, general limits of the sample
        if error_string!='':
           # print(error_string)
            st.error(error_string, icon="🚨")
        
        #calc mass of star
        mstar=10**(calculate_mstar(np.log10(dict_para['Teff']),np.log10(dict_para['Lstar'])))
        
        st.write('Predicted stellar Mass: '+str(np.round(mstar,2))+'$\\rm M_{sun}$')
        #check is mass of disk is allowed
        if dict_para['Mdisk']<mstar*10**(-5) or dict_para['Mdisk']>mstar:    
            st.error('Mass of the disk is not between stellar mass and 10^(-5) times that value', icon="🚨")
        
        if timing:
            end2=time()
            para_time=end2-start2
            start=time()
        data=10**(y_scaler.inverse_transform(model_saved(features)))[0]
        data=change_dist(dist_start,data)
        if not dereddeningdata:
            #reddening
            #print(e_bvstart,R_Vstart)
            data=reddening(wavelength,data,e_bvstart,R_Vstart)
        if timing:
            end=time()
            pred_time=end-start
        str_title=''
        if calc_mdisk:
            m_disk=calc_mass(data)
            rat_mass=m_disk/init_mass
            str_title_new=str_title+'\n '+r'$M_{disk,calc}: %8.2e ,  M_{calc}/M_{model}: %8.2e $' %(m_disk,rat_mass)
            ax.set_title(str_title_new,fontsize=12)
        if timing:
            start=time()
        t=wavelength
        s = data
        

        
        if fast_plotting:
            df_array=np.concatenate((np.expand_dims(wavelength,axis=0),np.expand_dims(data,axis=0)),axis=0).T
            
            df=pd.DataFrame(df_array,columns=['lambda','SED'])
            #print(df.shape)
            df=df[df['lambda']<= lam_max]
            
            df=df[df['lambda']>= lam_min]
            alt_chart=alt.Chart(df).mark_point().encode(
                x=alt.X('lambda',scale=alt.Scale(domain=[lam_min,lam_max],type="log")),y=alt.Y('SED',scale=alt.Scale(domain=[flux_min,flux_max],type="log")),
                    #tooltip=[alt.Tooltip('lam', title=r'$ \nu F_\nu [erg/cm^2/s]$'),
                   # alt.Tooltip("SED", title="Price (USD)"),]
            ).properties(
        width=800,
        height=500
    )

            st.altair_chart(alt_chart)
        
        else:
            
            
            l, = ax.plot(t, s,marker='+',linestyle='none')

            ax.axis([lam_min,lam_max ,flux_min,flux_max])



            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel(r'$ \nu F_\nu [erg/cm^2/s]$',fontsize=13)
            if not residual:
                ax.set_xlabel(r'$ \lambda \, [\mu m]$',fontsize=13)


            if observe:
                ax.errorbar(lam_obs,flux_obs,sig_obs,linestyle='None',marker='+')
                if residual:
                    ax_res = fig.add_subplot(gs[2, :])
                    ax_res.axhline(0,label='Perfect Prediction',color='tab:blue')
                    interp_flux=spline(wavelength,data,lam_obs)
                    diff=np.log10(interp_flux)-np.log10(flux_obs)
                    res,=ax_res.plot(lam_obs,diff,marker='+',linestyle='None')
                    ax_res.set_ylabel(r'$ log(F_{model})-log(F_{obs})$',fontsize=13)
                    ax_res.yaxis.tick_right()
                    ax_res.yaxis.set_label_position("right")
                    ax_res.set_xlabel(r'$ \lambda \, [\mu m]$',fontsize=13)
                    ax_res.set_xscale('log')
                    ax_res.set_ylim([-1,1])
                    if observe:
                        ax_res.set_xlim([np.min(lam_obs)*0.9, np.max(lam_obs)*1.1])
                    else:
                        ax_res.set_xlim([np.min(wavelength),10**3])
                    fig_ax.set_xticks([])
                    fig_ax.set_yticks([])
                    ax.set_xticklabels([])
                    max_diff=1.2*max(abs(np.min(diff)),abs(np.max(diff)))
                    ax_res.set_ylim([-max_diff,max_diff])
                    if chi_on:
                        if chi_mode=='DIANA':
                            chi=chi_window(interp_flux,flux_obs,sig_obs,lam_obs)
                            st.write(r'DIANA $\chi$ = %8.4f' %chi,fontsize=12)
                        elif chi_mode=='squared':

                            chi=chi_squared(interp_flux,flux_obs,sig_obs,lam_obs)
                            if loglike:
                                log_likelihood=log_like(interp_flux,flux_obs,sig_obs,lam_obs)
                                st.write(r'$\chi^2$ = %8.4f, loglike = %12.4e' %(chi,log_likelihood),fontsize=12)
                            else:
                                st.write(r'$\chi^2$ = %8.4f' %chi,fontsize=12)
                        if chi_mode=='squared_reduced':
                            chi=chi_squared_reduced(interp_flux,flux_obs,sig_obs,lam_obs)
                            st.write(r' reduced $\chi^2$ = %8.4f' %chi,fontsize=12)


            if timing:
                end=time()
                plot_time=end-start
                start=time()
            st.pyplot(fig)#,clear_figure=True)
            
            #plot density structure
            if create_density_plot:
                den=density_plot(dict_para)
                den.get_density_structure()
                fig_dens=den.plot_fig_density(zmax=zmax,cmin=cmin)
                st.pyplot(fig_dens)
                if plot_column_dens:
                    fig_col=den.plot_fig_columndensity()
                    st.pyplot(fig_col)
             
                    
             
            #plot hrd
            
            if plot_hrd_check:
                fig_hrd=plot_hrd(L,T,m,max_T,max_L,min_T,min_L,isochrome,chrome_list,lim_o_T,lim_o_L,T_low,L_low,T_up_mod,L_up_mod,lim_y_T,lim_y_L,dict_para['Teff'],dict_para['Lstar'])
                st.pyplot(fig_hrd)
             
            
            
            if timing:
                end=time()
                loadplot_time=end-start
                tot_time=end-start_tot
            if timing:
                st.write('Time:')
                st.write('Time to load data: '+str(np.round(loading_time,2))+'s')
                st.write('Time to load Sidebar: '+str(np.round(sidebar_time,2))+'s')
                st.write('Time to transform parameters: '+str(np.round(para_time,2))+'s')
                st.write('Time to predict SED: '+str(np.round(pred_time,4))+'s')
                st.write('Time to plot SED: '+str(np.round(plot_time,4))+'s')
                st.write('Time to load Plot: '+str(np.round(loadplot_time,2))+'s')
                st.write(' ')
                st.write('Total time: '+str(np.round(tot_time,2))+'s')
            #st.success('Successfully predicted the SED')
            st.markdown('----')
            exp_para=st.checkbox(label='Export Parameters',value=False)
            if exp_para:
                exp_features=scaler.inverse_transform(features)
                print(np.shape(exp_features),np.shape(header))
                para_string=''
                for i in range(len(header)):
                    if log_dict[header[i]]=='log':
                        exp_features[:,i]=10**(exp_features[:,i])
                    val=exp_features[0,i]
                    individual_string='%12.5e ' %val
                    individual_string= individual_string +' ' + header[i]+ '\n'
                    para_string=para_string+individual_string
                para_string=para_string+ str(e_bvstart)+' E(B-V) \n'
                para_string=para_string+ str(R_Vstart)+' R(V) \n'
                para_string=para_string+ str(dist_start)+' Dist[pc]'
                
                print(para_string)
                st.download_button('Download Parameter file [can be used for upload]', para_string,'Input.txt')
                
                parameterin_string=para_to_parameterin(para_string)
                st.download_button('Download Parameter file [in ProDiMo format]', parameterin_string,'Parameter.in')


                #print('export model parameters')
            exp_sed=st.checkbox(label='Export model SED',value=False)
            if exp_sed:
                #print('export SED')
                un_fac=0.05
                sed_string='lam[mic] nuF[erg/cm^2/s] sigma \n'
                for i in range(len(wavelength)):
                    lam_1,flux_1,sig_1=wavelength[i],data[i],data[i]*un_fac
                    sed_string=sed_string+ '%12.6e %12.6e %12.6e \n'%(lam_1,flux_1,sig_1)
                
                st.download_button('Download SED file', sed_string,'SED_model.txt')


        

    
#    st.latex()
if __name__ == '__main__':

#    logging.basicConfig(level=logging.CRITICAL)

    main()