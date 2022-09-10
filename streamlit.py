from time import time
timing=False
fast_plotting=True
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
#from para_transform import para_to_parameterin
#from rout import adjust_rout

import streamlit as st

#image icon
im = Image.open("icon.png")

                                
st.set_page_config(
    layout="wide",
    page_title='SED predictor',
    page_icon=im

)


name='single_45_rinlog' # what network to use
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

@st.cache(suppress_st_warning=True,allow_output_mutation=True)  
def load_data(path_data,name, delete_derived_paras=True):
    scaler=joblib.load(f'{path_data}/scaler/{name}_para_scaler.save')
    y_scaler=joblib.load(f'{path_data}/scaler/{name}_sed_scaler.save')
    model_saved=load_model(f'{path_data}/NeuralNets/{name}.h5')
        
    header_start=np.load(f'{path_data}/header.npy')
    header_start=np.concatenate((header_start,['incl']),axis=0)

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

scaler,y_scaler, model_saved,header, wavelength=load_data(path_data=path_data,name=name)


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

def angle_to_mcfost_val(angle):
    rad=angle*np.pi/180
    cosx=np.cos(rad)
    mcfost_incl=10-(cosx-0.05)/0.1    
    return mcfost_incl


def transform_parameter(name,val):
    dummy=np.zeros((1,len(header)))
##    if name in name_log:
#        val=np.log10(val)
    if name=='incl':
        val=float(val)
        val=angle_to_mcfost_val(val)
    if name in header:
        pos=np.where(header==name)[0][0]
        dummy[0,pos]=val
        val=scaler.transform(dummy)[0,pos]
        return val,pos

def change_dist(dist,data):
    new_data=data*(100/dist)**2
    return new_data
    
def reddening( lam,flux, e_bv, R_V):
        # lam in mu m 
        fluxRed = pyasl.unred(lam*10**4, flux, ebv=-e_bv, R_V=R_V)
        return fluxRed

def spline(lam,nuflux,new_lam):

    #interpolation on a double logarithmic scale
    s=interpolate.InterpolatedUnivariateSpline(np.log10(lam),np.log10(nuflux))
    interp=10**(s(np.log10(new_lam)))
#    return interp #returning the nex SED
    return interp #for visualisation 
if timing:
    end=time()
    loading_time=end-start
    
def main():
    st.title('SEDs of protoplanetary disk models')
    st.subheader('by [Till Kaeufer](https://tillkaeufer.github.io/)')

    st.markdown(
    """
    
    <br><br/>
    This tool predictions SEDs of protoplanetary disk using neural networks.
    <br/>
    If this tool is useful to your work, please cite (Kaeufer et al., in prep)
    """
    , unsafe_allow_html=True)
    should_tell_me_more = st.button('Tell me more')
    
    st.warning('This website is just in the making, so be careful.')
    
    if should_tell_me_more:
        tell_me_more()
        st.markdown('---')
        
        
    else:
        st.markdown('---')
        st.markdown("""### Different settings.""")
        fast_plotting=st.checkbox('Simplified faster plotting',value=False)
        observe=st.checkbox('Compare to observation',value=False)
        if observe:
            residual=st.checkbox('Plot the residual',value=True)
            obj=st.selectbox('Object', ['49Cet','AATau','ABAur','BPTau', 'CITau', 'CYTau','CQTau',
                                        'DNTau','DFTau','DMTau','DOTau','FTTau','GMAur','HD100546',
                                        'HD135344B','HD142666','HD163296','HD169142','HD95881','HD97048',
                                        'LkCa15','MWC480','PDS66','RECX15','RULup','RYLup','TWCha',
                                        'TWHya','UScoJ1604-2130','V1149Sco'])
            folder_observation='./Example_observation/'+str(obj)
            chi_on=st.checkbox('Write the Chi-value',value=False)
            #st.write(chi_on)
            if chi_on:
                chi_mode=st.selectbox('Chi-mode', ['squared','squared_reduced','DIANA'])

        else:
            residual=False
            chi_on=False

        file_name='SED_to_fit.dat' 
        if observe:
            lam_obs,flux_obs,sig_obs,filer_names,e_bvstart,R_Vstart=load_observations(folder_observation,file_name,dereddening_data=False)
        e_bvstart=0.1
        R_Vstart=3.1
        dist_start=100
        use_parafile=st.checkbox('Use your own parameter file',value=False)
        slider_dict={
            'Mstar':{
                'label':r'$log_{10}(M_{star}) [M_{sun}]$',
                'lims':[-0.69, 0.39],
                'x0':0.06,
                'priority':1}
                ,
            
            'Teff':{
                'label':r'$log_{10}(T_{eff})$',
                'lims':[3.5, 4.0], 
                'x0':3.69,
                'priority':1},
            
            'Lstar':{
                'label':r'$log_{10}(L_{star})$',
                'lims':[-1.3, 1.7],
                'x0':0.79,
                'priority':1}, 
            'fUV':{
                'label':r'$log_{10}(fUV)$',
                'lims':[-3, -1],
                'x0':-1.57, 
                'priority':1},
            
            'pUV':{
                'label':r'$log_{10}(pUV)$',
                'lims':[-0.3, 0.39],
                'x0':-0.02, 
                'priority':1},
            
            'Mdisk':{
                'label':r'$log_{10}(M_{disk})$',
                'lims':[-5, 0],
                'x0':-1.367, 
                'priority':2},
            
            'incl':{
                'label':r'$incl [Deg]$',
                'lims':[0.0, 90.0],
                'x0':20.0,
                'priority':2},
            
            'Rin':{
                'label':r'$log_{10}(R_{in}[AU])$',
                'lims':[-2.00, 2.00], 
                'x0':-1.34,
                'priority':2},
           
             'Rtaper':{
                'label':r'$log_{10}(R_{taper}[AU])$',
                'lims':[0.7, 2.5],
                 'x0':1.95, 
                'priority':2},
            
            'Rout':{
                'label':r'$log_{10}(R_{out}[AU])$',
                'lims':[1.3, 3.14],
                'x0':2.556, 
                'priority':2},
            
            'epsilon':{
                'label':r'$\epsilon$',
                'lims':[0, 2.5],
                'x0':1, 
                'priority':2},
            
            'MCFOST_BETA':{
                'label':r'$\beta$',
                'lims':[0.9, 1.4],
                'x0':1.15, 
                'priority':2},
            
            'MCFOST_H0':{
                'label':'H_0[AU]',
                'lims':[3, 35],
                'x0':12, 
                'priority':2},    
            
            'a_settle':{
                'label':r'$log_{10}(a_{settle})$',
                'lims':[-5, -1],
                'x0':-3, 
                'priority':3},
            
            'amin':{
                'label':r'$log_{10}(a_{min})$',
                'lims':[-3, -1],
                'x0':-1.5, 
                'priority':3},
            
            
            'amax':{
                'label':r'$log_{10}(a_{max})$',
                'lims':[2.48, 4],
                'x0':3.6, 
                'priority':3},
            
            'apow':{
                'label':r'$a_{pow}$',
                'lims':[3, 5],
                'x0':3.6, 
                'priority':3},
            
            'Mg0.7Fe0.3SiO3[s]':{
                'label':r'Mg0.7Fe0.3SiO3[s]',
                'lims':[0.45, 0.7],
                'x0':0.57, 
                'priority':3},
            
            'amC-Zubko[s]':{
                'label':r'amC-Zubko[s]',
                'lims':[0.05, 0.3],
                'x0':0.18, 
                'priority':3},
            
            'fPAH':{
                'label':r'$log_{10}(fPAH)$',
                'lims':[-3.5, 0],
                'x0':-1.5, 
                'priority':3},
            
            'PAH_charged':{
                'label':r'PAH_charged',
                'lims':[0, 1], 
                'priority':3},
        }
            
        log_dict={'Mstar': 'log', 'Lstar': 'log', 'Teff': 'log', 'fUV': 'log', 'pUV': 'log', 'amin': 'log', 'amax': 'log',
              'apow': 'linear', 'a_settle': 'log', 'Mg0.7Fe0.3SiO3[s]': 'linear', 'amC-Zubko[s]': 'linear', 'fPAH': 'log',
           'PAH_charged': 'linear', 'Mdisk': 'log', 'Rin': 'log', 'Rtaper': 'log', 'Rout': 'log', 'epsilon': 'linear',
           'MCFOST_H0': 'linear', 'MCFOST_BETA': 'linear', 'incl': 'linear'}#,'Dist[pc]':'linear'}
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
        
        #print(slider_dict)
        if timing:
            start=time()

        questions = {
     #   'Complexity': ['Single zone', 'Two-zone'], 
     #   'Two-zone flavor': ['discontinues', 'continues','smooth'],
        'Input version':['Slider','Text only']}
  
        st.sidebar.markdown('# Create your disk')


        for question, answers in questions.items():
            valid_to_select = True
            if question!='Two-zone flavor' or complexity=='Two-zone':

                st.sidebar.markdown("### " + question.replace('-', ' ').capitalize() + '?')
                if valid_to_select:
                            if question=='Complexity':
                                complexity = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')
                            else:
                                if question=='Input version':
                                    input_version = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')

                                else:
                                    selected_answer = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')

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
        
        st.sidebar.write('$e(B-V)$')
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
            name=slider_dict[key]['label']
            try:
                value=float(slider_dict[key]['x0'])
            except:
                value=float((maxi+mini)/2)
            if input_version=='Slider':
                col1,col2=st.sidebar.columns([1,1])
                with col1:
                    st.sidebar.write(name)
                with col2:
                    middle=st.sidebar.slider('',min_value=float(mini),max_value=float(maxi),value=value)
                if 'log' in name:
                    middle=st.sidebar.text_input(label='',value=float(np.round(10**middle,6)),key=c)#
                    middle=np.log10(float(middle))
                    c+=1       
    
                else:
                    middle=st.sidebar.text_input(label='',value=float(middle),key=c) #name
                    c+=1
                #print(c)
                st.sidebar.markdown('---')
            else:
                
                st.sidebar.write(name)
                if 'log' in name:
                    
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
            val_trans, pos=transform_parameter(key,middle)
            features[0,pos]=val_trans
            if key=='amC-Zubko[s]':
                middle_sio=0.75-middle
                val_trans_sio, pos_sio=transform_parameter('Mg0.7Fe0.3SiO3[s]',middle_sio)
                features[0,pos_sio]=val_trans_sio
             
            #print(val_trans)  
        #print(features)
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

            alt_chart=alt.Chart(df).mark_point().encode(
                x=alt.X('lambda',scale=alt.Scale(type="log")),y=alt.Y('SED',scale=alt.Scale(type="log")),
                    #tooltip=[alt.Tooltip('lam', title=r'$ \nu F_\nu [erg/cm^2/s]$'),
                   # alt.Tooltip("SED", title="Price (USD)"),]
            ).properties(
        width=800,
        height=500
    )

            st.altair_chart(alt_chart)
        
        else:
            
            
            l, = ax.plot(t, s,marker='+',linestyle='none')


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
            else:
                lam_min=lam_min_start
                lam_max=lam_max_start
                flux_min=flux_min_start
                flux_max=flux_max_start
                
                
            st.markdown('---')
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
            st.success('Successfully predicted the SED')
            st.markdown('----')
            exp_para=st.checkbox(label='Export Parameters',value=False)
            if exp_para:
                st.warning('This feature does not work yet.')
                #print('export model parameters')
            #exp_sed=st.checkbox(label='Export model SED',value=False)
            #if exp_sed:
                #print('export SED')
            un_fac=0.05
            sed_string='lam[mic] nuF[erg/cm^2/s] sigma \n'
            for i in range(len(wavelength)):
                lam_1,flux_1,sig_1=wavelength[i],data[i],data[i]*un_fac
                sed_string=sed_string+ '%12.6e %12.6e %12.6e \n'%(lam_1,flux_1,sig_1)
            
            st.download_button('Download SED file', sed_string,'SED_model.txt')


        
#@st.cache

def tell_me_more():
    st.title('Background')

    st.button('Back to SED predictions')  # will change state and hence trigger rerun and hence reset should_tell_me_more

    st.markdown("""Here we put more background information.""")

    
#    st.latex()
if __name__ == '__main__':

#    logging.basicConfig(level=logging.CRITICAL)

    main()