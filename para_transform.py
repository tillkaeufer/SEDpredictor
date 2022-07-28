import numpy as np

def para_to_parameterin(input_file,output_file):
    para_dict={}
    with open(input_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            split_line=line.split()
            value=float(split_line[0])
            parameter=split_line[1]
            para_dict[parameter]=value
    with open('TemplateParameter.in','r') as f_temp:
        lines=f_temp.readlines()
    with open(output_file,'w') as f:
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
            f.write(new_line)     