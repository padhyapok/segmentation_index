#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:48:11 2023

@author: priyom
"""

#python libraries
from matplotlib import image, pyplot
#import tifffile
from math import ceil,pi
import numpy as np
import re
from scipy import signal
import itertools
from PIL import Image
from scipy.signal import fftconvolve
from scipy import ndimage
from skimage import color
from skimage import measure
from skimage import restoration
import cv2
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from os import listdir
from os.path import isfile, join, isdir
#functions list
###########################################1) get areas 
def get_the_areas_here(input_im,thresh_im,area_to_ignore):
    #threshold, segment and measure areas of individual objects 
    #ignore areas below a certain threshold
    binary_im=input_im>thresh_im
    labeled_mask, num_labels = ndimage.label(binary_im, structure=[[0,1,0],[1,1,1],[0,1,0]])
    clusters = measure.regionprops(labeled_mask, input_im)
    area_vals=[]
    row_vals=[]
    col_vals=[]
    for n in range(len(clusters)):
        if(clusters[n].area>area_to_ignore):
            area_vals.append(clusters[n].area)
            row_vals.append((clusters[n].centroid)[0])
            col_vals.append((clusters[n].centroid)[1])
    col_vals_in_order_l=sorted(col_vals)  
    col_vals_in_order_l=[round(vs) for vs in col_vals_in_order_l]
    row_vals_in_order_l=[round(x) for x,y in sorted(zip(row_vals,col_vals), key=lambda pair: pair[1])]    
    area_vals_in_order_l=[x for x,y in sorted(zip(area_vals,col_vals), key=lambda pair: pair[1])]
    ##uncomment to show the displays    
    #pyplot.imshow(input_im)
    #pyplot.show()
    #pyplot.rcParams["figure.figsize"] = (15, 9) 
    #pyplot.imshow(color.label2rgb(labeled_mask, bg_label=0))
    #pyplot.show()
    return (area_vals_in_order_l,row_vals_in_order_l,col_vals_in_order_l)
###########################################2) join peaks close
def join_peaks_close(col_vals_in_order_l,dist_merge):
    point_list=list(range(len(col_vals_in_order_l)))
    #measure consecutive distances
    col_diff_list=[abs(p1-p2) for p1, p2 in zip(col_vals_in_order_l,col_vals_in_order_l[1:])]
    #measure cutoff
    find_breaks=[locs for locs in range(len(col_diff_list)) if col_diff_list[locs]>dist_merge]
    groups_to_merge=[]
    ini_pos=0
    for lc in find_breaks:
        groups_to_merge.append(point_list[ini_pos:lc+1])
        ini_pos=lc+1
    groups_to_merge.append(point_list[ini_pos:])    
    return (groups_to_merge)
###########################################3) merge intensities
def check_intensity_merger(groups_to_merge,area_vals_in_order_l,row_vals_in_order_l,col_vals_in_order_l):
    #loop over groups to merge
    new_list=[]
    ar_val_merge_cur_l=[]
    r_val_merge_cur_l=[]
    c_val_merge_cur_l=[]
    for kl in range(len(groups_to_merge)):
        list_grouped=groups_to_merge[kl]
        if(len(list_grouped)==1):
            new_list.append(list_grouped[0])
            ar_val_merge_cur_l.append(area_vals_in_order_l[list_grouped[0]])
            r_val_merge_cur_l.append(row_vals_in_order_l[list_grouped[0]])
            c_val_merge_cur_l.append(col_vals_in_order_l[list_grouped[0]])
        elif(len(list_grouped)>1):
            all_areas=[area_vals_in_order_l[vl] for vl in list_grouped]
            ar_val_merge_cur_l.append(np.sum(all_areas))
            all_centroid_r=[row_vals_in_order_l[vl] for vl in list_grouped]
            all_centroid_r_mean=round(np.mean(all_centroid_r))                       
            all_centroid_c=[col_vals_in_order_l[vl] for vl in list_grouped]
            all_centroid_c_mean=round(np.mean(all_centroid_c))
            r_val_merge_cur_l.append(all_centroid_r_mean)
            c_val_merge_cur_l.append(all_centroid_c_mean)
    return (ar_val_merge_cur_l,r_val_merge_cur_l,c_val_merge_cur_l)
##########################################4) combined list sorter
#this functions sorts any list based on another list
def combined_list_sorter(ar_val_merge_l,r_val_merge_l,c_val_merge_l,Arg):
    fake_index_list=list(range(len(ar_val_merge_l)))
    fake_index_sort_list=[x for _,x in sorted(zip(ar_val_merge_l,fake_index_list), key=lambda pair:pair[0],reverse=Arg)]
    r_val_l_sorted_l=[r_val_merge_l[fl] for fl in fake_index_sort_list]
    c_val_l_sorted_l=[c_val_merge_l[fl] for fl in fake_index_sort_list]
    return (r_val_l_sorted_l,c_val_l_sorted_l)
###########################################5)
#no need to cal distance
def removeanteriorsegments(pos_initial_row_l,pos_initial_col_l,r_val_merge_old_mod_l,c_val_merge_old_mod_l,curve_par_cutoff_l):
    we_begin_row=pos_initial_row_l
    we_begin_col=pos_initial_col_l
    (prev_list_row,prev_list_col)=we_begin_row,we_begin_col
    index_list=list(range(len(c_val_merge_old_mod_l)))
    #this won't update if the next array positions are less, gives control of bright segments in vicinity
    only_peaks_you_care=[ind for ind in index_list if c_val_merge_old_mod_l[ind]>=pos_initial_col_l]
    for peak_loop_ind in only_peaks_you_care:
        cur_row=r_val_merge_old_mod_l[peak_loop_ind]
        cur_col=c_val_merge_old_mod_l[peak_loop_ind]
        calc_len=abs(cur_col-we_begin_col)
        if(calc_len>curve_par_cutoff_l):
            break
        prev_list_row=we_begin_row
        prev_list_col=we_begin_col
        we_begin_row=cur_row
        we_begin_col=cur_col
    #what we match is the distance between the first and the peak   
    #since these are centroid positions, need to round off vales
    we_begin_row=round(we_begin_row)
    we_begin_col=round(we_begin_col)
    return (we_begin_row,we_begin_col)
##################################################6)
#normalized cross correlation for finding adjacent segment pos in the next image
def normcorr(template, image):
    cor_val=0
    col_num=0
    ydim=min(np.shape(template)[0],np.shape(image)[0])
    template1=template[min(ydim-150,0):ydim,:]
    collect_gh=[]
    for nm in range(0,np.shape(image)[1]-np.shape(template)[1]+1,1):
        template2=image[min(ydim-150,0):ydim,nm:nm+np.shape(template)[1]]
        #norm level
        t1_m=(template1-np.mean(template1))
        t2_m=(template2-np.mean(template2))
        cor_level=np.sum(t1_m*t2_m)
        den_level=np.sqrt(np.sum(t1_m*t1_m)*np.sum(t2_m*t2_m))
        cur_cor_val=cor_level/float(den_level)
        if((cur_cor_val)>cor_val):
            col_num=nm
            cor_val=cur_cor_val
    return col_num
    
################################################################
#this function organizes the pattern based on the distance history
def find_out_order_segments(rem_ar_val_merge_cur_l,rem_r_val_merge_cur_l,rem_c_val_merge_cur_l,rem_r_val_merge_old_l,rem_c_val_merge_old_l,\
                            r_val_l_sorted_l,c_val_l_sorted_l):
    area_vals_copy=rem_ar_val_merge_cur_l[:]
    rogues_col=[]
    rogues_row=[]
    area_new=[]
    #align the two lists and find the minimum position
    collect_col_pos=rem_c_val_merge_old_l[:]
    collect_row_pos=rem_r_val_merge_old_l[:]
    collect_col_pos2=rem_c_val_merge_cur_l[:]
    collect_row_pos2=rem_r_val_merge_cur_l[:]
    #save original copy here
    collect_col_pos2_store=rem_c_val_merge_cur_l[:]
    collect_row_pos2_store=rem_r_val_merge_cur_l[:]
    for ind1 in range(0,len(collect_col_pos)-1):
    
        dist1=abs(collect_col_pos[ind1+1]-collect_col_pos[ind1])
        dist2=0
        diff_storer=np.empty(len(collect_col_pos2)-1)
        diff_storer[:]=np.NaN
        counter=0
        for ind2 in range(counter,len(collect_col_pos2)-1):
            sd=abs(collect_col_pos2[ind2+1]-collect_col_pos2[ind2])
            dist2=dist2+sd
            diff_cur=abs(dist2-dist1)
            diff_storer[ind2]=diff_cur
        min_pos=np.amin(diff_storer)
        min_pos_loc=np.argmin(diff_storer)+1
        rogues_col.extend(collect_col_pos2[counter+1:min_pos_loc])  
        rogues_row.extend(collect_row_pos2[counter+1:min_pos_loc])
        area_new.extend(area_vals_copy[counter+1:min_pos_loc])
        list_to_del=range(counter,min_pos_loc)
        #delete the rest col
        collect_col_pos2_n=np.delete(collect_col_pos2,list_to_del)
        collect_col_pos2=collect_col_pos2_n.tolist()
        #delete the rest row
        collect_row_pos2_n=np.delete(collect_row_pos2,list_to_del)
        collect_row_pos2=collect_row_pos2_n.tolist()
        #delete area 
        area_vals_copy_n=np.delete(area_vals_copy,list_to_del)
        area_vals_copy=area_vals_copy_n.tolist()   
        
    leftovers_row=collect_row_pos2[1:]
    leftovers_col=collect_col_pos2[1:]
    area_new.extend(area_vals_copy[1:])
    #compiling new list
    compiled_row=rogues_row+leftovers_row#ordering ival_l+n style together
    compiled_col=rogues_col+leftovers_col
    r_val_l_temp,c_val_l_temp=combined_list_sorter(area_new,compiled_row,compiled_col,True)
    r_val_l_func=r_val_l_sorted_l+r_val_l_temp
    c_val_l_func=c_val_l_sorted_l+c_val_l_temp
    r_val_l_func_up=r_val_l_func[:]
    c_val_l_func_up=c_val_l_func[:]
    #this portion added to replace c_val
    collect_col_pos2_rem=[col_p for col_p in collect_col_pos2_store if col_p not in compiled_col]
    for old_peak_col_val in range(len(collect_col_pos)):
        col_val_to_search=collect_col_pos[old_peak_col_val]
        search_index_col=c_val_l_func.index(col_val_to_search)#search in the original function
        val_to_replace=collect_col_pos2_rem[old_peak_col_val]
        row_loc_inferred=collect_col_pos2_store.index(val_to_replace)
        c_val_l_func_up[search_index_col]=val_to_replace
        r_val_l_func_up[search_index_col]=collect_row_pos2_store[row_loc_inferred]    
    return (r_val_l_func_up,c_val_l_func_up)
###########################################################################
#main function for calculating order
def aligner(old_image,current_image,pos_initial_col_l,r_val_merge_cur_l,c_val_merge_cur_l,ar_val_merge_cur_l,too_little_diff_l):
    ############################################################
    #Steps that the algorithm follows-
    #########################################################
    ##ignore the regions that got taken off due to curvature from prev count, without losing the order counts
    old_im_to_read=old_image[:,:]
    l1_col=(max(pos_initial_col_l-200,0))#left=200,r=100
    col_old_im=np.shape(old_im_to_read)[0]
    col_new_im=np.shape(current_image)[0]
    template_from_image=old_im_to_read[:min(col_old_im,col_new_im),l1_col:pos_initial_col_l+80]#can probably run a loop across#modified in dapt code 50:250
    cur_im_to_read=current_image[:,:]
    l2_col=max(pos_initial_col_l-200,0)
    cur_im_break=cur_im_to_read[:,l2_col:pos_initial_col_l+200]#l=250,r=150
    offset_is_x=normcorr(template_from_image, cur_im_break)
    offset_is_x_wrt=offset_is_x+l2_col#check this offset
    diff_is=offset_is_x_wrt-l1_col
    if(abs(diff_is)<too_little_diff_l):#only update position if close enough,else registration foesnt'work
        new_peak_pos_col=pos_initial_col_l+diff_is
    else:
        new_peak_pos_col=pos_initial_col_l
    min_lists=[(pk-new_peak_pos_col)**2 for pk in c_val_merge_cur_l]
    no_need_to_update_first=min_lists.index(min(min_lists))
    val_cutoff_row_l=r_val_merge_cur_l[no_need_to_update_first]
    val_cutoff_col_l=c_val_merge_cur_l[no_need_to_update_first]
        
    no_need_to_update_row_list=r_val_merge_cur_l[:no_need_to_update_first]
    rem_r_val_merge_cur_l=r_val_merge_cur_l[no_need_to_update_first:]
    no_need_to_update_col_list=c_val_merge_cur_l[:no_need_to_update_first]
    rem_c_val_merge_cur_l=c_val_merge_cur_l[no_need_to_update_first:]
    rem_ar_val_merge_cur_l=ar_val_merge_cur_l[no_need_to_update_first:]
    return (rem_ar_val_merge_cur_l,rem_r_val_merge_cur_l,rem_c_val_merge_cur_l,val_cutoff_row_l,val_cutoff_col_l)
#####################
#sort all the files according to dpf
#base_path="/home/priyom/postdoc/somite_notochord_interaction_paper/fig7_final_code/data_post_2_14_23/dapt_ordering/"
base_path="/home/priyom/postdoc/somite_notochord_interaction_paper/fig7_final_code/data_post_2_14_23/wt_ordering/"
#base_path="/home/priyom/postdoc/somite_notochord_interaction_paper/fig7_final_code/data_post_2_14_23/fss_ordering/"
#Specifying all the parameters of the simulation
####################################
##parameters 
curve_par_cutoff=150
merging_cutoff=55#parameter which judges if detected clusters part of same segment
too_little_diff=90#parameter which merges segments if little difference
#thresholding parameter
area_to_ignore=50
#erosion parameter
structure_el_len=10
num_iterations=3
keyword='wt'#possible keywords are 'wt','fss' or 'dapt'
dapt_course="/"
dapt_key=""
#####################################
#wildtype fish list
#fish_no_list=["f1","f2","f3","f5","f6","f7","f8","f9","f11","f12"]
#fss fish list
#fish_no_list=["fss2","fss3","fss4","fss5","fss6","fss8","fss10"]
#dapt fish list
#timecourse1 
#fish_no_list=["dapt3","dapt6","dapt11","dapt12"]
#timecourse2
#fish_no_list=["dapt1"]
#timecourse3
#fish_no_list=["dapt2","dapt8","dapt11","dapt14","dapt15"]
#timecourse4
fish_no_list=["dapt2"]
order_pre_all=[]
order_during_all=[]
order_post_all=[]
fig=pyplot.figure()
for fl in range(len(fish_no_list)):
    pos_initial_row_in=0
    pos_initial_col_in=0
    base_path_actual=base_path+dapt_course+fish_no_list[fl]+"/"
    base_path_full=base_path+dapt_course+fish_no_list[fl]+"_g_roll_ball_125/"
    base_path_binary=base_path+dapt_course+fish_no_list[fl]+"_mask/"
    all_files=[f for f in listdir(base_path_full) if isfile(join(base_path_full, f))]
    g3_collect=[]
    #file named differently
    for ff in range(len(all_files)):
        s=all_files[ff]
        #wt collection
        if keyword==('wt'):
            r1=re.search(r'(\w+)_(\w+)_(\w+)',s)
            g3=r1.group(3)
            g3_replace=g3.replace('dpf','')
            g3_collect.append(int(g3_replace))
        #fss collection
        else:
            r1=re.search(r'(\w+)_(\w+)',s)
            g3=r1.group(2)
            g3_replace=g3.replace('dpf','')
            g3_collect.append(int(g3_replace))
    sorted_files=[x for x,_ in sorted(zip(all_files,g3_collect), key=lambda pair: pair[1], reverse = False)]
    #######s#####################################################################
    ##this loops over all timepoints of a particular fish
    rem_r_val_merge_cur=[]
    rem_r_val_merge_cur=[]
    r_val_l_sorted_cur=[]
    c_val_sorted_cur=[]
    old_image=[]
    #metrics for writing
    folder_write=base_path+'write_files2/'
    f_time_writer=open(folder_write+fish_no_list[fl]+dapt_key+'_time.txt', 'w')
    f_mismatch_writer=open(folder_write+fish_no_list[fl]+dapt_key+"_seq_over_exp.txt","w")
    f_delta_p=open(folder_write+fish_no_list[fl]+dapt_key+"_delta_p.txt", 'w')
    f3_mean_delta_p=open(folder_write+fish_no_list[fl]+dapt_key+"_mean_p.txt","w")
    for file in range(len(sorted_files)):#need 
        ############################################################
        ##this part does basic operations of reading and making profiles
        #basic file 
        file_actual=image.imread(base_path_actual+sorted_files[file])
        file_to_read=image.imread(base_path_full+sorted_files[file])
        #binary file read here
        #use extracted notochord mask
        name_bin_file=sorted_files[file].split('.tif')
        binary_file_name=base_path_binary+name_bin_file[0]+"_bw.txt"
        load_bn_file=np.loadtxt(binary_file_name,dtype=int,delimiter=",")
        bn_file_as_uint8=load_bn_file.astype(np.uint8)
        #erode a little since seeds are indistinguishable from one another
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,structure_el_len))
        erode_bin_file=cv2.erode(bn_file_as_uint8, kernel, iterations=num_iterations)#4
        extracted_im_act=np.multiply(file_to_read,erode_bin_file)
        extracted_im=np.multiply(file_to_read,erode_bin_file)
        #
        if(file<=2):
            thresh_im=25
        else:
            thresh_im=15
        
        #threshold each segment, threshold is set to a low value and additional area cutoff removes noise
        (area_vals_in_order,row_vals_in_order,col_vals_in_order)=get_the_areas_here(extracted_im,thresh_im,area_to_ignore)
        #disconnected regions which are part of the same segment are joined by first identifying close groups, then merging
        id_groups=join_peaks_close(col_vals_in_order,merging_cutoff)
        ar_val_merge_cur,r_val_merge_cur,c_val_merge_cur=check_intensity_merger(id_groups,area_vals_in_order,row_vals_in_order,col_vals_in_order)
        if(len(r_val_l_sorted_cur)==0):
            (r_val_l_sorted_cur,c_val_l_sorted_cur)=combined_list_sorter(ar_val_merge_cur,r_val_merge_cur,c_val_merge_cur,True) 
            
        else:
            #calc remove anterior segments here
            (pos_initial_row_out,pos_initial_col_out)=removeanteriorsegments(pos_initial_row_in,pos_initial_col_in,\
                                                                             r_val_merge_old,c_val_merge_old,curve_par_cutoff)
            #generate positions beyonf
            fake_index=list(range(len(c_val_merge_old)))
            fake_index_locs=[pk for pk in fake_index if c_val_merge_old[pk]>=pos_initial_col_out] 
            rem_r_val_merge_old=[r_val_merge_old[flocs] for flocs in fake_index_locs]
            rem_c_val_merge_old=[c_val_merge_old[flocs] for flocs in fake_index_locs]
            #using unfiltered images for aligning
            (rem_ar_val_merge_cur,rem_r_val_merge_cur,rem_c_val_merge_cur,val_cutoff_row,val_cutoff_col)=\
            aligner(old_image,extracted_im_act,pos_initial_col_out,r_val_merge_cur,c_val_merge_cur,ar_val_merge_cur,too_little_diff)
            #update new position col,row found here
            pos_initial_row_in=val_cutoff_row
            pos_initial_col_in=val_cutoff_col
            if(len(rem_c_val_merge_cur)>0):
                (r_val_l_sorted_cur,c_val_l_sorted_cur)=find_out_order_segments(rem_ar_val_merge_cur,rem_r_val_merge_cur,rem_c_val_merge_cur,\
                                                                rem_r_val_merge_old,rem_c_val_merge_old,r_val_l_sorted_cur,c_val_l_sorted_cur)
    
        peak_nos=range(len(c_val_l_sorted_cur))
        sorted_peak_nos=[x for _,x in sorted(zip(c_val_l_sorted_cur,peak_nos), key=lambda pair:pair[0],reverse=False)]
        ##########################################
        #write current peak positions
        for peak_num in c_val_l_sorted_cur:
                f_time_writer.write(str(peak_num))
                f_time_writer.write('\t')
        f_time_writer.write("\n")
        ##########################################
        #updates old image parameters 
        ar_val_merge_old=ar_val_merge_cur[:]#this could be merged to the rem r_val_merge_list
        r_val_merge_old=r_val_merge_cur[:]
        c_val_merge_old=c_val_merge_cur[:]
        old_image=extracted_im_act[:,:]
    f_time_writer.flush()
    f_time_writer.close()
    #################################################################
    peak_nos=list(range(0,len(sorted_peak_nos)))
    #metric1
    hist0=np.subtract(list(peak_nos),sorted_peak_nos)
    for h0 in hist0:
        f_mismatch_writer.write(str(h0))
        f_mismatch_writer.write("\t")
    f_mismatch_writer.flush()
    f_mismatch_writer.close()
    #use the sorted peak nos to determine order parameter
    #metric2
    if(keyword=='wt' or keyword=='fss'):
        p_ini=0
        p_ini_pos=sorted_peak_nos.index(p_ini)
        delta_p_list=[]
        absolute_dis_list=[]
        for p_add in range(1,len(sorted_peak_nos)):
            p_next=p_add
            p_next_pos=sorted_peak_nos.index(p_next)
            #print('df',p_next_pos)
            delta_p=p_next_pos-p_ini_pos
            #print('df_m',delta_p)
            delta_p_list.append(delta_p)
            absolute_dis_list.append(abs(delta_p))
            p_ini_pos=p_next_pos
        total_dis=np.mean(absolute_dis_list)    
        for kk in delta_p_list:
            f_delta_p.write(str(kk))
            f_delta_p.write('\n')
        f_delta_p.flush()    
        f_delta_p.close()
        f3_mean_delta_p.write(str(total_dis))
        f3_mean_delta_p.flush()
        f3_mean_delta_p.close()
    #additional for dapt and wt plots       
    if(keyword=='dapt'):
        search_ph=base_path+dapt_course+fish_no_list[fl]+"_val.txt"
        with open(search_ph) as to_read:
            fgh=to_read.readline().rstrip()
            fgh_o=[vals_fg for vals_fg in fgh.split('\t')]
        min_val=int(fgh_o[0])
        if(fgh_o[1]=='All'):
            max_val=len(peak_nos)
        else:
            max_val=min(int(fgh_o[1]),len(peak_nos))
        mid_point_gaps=(min_val-1+max_val-1)/2.    
        to_scale=(mid_point_gaps-(min_val-1))
        #write to the file here
        #calculate original order
        hist1=np.subtract(list(peak_nos),sorted_peak_nos)
        rescaled_vals_are=np.multiply((np.subtract(peak_nos,mid_point_gaps)),1.0/to_scale)
        #
        f_p=open(folder_write+fish_no_list[fl]+dapt_key+"_spatial.txt",'w')
        for kk in range(len(hist1)):
            print(str(rescaled_vals_are[kk]),str(hist1[kk]),file=f_p)
        f_p.flush()
   
    


