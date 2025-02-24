# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:34:53 2024

@author: jdrevon
"""
import matplotlib
matplotlib.use('Agg') 
from image_selection import selected_data    
from extract_information_from_fits import extract_information_from_fits
from extract_interferometric_quantities_from_image import reading_info_OIFITS, extract_header_info, create_coordinate_arrays
from mean_image import mean_image_func, stored_mean_fits_file, create_gif
from compare_model_data import compare_model_obs
from optimize_alignment import optimize_centering2
from MiRA_resampling import resampling_by_MiRA_parallelize
from DATA_reading import OIFITS_READING_concatenate

path_MiRA =    ["path_to_output/regularization/"]
path_data = ["path_to_data/blabla.fits"]

num_cores = 30 # number of parallelization of MiRA instance for the resampling
maxeval = 50 # number of evaluation function to resample the image with MiRA and convegres on the solution

bin_num = 12 
negative_offset = 0.10
window = 0.04

chi2_threshold= 10 
chi2_mean_V2 = 5
chi2_mean_CP = 5

for i in range(len(path_MiRA)):
    
    path_filtered_tmp, FoV_filtered_tmp, pixsize_filtered_tmp, header_filtered_tmp, chi2_filtered_tmp, param_filtered_tmp, hyperparameter_filtered_tmp  = selected_data(path_MiRA[i], bin_num, negative_offset, window) 
    DATA_OBS         = OIFITS_READING_concatenate(path_data[i])
    q_u_interp, q_v_interp, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, CP_MATISSE, image, header, CP_MATISSE_ERR = reading_info_OIFITS(DATA_OBS, path_filtered_tmp[0])
    x_center, y_center, x_scale, y_scale = extract_header_info(header) #read header info
    x_image, y_image = create_coordinate_arrays(image.shape, x_center, y_center, x_scale, y_scale) #create the x and y-axis list
    
    max_FoV, min_FoV, min_pixel_size = max(FoV_filtered_tmp), min(FoV_filtered_tmp), min(pixsize_filtered_tmp)/2
    
    path_filtered  = resampling_by_MiRA_parallelize(path_data[i], path_filtered_tmp, path_MiRA[i], min_FoV, param_filtered_tmp, hyperparameter_filtered_tmp, min_pixel_size, num_cores, maxeval)
        
    resampled_images, chi2_filtered, FoV_filtered = extract_information_from_fits(path_filtered)
    create_gif(resampled_images, path_MiRA[i] + "/MiRA-aligned_images.gif")
    
    max_FoV, min_FoV = max(FoV_filtered)[0], min(FoV_filtered)[0]

    opt_images_centered = optimize_centering2(resampled_images, path_data[i], min_FoV, min_pixel_size, path_filtered, chi2_filtered, chi2_threshold, chi2_mean_V2, chi2_mean_CP)
    mean_image, error_image  = mean_image_func(opt_images_centered, path_MiRA[i])
    store_mean_image, path_mean_fits = stored_mean_fits_file(mean_image, error_image, min_pixel_size, path_MiRA[i])
    chi2_V2, chi2_CP, chi2_tot, res = compare_model_obs(path_data[i], path_mean_fits, path_MiRA[i])







