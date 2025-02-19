# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:34:53 2024

@author: jdrevon
"""
import matplotlib
matplotlib.use('Agg') 
from image_selection import selected_data    
from extract_information_from_fits import extract_information_from_fits
from padding_image import zero_pad_images_v2
from mean_image import mean_image_func, resize_mean_image, stored_mean_fits_file, centering_function
from compare_model_data import compare_model_obs
from optimize_alignment import optimize_centering2
from MiRA_resampling import resampling_by_MiRA_parallelize

path_MiRA =    [
                "/home2/jdrevon/MiRA_rec_new_v4/piGru_40325/compactness/"
              ]



path_data = [
             "/home2/jdrevon/DATA_rec/continuum_4.0325-4.0368_increased_err_10.fits"
             ]

num_cores = 30 # number of parallelization of MiRA instance for the resampling
maxeval = 50 # number of evaluation function to resample the image with MiRA and convegres on the solution

for i in range(len(path_MiRA)):
    
    path_filtered_tmp, FoV_filtered_tmp, pixsize_filtered_tmp, header_filtered_tmp, chi2_filtered_tmp, param_filtered_tmp, hyperparameter_filtered_tmp  = selected_data(path_MiRA[i], 10, 0.1, 0.04) #bin, negative offset, zindow

    max_FoV, min_FoV, min_pixel_size = max(FoV_filtered_tmp), min(FoV_filtered_tmp), min(pixsize_filtered_tmp)/2
    
    path_filtered  = resampling_by_MiRA_parallelize(path_data[i], path_filtered_tmp, path_MiRA[i], FoV_filtered_tmp, param_filtered_tmp, hyperparameter_filtered_tmp, min_pixel_size, num_cores, maxeval)
    
    
    #Il faudrait stocker les nouvelles images resempler sous le nom de resampled images
    # Il faudrait egalement extraire les information sur le chi2 lié à chaque images resamplé (nettre ca en sortie de resampling_by_MiRA_parallelize)
    
    resampled_images, chi2_filtered, FoV_filtered = extract_information_from_fits(path_filtered)

    max_FoV, min_FoV = max(FoV_filtered)[0], min(FoV_filtered)[0]

    print(min_pixel_size, max_FoV, min_FoV)

    padded_images, x_padded, y_padded = zero_pad_images_v2(resampled_images, FoV_filtered, max_FoV, min_pixel_size)
    images_centered = centering_function(padded_images, path_MiRA[i])
    opt_images_centered = optimize_centering2(images_centered, path_data[i], min_FoV, min_pixel_size, path_filtered, chi2_filtered)
    
    mean_image, error_image  = mean_image_func(opt_images_centered, path_MiRA[i])
    new_mean_image, x_image, y_image = resize_mean_image(mean_image, min_pixel_size, min_FoV) # recentring the median images
    new_error_image, x_image_err, y_image_err  = resize_mean_image(error_image, min_pixel_size, min_FoV)
    store_mean_image, path_mean_fits = stored_mean_fits_file(new_mean_image, new_error_image, min_pixel_size, path_MiRA[i])
    chi2_V2, chi2_CP, chi2_tot, res = compare_model_obs(path_data[i], path_mean_fits, path_MiRA[i])







