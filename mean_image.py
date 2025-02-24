# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:30:20 2024

@author: jdrevon
"""


from PIL import Image
import numpy as np
from scipy.ndimage import shift
from astropy.io import fits
import time
import cv2

def create_coordinate_arrays(image_shape, x_center, y_center, x_scale, y_scale):
    x_size, y_size = image_shape
    # x_image = (np.arange(x_size) - x_center) * x_scale
    # y_image = (np.arange(y_size) - y_center) * y_scale
    return np.linspace(-x_size*x_scale/2,x_size*x_scale/2,x_size), np.linspace(-y_size*y_scale/2,y_size*y_scale/2,y_size)


def normalize_image(image):
    """
    Normalise une image numpy pour que ses valeurs soient dans la plage [0, 255].
    """
    image = image - np.min(image)  # Mettre les valeurs entre 0 et max
    image = image / np.max(image)  # Normaliser entre 0 et 1
    image = (image * 255).astype(np.uint8)  # Mettre entre 0 et 255
    return image

def create_gif(images, output_path, duration=100):
    """
    Crée un GIF à partir d'une liste d'images normalisées.
    :param images: Liste d'images numpy alignées.
    :param output_path: Chemin du fichier GIF de sortie.
    :param duration: Durée d'affichage de chaque image en millisecondes.
    """
    pil_images = [Image.fromarray(normalize_image(image)) for image in images]
    pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:], duration=duration, loop=0)
    print(f"GIF créé avec succès à {output_path}")


def center_on_photocenter(image):
    """
    Centers an image based on its photocenter (center of mass of intensity).
    :param image: 2D numpy array of the image.
    :return: Centered image.
    """
    y, x = np.indices(image.shape)
    total_intensity = np.sum(image)
    center_y = np.sum(y * image) / total_intensity
    center_x = np.sum(x * image) / total_intensity
    
    shift_y = image.shape[0] / 2 - center_y
    shift_x = image.shape[1] / 2 - center_x
    
    return shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)

def correlation_discrepancy(image, reference):
    """
    Computes the correlation discrepancy map between an image and a reference image
    using cv2.filter2D for cross-correlation.
    :param image: 2D numpy array of the image.
    :param reference: 2D numpy array of the reference image.
    :return: 2D numpy array of the correlation discrepancy map.
    """
    corr_map = cv2.filter2D(image, ddepth=-1, kernel=reference)
    return corr_map

def compute_shift_from_discrepancy(discrepancy_map):
    """
    Determines the shift required to align the image based on the maximum point in the discrepancy map.
    :param discrepancy_map: 2D numpy array of the correlation discrepancy map.
    :return: Tuple of (shift_y, shift_x) indicating the required shift.
    """
    max_pos = np.unravel_index(np.argmax(discrepancy_map), discrepancy_map.shape)
    center_pos = (discrepancy_map.shape[0] // 2, discrepancy_map.shape[1] // 2)
    shift_y = center_pos[0] - max_pos[0]
    shift_x = center_pos[1] - max_pos[1]
    
    return shift_y, shift_x

def refine_with_l1(image, reference, l1_iterations=5):
    """
    Refines the alignment of an image using L1 norm minimization relative to a reference image.
    :param image: 2D numpy array of the image.
    :param reference: 2D numpy array of the reference image.
    :param l1_iterations: Number of iterations for L1 norm adjustment.
    :return: L1-aligned image.
    """
    aligned_image = image.copy()
    for _ in range(l1_iterations):
        diff_y, diff_x = np.gradient(aligned_image - reference)
        shift_y = -np.sum(diff_y) / image.size
        shift_x = -np.sum(diff_x) / image.size
        aligned_image = shift(aligned_image, shift=(shift_y, shift_x), mode='constant', cval=0)
    return aligned_image

def refine_with_l2(image, reference):
    """
    Refines the alignment of an image using L2 norm minimization relative to a reference image.
    :param image: 2D numpy array of the image.
    :param reference: 2D numpy array of the reference image.
    :return: L2-aligned image.
    """
    diff_y, diff_x = np.gradient(image - reference)
    shift_y = -np.sum(diff_y * (image - reference)) / np.sum(diff_y**2 + diff_x**2)
    shift_x = -np.sum(diff_x * (image - reference)) / np.sum(diff_y**2 + diff_x**2)
    
    return shift(image, shift=(shift_y, shift_x), mode='constant', cval=0)

def centering_function(padded_images, path):
    # Compute the median image to use as a reference
    median_image = np.median(np.stack(padded_images), axis=0)

    # Center the reference (median image) on its photocenter
    centered_median = center_on_photocenter(median_image)

    create_gif(padded_images, path + "/non-aligned_images.gif")

    # Initial alignment based on correlation discrepancy
    initial_aligned_images = []
    for image in padded_images:
        # Center each image on its photocenter
        centered_image = center_on_photocenter(image)
        
        # Compute the discrepancy map relative to the centered median
        discrepancy_map = correlation_discrepancy(centered_image, centered_median)
        
        # Calculate the shift needed to align the image with the reference based on discrepancy
        shift_y, shift_x = compute_shift_from_discrepancy(discrepancy_map)
        
        # Shift the image accordingly
        aligned_image = shift(centered_image, shift=(shift_y, shift_x), mode='constant', cval=0)
        initial_aligned_images.append(aligned_image)
    
    # Apply L1 refinement for finer alignment
    l1_aligned_images = [refine_with_l1(image, centered_median) for image in initial_aligned_images]

    # Apply L2 refinement for even finer alignment
    refined_images = [refine_with_l2(image, centered_median) for image in l1_aligned_images]

    # Optional: Create a GIF of aligned images for visual comparison
    create_gif(refined_images, path + "/aligned_images.gif")

    return refined_images


def mean_image_func(refined_images, path):

    mean_image_refined = np.mean(np.stack(refined_images), axis=0)        
    stacked_images = np.stack(refined_images)
    error_map = np.std(stacked_images, axis=0)
    
    return mean_image_refined, error_map

    

############## RESIZE 

def resize_mean_image(image_data, min_pixel_size, min_fov):
    # Obtenir la forme de l'image
    original_shape = image_data.shape
    if original_shape[0] != original_shape[1]:
        raise ValueError("L'image doit être carrée.")
    
    # Dimension de l'image
    size = original_shape[0]
    
    # Calculer la nouvelle taille de l'image en fonction du FoV
    new_size = int(min_fov / min_pixel_size)  # Nouvelle taille en nombre de pixels
    
    # S'assurer que la nouvelle taille est inférieure à l'originale
    if new_size >= size:
        raise ValueError("La nouvelle taille doit être plus petite que la taille originale.")
    
    # Calculer les indices pour le recadrage
    start_index = (size - new_size) // 2  # Indice de début pour le crop
    end_index = start_index + new_size     # Indice de fin pour le crop
    
    # Découper l'image pour obtenir la nouvelle forme
    cropped_image = image_data[start_index:end_index, start_index:end_index]
    
    image_data = cropped_image
    original_shape = cropped_image.shape

    # Ensure the image is square
    if original_shape[0] != original_shape[1]:
        raise ValueError("L'image doit être carrée.")
    
    # Determine image size and make it even if it's odd
    size = original_shape[0]
    new_size = size - 1 if size % 2 != 0 else size

    # Crop the image to ensure even dimensions
    image_data_paire = image_data[:new_size, :new_size]

    
    x_image, y_image = create_coordinate_arrays(np.shape(image_data_paire), size / 2, size / 2, min_pixel_size, min_pixel_size)
    
    # from optimize_alignment import reading_info_OIFITS_data, compare_model_obs_from_image, mas_to_rad
    # path_data = ['C:/Users/jdrevon/Desktop/VLTI Observations/pi01Gru/LM/4_DATA_SORTED_FLUX_CAL_MEAN_QC/SiO53_4.1249-4.1383_increased_err_10.fits']
    # q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR = reading_info_OIFITS_data(path_data[0])
    # chi2_V2_r, chi2_CP_r, chi2_total, res_instru = compare_model_obs_from_image(store_mean_image, mas_to_rad(x_image), mas_to_rad(y_image), mas_to_rad(min_pixel_size), q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR)
    
    return image_data_paire, x_image, y_image


#################### STORE THE PICTURE IN FITS FILE

def stored_mean_fits_file(mean_image, error_image, min_pixel_size, path):

    image_data = mean_image
    original_shape = image_data.shape

    # Ensure the image is square
    if original_shape[0] != original_shape[1]:
        raise ValueError("L'image doit être carrée.")
    
    # Determine image size and make it even if it's odd
    size = original_shape[0]

    # Configure header for median image
    header_dict_image = {
        'SIMPLE': True,
        'BITPIX': -32,
        'NAXIS': 2,
        'NAXIS1': size,
        'NAXIS2': size,
        'CRPIX1': size / 2,
        'CRPIX2': size / 2,
        'CRVAL1': 0.0,
        'CRVAL2': 0.0,
        'CDELT1': min_pixel_size,
        'CDELT2': min_pixel_size,
        'CUNIT1': 'mas',
        'CUNIT2': 'mas',
        'HDUNAME': 'IMAGE-OI FINAL IMAGE'
    }
    header_image = fits.Header(header_dict_image)

    # Save the median image
    hdu_image = fits.PrimaryHDU(data=image_data, header=header_image)
    hdu_image.writeto(path + "/mean_image.fits", overwrite=True)
    hdu_image = None  # Explicitly delete the object to release the file
    print(f"Image médiane enregistrée avec succès dans {path}/mean_image.fits")

    # Delay to avoid file access issues
    time.sleep(1)

    # Configure header for error map
    header_dict_error = header_dict_image.copy()
    header_dict_error['HDUNAME'] = 'ERROR MAP'
    header_error = fits.Header(header_dict_error)

    # Save the error map
    hdu_error = fits.PrimaryHDU(data=error_image, header=header_error)
    hdu_error.writeto(path + "/error_map.fits", overwrite=True)
    hdu_error = None  # Explicitly delete the object to release the file
    print(f"Carte d'erreur enregistrée avec succès dans {path}/error_map.fits")

    # Final delay to ensure file operations are complete
    time.sleep(1)

    return image_data, path + "/mean_image.fits"
