# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:40:47 2024
@author: jdrevon
"""

import numpy as np

def mas_to_rad(diam_star):
    y = diam_star*1E-3*np.pi/(180*3600)
    return y 


def create_coordinate_arrays(image_shape, x_center, y_center, x_scale, y_scale):
    x_size, y_size = image_shape
    # x_image = (np.arange(x_size) - x_center) * x_scale
    # y_image = (np.arange(y_size) - y_center) * y_scale
    return np.linspace(-x_size*x_scale/2,x_size*x_scale/2,x_size), np.linspace(-y_size*y_scale/2,y_size*y_scale/2,y_size)


from scipy.ndimage import gaussian_filter

def zero_pad_images_v2(resampled_images, fovs, max_fov, min_pixel_size, sigma=0.5):
    """
    Effectuer un remplissage plus doux avec un filtre gaussien sur les images pour qu'elles aient la même FoV.
    
    :param resampled_images: Liste d'images rééchantillonnées.
    :param fovs: Liste de tuples contenant la FoV de chaque image.
    :param max_fov: Tuple représentant la plus grande FoV (largeur, hauteur).
    :param min_pixel_size: Tuple représentant la plus petite taille de pixel (pixel_x, pixel_y).
    :param sigma: Paramètre de largeur du filtre gaussien pour l'adoucissement.
    :return: Liste d'images avec remplissage doux appliqué.
    """
    padded_images = []
    
    # Calculer les dimensions cibles en pixels pour correspondre au FoV maximal
    target_width = int(np.ceil(max_fov / min_pixel_size))  # Largeur en pixels
    target_height = int(np.ceil(max_fov / min_pixel_size))  # Hauteur en pixels

    for image in resampled_images:

        # Dimensions de l'image
        height, width = image.shape

        # Calculer les quantités de padding
        pad_width = (target_width - width) // 2
        pad_height = (target_height - height) // 2

        # Appliquer du padding avec des valeurs nulles (zero padding) au départ
        padded_image = np.pad(image, ((pad_height, target_height - height - pad_height),
                                      (pad_width, target_width - width - pad_width)), 
                              mode='constant', constant_values=0)

        # Appliquer un filtre gaussien pour adoucir le padding
        padded_image_smooth = gaussian_filter(padded_image, sigma=sigma)

        # Ajouter l'image lissée à la liste
        padded_images.append(padded_image_smooth)
        
    # Générer les nouvelles coordonnées pour les images paddées
    x_image_padded, y_image_padded = create_coordinate_arrays(
        (target_width, target_height), target_width // 2, target_height // 2, 
        min_pixel_size, min_pixel_size
    )

    return padded_images, x_image_padded, y_image_padded

