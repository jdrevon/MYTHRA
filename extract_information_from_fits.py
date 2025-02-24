# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:15:52 2024

@author: jdrevon
"""

import numpy as np
from astropy.io import fits



def orientation_image(file_path):

    """
    Lit les valeurs de CDELT1 et CDELT2 dans un fichier OIFITS et retourne 
    une image réorientée de sorte que les axes X et Y soient croissants.
    
    Args:
        file_path (str): Chemin du fichier OIFITS.
    
    Returns:
        numpy.ndarray: L'image orientée.
    """
    # Ouvrir le fichier FITS
    with fits.open(file_path) as hdul:
        image = hdul[0].data  # Charger les données de l'image
        header = hdul[0].header
        
        # Lire les valeurs de CDELT1 et CDELT2
        cdelt1 = header.get('CDELT1', 1.0)  # Valeur par défaut 1.0 si absente
        cdelt2 = header.get('CDELT2', 1.0)
        
        # Appliquer les flips sans modifier le header
        if cdelt1 < 0:  # Flip horizontal si CDELT1 est négatif
            image = np.flip(image, axis=1)
        if cdelt2 < 0:  # Flip vertical si CDELT2 est négatif
            image = np.flip(image, axis=0)
    
    return image




def extract_information_from_fits(file_paths):
    images = []
    chi2 = []
    fovs = []    
    for file_path in file_paths:
        with fits.open(file_path) as hdul:
            header  = hdul[0].header
            header2 = hdul[2].header
            chi2_data = header2.get('CHISQ', None)
            # Lire les valeurs CRPIX, CRVAL, et CDELT
            crpix_x = header.get('CRPIX1', None)
            crpix_y = header.get('CRPIX2', None)
            cdelt_x = header.get('CDELT1', None)
            cdelt_y = header.get('CDELT2', None)
            image_data_x = header.get('NAXIS1', None)
            image_data_y = header.get('NAXIS2', None)
            if crpix_x is not None and crpix_y is not None and cdelt_x is not None and cdelt_y is not None:
                # Calculer la FoV
                fov_x = abs(cdelt_x) * image_data_x  # Largeur en unités physiques
                fov_y = abs(cdelt_y) * image_data_y  # Hauteur en unités physiques
                fovs.append((fov_x, fov_y))

        image_data = orientation_image(file_path)      
        
        images.append(image_data)
        chi2.append(chi2_data)
    return images, np.array(chi2), fovs

