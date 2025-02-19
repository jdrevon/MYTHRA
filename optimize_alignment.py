# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:40:47 2024

@author: jdrevon
"""

import numpy as np


from scipy.ndimage import shift
from DATA_reading import OIFITS_READING_concatenate
from numpy.fft import ifftshift, fftshift, fftfreq, fft2
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def create_coordinate_arrays(image_shape, x_center, y_center, x_scale, y_scale):
    x_size, y_size = image_shape
    # x_image = (np.arange(x_size) - x_center) * x_scale
    # y_image = (np.arange(y_size) - y_center) * y_scale
    return np.linspace(-x_size*x_scale/2,x_size*x_scale/2,x_size), np.linspace(-y_size*y_scale/2,y_size*y_scale/2,y_size)

def shift_image(image, dx, dy, intensity):
    return shift(image*intensity, shift=(dy, dx), mode='constant', cval=0)

def resize_image(image_data, min_pixel_size, min_fov):
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

    return image_data_paire, x_image, y_image



def mas_to_rad(values):
    """Convert milliarcseconds to radians."""
    return values * (np.pi / (180 * 3600 * 1000))

def rad_to_mas(values):
    """Convert radians to milliarcseconds."""
    return values / (np.pi / (180 * 3600 * 1000))

def pad_image(image, padding):
    """Pads an image with additional zeros for Fourier transform."""
    dimy, dimx = image.shape
    padx = (dimx * padding - dimx) // 2
    pady = (dimy * padding - dimy) // 2
    return np.pad(image, ((pady, pady), (padx, padx)), 'constant', constant_values=0)


def spatial_to_frequency_rad(image, pixelsize):
    """Convert spatial lengths in rad to spatial frequencies in rad^-1 for a 2D grid."""
    # delta_x = x_image_rad[1] - x_image_rad[0]
    # delta_y = y_image_rad[1] - y_image_rad[0]

    # freq_x = fftshift(fftfreq(len(x_image_rad), d=delta_x)) 
    # freq_y = fftshift(fftfreq(len(y_image_rad), d=delta_y))

    freq  = fftshift(fftfreq(image.shape[0], pixelsize))

    return freq


def image_to_FFT(image):
    """Compute the 2D Fourier Transform of the image."""
    # return   np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(image, axes=[-2, -1]), axes=[-2, -1]), axes=[-2, -1])
    return   ifftshift(fft2(fftshift(image)))
    # return np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(image, axes=[-2, -1]), axes=[-2, -1]), axes=[-2, -1]))

def calculate_total_chi2(chi2_V2, chi2_CP, N_V2, N_CP):
    """
    Calcule le chi² total pondéré en fonction des contributions de chi² V2 et chi² CP.

    Parameters:
    - chi2_V2 : float, valeur de chi² pour les visibilités (V2)
    - chi2_CP : float, valeur de chi² pour les clôtures de phase (CP)
    - N_V2 : int, nombre total de valeurs pour les visibilités
    - N_CP : int, nombre total de valeurs pour les clôtures de phase

    Returns:
    - chi2_total : float, chi² total pondéré
    """
    if N_V2 + N_CP == 0:
        raise ValueError("Le nombre total de valeurs (N_V2 + N_CP) ne peut pas être zéro.")

    # Calcul du chi² total pondéré
    chi2_total = (N_V2 * chi2_V2 + N_CP * chi2_CP) / (N_V2 + N_CP)
    return chi2_total


def reading_info_OIFITS_data(path_data):
    
    # Read data
    
    DATA_OBS         = OIFITS_READING_concatenate(path_data)
    wavel            = DATA_OBS['VIS2']['WAVEL'][np.invert(DATA_OBS['VIS2']['FLAG'])]
    U                = DATA_OBS['VIS2']['U'][np.invert(DATA_OBS['VIS2']['FLAG'])]
    V                = DATA_OBS['VIS2']['V'][np.invert(DATA_OBS['VIS2']['FLAG'])]
    V2_MATISSE       = DATA_OBS['VIS2']['VIS2'][np.invert(DATA_OBS['VIS2']['FLAG'])]
    V2_MATISSE_ERR   = DATA_OBS['VIS2']['VIS2_ERR'][np.invert(DATA_OBS['VIS2']['FLAG'])]
    q_u              = U / wavel
    q_v              = V / wavel
    
    U1, U2           = DATA_OBS['T3']['U1'][np.invert(DATA_OBS['T3']['FLAG'])],DATA_OBS['T3']['U2'][np.invert(DATA_OBS['T3']['FLAG'])]
    U3               = U1+U2
    
    V1,V2            = DATA_OBS['T3']['V1'][np.invert(DATA_OBS['T3']['FLAG'])], DATA_OBS['T3']['V2'][np.invert(DATA_OBS['T3']['FLAG'])]
    V3               = V1+V2
    
    T3               = DATA_OBS['T3']['T3'][np.invert(DATA_OBS['T3']['FLAG'])]
    T3_ERR           = DATA_OBS['T3']['T3_ERR'][np.invert(DATA_OBS['T3']['FLAG'])]
    WL               = DATA_OBS['T3']['WAVEL'][np.invert(DATA_OBS['T3']['FLAG'])]
    
    q_u1, q_u2, q_u3 = U1/WL, U2/WL, U3/WL 
    q_v1, q_v2, q_v3 = V1/WL, V2/WL, V3/WL

    return q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR

    
def extract_V2_CP_from_image(image, pixelsize, q_u_interp, q_v_interp, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3):
    
    #  q_u_interp, q_v_interp = qu,qv from data of V2 table
    
    padding = 8
    padded_image = pad_image(image, padding)

    FFT_image = image_to_FFT(padded_image)
    q_image = spatial_to_frequency_rad(padded_image, pixelsize)

    FFT_real = np.real(FFT_image)
    FFT_imag = np.imag(FFT_image)
    
    
    interpolator_real = RegularGridInterpolator((q_image, q_image), FFT_real, method='linear', bounds_error=False, fill_value=None)
    interpolator_imag = RegularGridInterpolator((q_image, q_image), FFT_imag, method='linear', bounds_error=False, fill_value=None)    

    def MIRA_complex(y, x):
          return interpolator_real((y, x)) + 1j*interpolator_imag((y, x))

    # interpolator = RegularGridInterpolator((q_image, q_image), FFT_final, method='linear', bounds_error=False, fill_value=None)
    V2_image     = np.abs([MIRA_complex(q_v_interp[b], q_u_interp[b])/MIRA_complex(0, 0) for b in range(len(q_u_interp))])**2

    # FFT_real = np.real(FFT_image)
    # FFT_imag = np.imag(FFT_image)

    # interpolator_real = RegularGridInterpolator((q_image, q_image), FFT_real, method='linear', bounds_error=False, fill_value=None)
    # interpolator_imag = RegularGridInterpolator((q_image, q_image), FFT_imag, method='linear', bounds_error=False, fill_value=None)

    # def MIRA_complex(y, x):
    #       return interpolator_real((y, x)) + 1j*interpolator_imag((y, x))

    q1 = np.sqrt(q_u1**2+q_v1**2)
    q2 = np.sqrt(q_u2**2+q_v2**2)
    q3 = np.sqrt(q_u3**2+q_v3**2)

    Vis_B1 = np.array([MIRA_complex(q_v1[b], q_u1[b]) for b in range(len(q_u1))])
    Vis_B2 = np.array([MIRA_complex(q_v2[b], q_u2[b]) for b in range(len(q_u2))])
    Vis_B3 = np.array([MIRA_complex(q_v3[b], q_u3[b]) for b in range(len(q_u3))])
        
    W = Vis_B1*Vis_B2*np.conj(Vis_B3)/MIRA_complex(0, 0)**3
            
    CP_image = [np.angle(W[b], deg=True) for b in range(len(W))]
    # CP_MiRA = np.array(CP)/np.pi*180
    
    q_CP_image = np.amax ([q1,q2,q3], axis=0)

    q_image = np.sqrt(q_u_interp**2+q_v_interp**2)
    
    # print(q_image, V2_image)
    return q_image, V2_image, q_CP_image, CP_image




def compare_model_obs_from_image(image, x_image, y_image, pixelsize, q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR):

    def image_is_even(image):
        """Vérifie si les dimensions de l'image sont paires."""
        rows, cols = image.shape
        return rows % 2 == 0 and cols % 2 == 0
    
    def pad_image_to_even(image):
        """Ajoute un padding pour rendre l'image de taille paire."""
        rows, cols = image.shape
        pad_y = 0 if rows % 2 == 0 else 1
        pad_x = 0 if cols % 2 == 0 else 1
        
        padded_image = np.pad(image, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
        
        return padded_image

    if image_is_even(image):  # Exemple de vérification
        print("Traitement de l'image paire")
    else:
        print("Traitement de l'image impaire")
        image = pad_image_to_even(image)
    
    q_image, V2_image, q_CP_image, CP_image = extract_V2_CP_from_image(image, pixelsize, q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3)
    # print(V2_image)
    res_instru = rad_to_mas(min(1/q_image))/2
    print(res_instru)

    # print(q_u)
    # print(q_u1)

    V2_MiRA = V2_image
    CP_MiRA = CP_image
    CP_MATISSE = T3
    CP_MATISSE_ERR = T3_ERR
        
    diff_CP = np.sum((np.rad2deg(np.angle(np.exp(1j*np.deg2rad(CP_MATISSE))
                               * np.exp(-1j*np.deg2rad((CP_MiRA)))))/CP_MATISSE_ERR)**2)


    chi2_V2_r = np.sum((V2_MiRA - V2_MATISSE)**2/V2_MATISSE_ERR**2)/len(V2_MiRA)
    chi2_CP_r = diff_CP/len(CP_MiRA)
    
    # print(q_CP[0],CP_MATISSE[0], CP_MiRA[0])
    
    chi2_total = calculate_total_chi2(chi2_V2_r, chi2_CP_r, len(V2_MiRA), len(CP_MiRA))
    
    print(r"$\chi^2_{V2}$ = %.2f"%chi2_V2_r)
    print(r"$\chi^2_{CP}$ = %.2f"%chi2_CP_r)
    print(r"$\chi^2_{TOT}$ = %.2f"%chi2_total)
    
    return chi2_V2_r, chi2_CP_r, chi2_total, res_instru



# #######################################################################################

def chi2_func_final(image_ref, image, min_pixelsize, min_FoV, q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR):
    """
    Calcule le chi² global (chi² CP et chi² V2) entre une image transformée et une image de référence.
    
    :param image: Image (np.array)
    :param ref_image: Image de référence (np.array)
    :return: Valeur totale du chi² global (somme de chi² CP et chi² V2)
    """

    #Calculer la moyenne de l image
    mean_image = (image + image_ref)/2    
    mean_image = mean_image/np.sum(mean_image)
    
    mean_cropped, x_image, y_image = resize_image(mean_image, min_pixelsize, min_FoV)
    
    # Calculer les chi² CP et V2 en appelant la fonction existante
    chi2_cp, chi2_v2, chi2_TOT, res_instru = compare_model_obs_from_image(mean_cropped, mas_to_rad(x_image), mas_to_rad(y_image), mas_to_rad(min_pixelsize), q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR)
        
    return chi2_TOT, chi2_cp, chi2_v2



def image_optimization_selection(image, ref_image, min_FoV, min_pixelsize, path_data, intensity_range=(0.1, 1.9), 
                                                       max_iter=10, pixel_range=(-10, 10)):
    """
    :param image: Image à recentrer (np.array)
    :param ref_image: Image de référence (np.array)
    :param chi2_func: Fonction calculant le chi² global
    :param min_FoV: Champ de vision minimal
    :param min_pixelsize: Taille minimale des pixels
    :param path_data: Chemin vers les données nécessaires pour chi2_func
    :param intensity_range: Plage initiale pour l'intensité
    :param max_iter: Nombre maximum d'itérations pour la dichotomie
    :param pixel_range: Plage initiale pour dx et dy (en pixels entiers)
    :return: Image alignée, meilleurs décalages dx, dy, et intensité optimale
    """
    q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR = reading_info_OIFITS_data(path_data)

    # Calcul final du chi²
    chi2_final, chi2_cp, chi2_v2 = chi2_func_final(ref_image, image, min_pixelsize, min_FoV, q_u, q_v, 
                           q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, T3_ERR)
    print(f"chi² = {chi2_final:.6f}")
    
    return image, chi2_final, chi2_cp, chi2_v2

    
    
def optimize_centering2(image_centered, path_data, min_FoV, min_pixelsize, path_filtered, chi2_filtered, chi2_threshold, chi2_mean_V2, chi2_mean_CP):
    
    index = np.where(np.abs(chi2_filtered - 1) == np.min(np.abs(chi2_filtered - 1)))[0][0]
    image_reference = image_centered[index]
    images_centered = []

    for i in range(len(image_centered)):
        k = len(image_centered)
        print(f'Iteration : {i}/{k}')
        
        # Effectuer le recentrage et obtenir les résultats
        image_centered_tmp, chi2_final, chi2_cp, chi2_v2 = image_optimization_selection(
            image_centered[i], image_reference, min_FoV, min_pixelsize, path_data
        )
        
        # Vérifier si l'image centrée doit être incluse
        if np.median(chi2_filtered) < chi2_threshold:
            if np.logical_and(chi2_cp < chi2_mean_V2, chi2_v2 < chi2_mean_CP):
                images_centered.append(image_centered_tmp)
            else:
                path_rejected = path_filtered[i]
                print(f"This image has not been selected because its mean produces a high chi2. The corresponding path = {path_rejected}")
        else:
            images_centered.append(image_centered_tmp)
        
        # Recalculer l'image de référence comme la moyenne normalisée de toutes les images centrées
        if images_centered:
            image_reference = np.mean(images_centered, axis=0)
            image_reference = image_reference / np.sum(image_reference)  # Normaliser l'image de référence
    
    return images_centered
