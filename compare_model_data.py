# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 08:19:27 2024

@author: jdrevon
"""

from DATA_reading import OIFITS_READING_concatenate
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import ifftshift, fftshift, fftfreq, fft2
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from astropy import units as units
from scipy.ndimage import shift



from matplotlib.colors import LinearSegmentedColormap

# Définir les segments de couleurs pour la colormap ajustée
colors = [
    (0, 0, 0),    # Noir (début)
    (1, 0, 0),    # Rouge
    # Spectre inversé
    (1, 0.5, 0),  # Orange
    (1, 1, 0),    # Jaune
    (0, 1, 0),    # Vert
    (0, 1, 1),    # Cyan
    (0, 0, 1),    # Bleu (transition intermédiaire)
    (1, 1, 1)     # Blanc (fin)
]

# Créer la colormap avec des transitions lisses
custom_cmap = LinearSegmentedColormap.from_list('custom_spectrum', colors, N=256)


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
        cdelt1 = header.get('CDELT1')  # Valeur par défaut 1.0 si absente
        cdelt2 = header.get('CDELT2')
        
        # Appliquer les flips sans modifier le header
        if cdelt1 < 0:  # Flip horizontal si CDELT1 est négatif
            image = np.flip(image, axis=1)
        if cdelt2 < 0:  # Flip vertical si CDELT2 est négatif
            image = np.flip(image, axis=0)
    
    return image

def reading_info_OIFITS(path_data, path_image):

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

    image, header = read_fits_image(path_image) #Read the image
    
    return q_u, q_v, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, T3, image, header, T3_ERR

def mas_to_rad(values):
    """Convert milliarcseconds to radians."""
    return values * (np.pi / (180 * 3600 * 1000))

def rad_to_mas(values):
    """Convert radians to milliarcseconds."""
    return values / (np.pi / (180 * 3600 * 1000))

def create_coordinate_arrays(image_shape, x_center, y_center, x_scale, y_scale):
    x_size, y_size = image_shape
    x = np.linspace(-0.5, 0.5, x_size)
    x_image = x*(x_size*x_scale)
    y_image = x_image

    # return np.linspace(-x_size*x_scale/2,x_size*x_scale/2,x_size), np.linspace(-y_size*y_scale/2,y_size*y_scale/2,y_size)
    return x_image, y_image



def pad_image(image, padding):
    """Pads an image with additional zeros for Fourier transform."""
    dimy, dimx = image.shape
    padx = (dimx * padding - dimx) // 2
    pady = (dimy * padding - dimy) // 2
    return np.pad(image, ((pady, pady), (padx, padx)), 'constant', constant_values=0)

def spatial_to_frequency_rad(image, pixelsize):
    """Convert spatial lengths in rad to spatial frequencies in rad^-1 for a 2D grid."""

    freq  = fftshift(fftfreq(image.shape[0], pixelsize))

    return freq

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



def image_to_FFT(image):
    """Compute the 2D Fourier Transform of the image."""
    return   ifftshift(fft2(fftshift(image)))

def read_fits_image(filename):
    """Read the FITS file."""
    with fits.open(filename) as hdul:
       header = hdul[0].header       
    image = orientation_image(filename)
       
    return image, header

def extract_header_info(header):
    """Extract necessary information from the FITS header and convert them in radians."""
    try:
        unit= header['CUNIT1']
    except:
        unit = 'mas'
        print('WARNING no units found in the header put to mas by default')

    if unit == 'mas':
        
        if header['CDELT1'] < 0:
        
            x_scale  = -header['CDELT1']*units.mas.to(units.rad)
            y_scale  = header['CDELT2']*units.mas.to(units.rad)

        elif header['CDELT2'] < 0:
        
            x_scale  = header['CDELT1']*units.mas.to(units.rad)
            y_scale  = -header['CDELT2']*units.mas.to(units.rad)

        else:
            
            x_scale  = header['CDELT1']*units.mas.to(units.rad)
            y_scale  = header['CDELT2']*units.mas.to(units.rad)


    elif unit == 'rad':

        if header['CDELT1'] < 0:
            x_scale  = -header['CDELT1']
            y_scale  = header['CDELT2']

        elif header['CDELT2'] < 0:
            x_scale  = header['CDELT1']
            y_scale  = -header['CDELT2']

        else:            
            x_scale  = header['CDELT1']
            y_scale  = header['CDELT2']
        
        
    x_center = header['CRPIX1'] #-1  # FITS headers are 1-indexed
    y_center = header['CRPIX2'] #- 1  # FITS headers are 1-indexed
    
    return x_center, y_center, x_scale, y_scale


def IMAGE_to_V2(path_data, path_image, padding):
    """Convert an image to visibility squared (V2) data and extract the CP values"""

    # #Center the original image on the brightest pixel
    # image_centered = center_image_on_brightest_pixel(image)
    
    q_u_interp, q_v_interp, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, CP_MATISSE, image, header, CP_MATISSE_ERR = reading_info_OIFITS(path_data, path_image)
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
    x_center, y_center, x_scale, y_scale = extract_header_info(header) #read header info
    x_image, y_image = create_coordinate_arrays(image.shape, x_center, y_center, x_scale, y_scale) #create the x and y-axis list

    if padding > 0:
        padded_image = pad_image(image, padding)
        FFT_image = image_to_FFT(padded_image)
        q_image = spatial_to_frequency_rad(padded_image, x_scale)
        
    else:
        FFT_image = image_to_FFT(image)
        q_image = spatial_to_frequency_rad(image, x_scale)
 
    FFT_real = np.real(FFT_image)
    FFT_imag = np.imag(FFT_image)
    
    interpolator_real = RegularGridInterpolator((q_image, q_image), FFT_real, method='linear', bounds_error=False, fill_value=None)
    interpolator_imag = RegularGridInterpolator((q_image, q_image), FFT_imag, method='linear', bounds_error=False, fill_value=None)    

    def MIRA_complex(y, x):
          return interpolator_real((y, x)) + 1j*interpolator_imag((y, x))

    # interpolator = RegularGridInterpolator((q_image, q_image), FFT_final, method='linear', bounds_error=False, fill_value=None)
    V2_MiRA     = np.abs([MIRA_complex(q_v_interp[b], q_u_interp[b])/MIRA_complex(0, 0) for b in range(len(q_u_interp))])**2

    q1 = np.sqrt(q_u1**2+q_v1**2)
    q2 = np.sqrt(q_u2**2+q_v2**2)
    q3 = np.sqrt(q_u3**2+q_v3**2)

    Vis_B1 = np.array([MIRA_complex(q_v1[b], q_u1[b]) for b in range(len(q_u1))])
    Vis_B2 = np.array([MIRA_complex(q_v2[b], q_u2[b]) for b in range(len(q_u2))])
    Vis_B3 = np.array([MIRA_complex(q_v3[b], q_u3[b]) for b in range(len(q_u3))])
        
    W = Vis_B1*Vis_B2*np.conj(Vis_B3)/MIRA_complex(0, 0)**3
            
    CP_MiRA = [np.angle(W[b], deg=True) for b in range(len(W))]
    # CP_MiRA = np.array(CP)/np.pi*180
    
    q_CP = np.amax ([q1,q2,q3], axis=0)

    q = np.sqrt(q_u_interp**2+q_v_interp**2)
        
    return q, V2_MATISSE, V2_MATISSE_ERR, V2_MiRA, q_CP, CP_MiRA, CP_MATISSE, CP_MATISSE_ERR

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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



def compare_model_obs(path_data, path_image, path_MiRA):

    # Suppose que ces données sont déjà définies
    q, V2_MATISSE, V2_MATISSE_ERR, V2_MiRA, q_CP, CP_MiRA, CP_MATISSE, CP_MATISSE_ERR = IMAGE_to_V2(path_data, path_image, 8)
    
    q_u_interp, q_v_interp, q_u1, q_u2, q_u3, q_v1, q_v2, q_v3, V2_MATISSE, V2_MATISSE_ERR, CP_MATISSE, image, header, CP_MATISSE_ERR = reading_info_OIFITS(path_data, path_image)
    x_center, y_center, x_scale, y_scale = extract_header_info(header) #read header info
    x_image, y_image = create_coordinate_arrays(image.shape, x_center, y_center, x_scale, y_scale) #create the x and y-axis list
    res_instru = rad_to_mas(min(1/q))/2
    print(res_instru)
    # image = np.flip(image, axis=1)
    # print(rad_to_mas(x_image),rad_to_mas(y_image))
    # Supposons que vous avez déjà extrait toutes les données nécessaires
    # (q, V2_MATISSE, V2_MATISSE_ERR, V2_MiRA, q_CP, CP_MiRA, CP_MATISSE, CP_MATISSE_ERR)
    
    # Crée une figure avec 2 lignes et 2 colonnes
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Premier sous-graphe : Visibilités Log --- 
    axs[0, 0].errorbar(q, V2_MATISSE, yerr=V2_MATISSE_ERR, label='MATISSE', 
                  fmt='o', markersize=2, alpha=0.6, ecolor='tab:red', 
                  elinewidth=1.0, capsize=3, c='tab:orange', zorder=2)
    
    axs[0, 0].scatter(q, V2_MiRA, s=12, label='MIRA', c='tab:blue', zorder=3)
    
    axs[0, 0].set_xlabel("Spatial frequency (cycles/rad)", fontsize=15)
    axs[0, 0].set_ylabel("Squared visibilities", fontsize=15)
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylim(1e-5, 1e0)
    axs[0, 0].set_xlim(0, 3.5E7)
    axs[0, 0].legend(fontsize=10, loc='best', frameon=True, shadow=True)
    axs[0, 0].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
    
    # --- Deuxième sous-graphe : Visibilités Lin --- 
    axs[0, 1].errorbar(q, V2_MATISSE, yerr=V2_MATISSE_ERR, label='MATISSE', 
                  fmt='o', markersize=2, alpha=0.6, ecolor='tab:red', 
                  elinewidth=1.0, capsize=3, c='tab:orange', zorder=2)
    
    axs[0, 1].scatter(q, V2_MiRA, s=12, label='MIRA', c='tab:blue', zorder=3)
    
    axs[0, 1].set_xlabel("Spatial frequency (cycles/rad)", fontsize=15)
    axs[0, 1].set_ylabel("Squared visibilities", fontsize=15)
    axs[0, 1].set_ylim(1e-5, 1e0)
    axs[0, 1].set_xlim(0, 3.5E7)
    axs[0, 1].legend(fontsize=10, loc='best', frameon=True, shadow=True)
    axs[0, 1].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
    
    # --- Troisième sous-graphe : Closure Phase ---
    axs[1, 0].scatter(q_CP, np.array(CP_MiRA), s=12, label='MIRA', c='tab:blue', zorder=2, alpha=0.7)
    axs[1, 0].errorbar(q_CP, np.array(CP_MATISSE), yerr=np.array(CP_MATISSE_ERR), label='MATISSE', 
                  fmt='o', markersize=2, alpha=0.3, ecolor='tab:red', 
                  elinewidth=1.0, capsize=3, c='tab:orange', zorder=1)
    
    axs[1, 0].set_xlabel("Max. Spatial frequency (cycles/rad)", fontsize=15)
    axs[1, 0].set_ylabel("Closure Phase (degrees)", fontsize=15)
    axs[1, 0].set_xlim(0, 3.5E7)
    axs[1, 0].set_ylim(-185, 185)
    axs[1, 0].legend(fontsize=10, loc='best', frameon=True, shadow=True)
    axs[1, 0].grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
    
    # --- Quatrième sous-graphe : Image ---
    # Créez votre image en utilisant les informations que vous avez extraites
    plt.subplot(224)  # (2, 2, 4)

    plt.imshow(image / np.max(image), extent=(min(rad_to_mas(x_image)), max(rad_to_mas(x_image)), min(rad_to_mas(y_image)), max(rad_to_mas(y_image))), 
                origin='lower', cmap=custom_cmap)
    
    # Ajustement des axes
    ax = plt.gca()
    ax.set_xlim(min(rad_to_mas(x_image)), max(rad_to_mas(x_image)))
    ax.set_ylim(min(rad_to_mas(y_image)), max(rad_to_mas(y_image)))
    
    # Ajout d'une barre de couleur avec un label
    cbar = plt.colorbar(label="Normalized Intensity")
    cbar.ax.tick_params(labelsize=15)  # Ajuste la taille des labels de la barre de couleur
    
    # Ajout d'un titre
    # plt.title("Image Normalized Intensity", fontsize=16, pad=15)
    
    # Ajout de labels pour les axes
    plt.xlabel(r" $\alpha$ (mas)", fontsize=15)
    plt.ylabel(r" $\delta$ (mas)", fontsize=15)
    
    # Ajustement de la taille des ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Position des flèches pour désigner le Nord et l'Est
    north_x = min(rad_to_mas(x_image)) + 10  # Ajusté pour le coin inférieur droit
    north_y = min(rad_to_mas(y_image)) + 10
    east_x = north_x + 0
    east_y = north_y - 0  # Positionné juste en dessous du Nord
    
    # Flèche pour le Nord
    plt.arrow(north_x, north_y, 0, 15, head_width=2, head_length=2, fc='white', ec='white', zorder=5)
    plt.text(north_x + 3 , north_y + 20, 'N', fontsize=12, color='white', ha='left')
    
    # Flèche pour l'Est
    plt.arrow(east_x, east_y, 15, 0, head_width=2, head_length=2, fc='white', ec='white', zorder=5)
    plt.text(east_x + 25, east_y - 2, 'E', fontsize=12, color='white', ha='left')
    
    # Affichage de l'image
    plt.tight_layout()
    plt.show()
        
    # Ajustement du layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Laisser de l'espace pour le titre
    
    diff_CP = np.sum((np.rad2deg(np.angle(np.exp(1j*np.deg2rad(CP_MATISSE))
                               * np.exp(-1j*np.deg2rad((CP_MiRA)))))/CP_MATISSE_ERR)**2)


    chi2_V2_r = np.sum((V2_MiRA - V2_MATISSE)**2/V2_MATISSE_ERR**2)/len(V2_MiRA)
    chi2_CP_r = diff_CP/len(CP_MiRA)
        
    chi2_total = calculate_total_chi2(chi2_V2_r, chi2_CP_r, len(V2_MiRA), len(CP_MiRA))
    
    print(r"$\chi^2_{rV2}$ = %.2f"%chi2_V2_r)
    print(r"$\chi^2_{rCP}$ = %.2f"%chi2_CP_r)
    print(r"$\chi^2_{TOT}$ = %.2f"%chi2_total)

    # Affichage final
    plt.show()
    
    fig.savefig(path_MiRA+'/mean_final_plot_CP_%.2f_V2_%.2f_TOT_%.2f.png'%(chi2_CP_r, chi2_V2_r, chi2_total))
    
    return chi2_V2_r, chi2_CP_r, chi2_total, res_instru
