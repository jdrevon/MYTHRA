# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:14:04 2025

@author: jdrevon
"""

import numpy as np
import os 
from astropy.io import fits 
import matplotlib
matplotlib.use('Agg') 
from extract_interferometric_quantities_from_image import IMAGE_to_V2
import matplotlib.pyplot as plt
from DATA_reading import OIFITS_READING_concatenate
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

def rad_to_mas(rad):
    y = rad*1E3/(np.pi/(180*3600))
    return y    

def orientation_image(file_path):

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



def lister_dossiers(repertoire):
    dossiers = []
    for element in os.listdir(repertoire):
        if os.path.isdir(os.path.join(repertoire, element)):
            dossiers.append(element)
    return dossiers

def regroup_image(general_path):
    # general_path = path_results_tmp
    # general_path = "C:/Users/jdrevon/Desktop/MiRA_images/"
    
    noms_des_objets = lister_dossiers(general_path)
    
    for i in range(len(noms_des_objets)):
    
        reg = lister_dossiers(general_path + noms_des_objets[i])
        
        for j in range(len(reg)):
            
            path = general_path + noms_des_objets[i] + '/' + reg[j] + '/'
    
            path_param = lister_dossiers(path)
            
            if not os.path.exists(path+'ALL_images/'):
                os.mkdir(path+'ALL_images/')
            
            for k in range(len(path_param)):
                try:
                    shutil.copy(path+path_param[k]+'/reg_'+reg[j]+'_'+path_param[k]+"_image.png", path+'ALL_images/')
                except: 
                    None

    return

 
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def run_calculation_compactness(reg, param, pix, fov, h, path_param, maxeval, ftol, gtol, xtol, ftot, verb, prior_image, path_data):

    path_hyper = path_param + 'reg_%s_FoV_%.4f_pixsize_%.4f_param_%.4f_mu_%.4E.fits' % (reg, fov, pix, param, h)

    os.system("ymira -pixelsize=%.4fmas -fov=%.4fmas -regul=%s -mu=%.4E -gamma=%imas" % (
        pix, fov, reg, h, param) + \
              " -ftol=%.1f -gtol=%.1f -xtol=%.1f -maxeval=%i -verb=%i -overwrite -save_visibilities -save_initial " % (
                  ftol, gtol, xtol, maxeval, verb) + \
              " -flux=%.1f --use_vis=none -save_dirty_beam -save_dirty_map -save_residual_map --use_vis2=all --use_t3=all -initial=%s -recenter %s %s" % (ftot, prior_image, path_data, path_hyper))
    return path_hyper



def run_calculation_hyperbolic(reg, param, pix, fov, h, path_param, maxeval, ftol, gtol, xtol, ftot, verb, prior_image, path_data):

    path_hyper = path_param + 'reg_%s_FoV_%.4f_pixsize_%.4f_param_%.4E_mu_%.4E.fits' % (reg, fov, pix, param, h)

    os.system("ymira -pixelsize=%.4fmas -fov=%.4fmas -regul=%s -mu=%.4E -tau=%.4E" % (
        pix, fov, reg, h, param) + \
              " -ftol=%.1f -gtol=%.1f -xtol=%.1f -maxeval=%i -verb=%i -overwrite -save_visibilities -save_initial" % (
                  ftol, gtol, xtol, maxeval, verb) + \
              " -flux=%.1f --use_vis=none -save_dirty_beam -save_dirty_map -save_residual_map --use_vis2=all --use_t3=all -initial=%s -recenter %s %s" % (ftot, prior_image, path_data, path_hyper))
    return path_hyper

    # Creating the first folder in the folder architecture #regularization 


def run_MiRA(DATA_OBS, path_data, prior_image, pixelsize, FoV, hyperparameter, param, regularization, path_save, maxeval, num_cores, obs_res):    

    ftol = 0
    gtol = 0
    xtol = 0
    ftot = 1 #sum of total flux
    verb = int(maxeval/5)
    
    path_param = path_save + 'FoV_%.4f_pixsize_%.4f_param_%.4E_mu_%.4E/' % (FoV, pixelsize, param, hyperparameter)
                   
    if not os.path.exists(path_param):
        os.makedirs(path_param)

    if regularization == 'compactness':            
    
        path_file = run_calculation_compactness(regularization, param, pixelsize, FoV, hyperparameter, path_param, maxeval, ftol, gtol, xtol, ftot, verb, prior_image, path_data)

    if regularization == 'hyperbolic':            
    
        path_file = run_calculation_hyperbolic(regularization, param, pixelsize, FoV, hyperparameter, path_param, maxeval, ftol, gtol, xtol, ftot, verb, prior_image, path_data)

    # PLOT FIGURE

    fig1, axes = plt.subplots(1, 6, figsize=(30, 4))
        
    hdul = fits.open(path_file)
    
    chi2        =hdul[2].header['CHISQ']
    converge    =hdul[2].header['CONVERGE']
    neval       =hdul[2].header['NEVAL']
    gpnorm      =hdul[2].header['GPNORM']


    note = ""
    
    if np.logical_and(np.logical_and(converge == True, gpnorm<1), neval<maxeval-1) == True:        
        note        = 'C'
            
    elif np.logical_and(converge == True, np.logical_or(neval >= maxeval-1, gpnorm>1))== True:        
        note        = 'W'

    elif converge == False :
        note        = 'NC'


    q_MiRA, V2_MATISSE, V2_MATISSE_ERR, V2_MiRA, q_CP, CP_MiRA, CP_MATISSE = IMAGE_to_V2(DATA_OBS, path_file, 8) #compute the visibilities and closure phase

    
    image = orientation_image(path_file)
    shape = np.shape(image)
    x_image = np.linspace(-shape[0]/2,shape[0]/2,shape[0],endpoint=False)*pixelsize
    

    axes[0].imshow(image,extent=[min(x_image),max(x_image),min(x_image),max(x_image)], cmap = 'hot', origin='lower')#,norm=colors.PowerNorm(gamma=0.5))#, norm=colors.PowerNorm(gamma=1.5))
    axes[0].set_xlabel(r'$\alpha$ [mas]',fontsize = 12)
    axes[0].set_ylabel(r'$\delta$ [mas]',fontsize = 12)
    axes[0].text(max(x_image)*40/100, max(x_image)*80/100, r'$\mu = %.1E $'%hyperparameter, color='white')
    axes[0].text(min(x_image)+max(x_image)*10/100, max(x_image)*80/100, r'$\chi^2 = %.2E, %s $'%(chi2, note), color='white')                        
    axes[0].minorticks_on()
    axes[0].tick_params(axis='x', labelsize=13)
    axes[0].tick_params(axis='y', labelsize=13)
    axes[0].set_xlim([min(x_image),max(x_image)])
    axes[0].set_ylim([min(x_image),max(x_image)])

    axes[2].errorbar(q_MiRA, V2_MATISSE, V2_MATISSE_ERR, label='Observations', alpha=0.05, c='tab:blue',  fmt='o', ecolor='red', ms=2)
    axes[2].scatter(q_MiRA, V2_MiRA, s=3, label = 'MiRA', c='tab:orange')
    axes[2].set_xlabel(r'B/$\lambda$ [rad$^-1$]',fontsize = 12)
    axes[2].set_ylabel(r'V$^2$',fontsize = 12)
    axes[2].legend()
    plt.tight_layout()

    axes[3].errorbar(q_MiRA, V2_MATISSE, V2_MATISSE_ERR, label='Observations', alpha=0.05, c='tab:blue',  fmt='o', ecolor='red', ms=2)
    axes[3].scatter(q_MiRA, V2_MiRA, s=3, label = 'MiRA', c='tab:orange')
    axes[3].set_xlabel(r'B/$\lambda$ [rad$^-1$]',fontsize = 12)
    axes[3].set_ylabel(r'V$^2$',fontsize = 12)
    axes[3].set_yscale('log')
    axes[3].legend()
    plt.tight_layout()

    axes[4].scatter(q_CP,np.array(CP_MATISSE), s=2, label='Observation', alpha=0.1)
    axes[4].scatter(q_CP,np.array(CP_MiRA), s=2, label='MiRA')
    axes[4].set_xlabel(r'B/$\lambda$ [rad$^-1$]',fontsize = 12)
    axes[4].set_ylabel('CP',fontsize = 12)
    axes[4].legend()
    plt.tight_layout()

    axes[5].scatter(q_MiRA, (V2_MATISSE-V2_MiRA)/V2_MATISSE_ERR, s=3)
    axes[5].set_xlabel(r'B/$\lambda$ [rad$^-1$]',fontsize = 12)
    axes[5].set_ylabel(r'V2 Pearson residuals',fontsize = 12)
    plt.tight_layout()

    FWHM_pixels = obs_res / pixelsize
    
    # Calcul de sigma en pixels
    sigma_pixels = FWHM_pixels / 2.35482
    
    
    image_conv = gaussian_filter(image, sigma=sigma_pixels)

    axes[1].imshow(image_conv,extent=[min(x_image),max(x_image),min(x_image),max(x_image)], cmap = 'hot',  origin='lower')#,norm=colors.PowerNorm(gamma=0.5))#, norm=colors.PowerNorm(gamma=1.5))
    
    axes[1].set_xlabel(r'$\alpha$ [mas]',fontsize = 12)
    axes[1].set_ylabel(r'$\delta$ [mas]',fontsize = 12)
    axes[1].text(max(x_image)*40/100, max(x_image)*80/100, r'$\mu = %.1E $'%hyperparameter, color='white')
    axes[1].text(min(x_image)+max(x_image)*10/100, max(x_image)*80/100, r'$\chi^2 = %.2E, %s $'%(hdul[2].header['CHISQ'], note), color='white')
    

    axes[1].minorticks_on()
    axes[1].tick_params(axis='x', labelsize=13)
    axes[1].tick_params(axis='y', labelsize=13)
    axes[1].set_xlim([min(x_image),max(x_image)])
    axes[1].set_ylim([min(x_image),max(x_image)])
    plt.tight_layout()
    
    #For Astropy Version < 6.0 fixed a bug
    
    with fits.open(path_file, mode='update', ignore_missing_end=True) as hdul:

        header = hdul[0].header.copy()
        
        hdul[3].header.insert(5, ('PCOUNT', 0))
        hdul[3].header.insert(6, ('GCOUNT', 1))                     
  
        hdul[4].header.insert(5, ('PCOUNT', 0))
        hdul[4].header.insert(6, ('GCOUNT', 1))
  
        hdul[5].header.insert(5, ('PCOUNT', 0))
        hdul[5].header.insert(6, ('GCOUNT', 1))
        

        data = image_conv  
        new_image_hdu = fits.ImageHDU(data, header=header, name='CONVOLVED_IMAGE')

        hdul.append(new_image_hdu)


    column_titles = ["Image", "Image Convolved", 'V2', 'V2 log', 'CP', 'V2 residuals']

    for c in range(len(column_titles)):
      axes[c].set_title(column_titles[c], fontsize=20)

    
    plt.subplots_adjust(left=0.4, right=0.99, top=0.92, bottom=0.4, wspace=0.1, hspace=0.2)
    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.96])
    fig1.suptitle("reg_%s_FoV_%.1f_pixsize_%.1f_param_%.1f_mu_%.2E_resampled"%(regularization,FoV, pixelsize, param, hyperparameter), fontsize=16)

    
    # Convert the figure to a NumPy array
    canvas = FigureCanvas(fig1)
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.close(fig1)            
    
    return path_file, image_array, chi2




def parallel_resampling(data_args):
    return run_MiRA(*data_args)

def resampling_by_MiRA_parallelize(path_data, path_filtered, path_MiRA, min_FoV, param, hyperparameter, pixelsize, num_cores, maxeval):
    
    new_path_all, image_all, chi2_all = [], [], []
    parent_path = os.path.dirname(path_MiRA.rstrip('/'))
    regularization = path_MiRA.split('/')[-2]  # Dernier élément non vide avant le "/"
    path_save = parent_path + '/' + regularization + '_resampled/' 

    DATA_OBS         = OIFITS_READING_concatenate(path_data)
    obs_res       =  rad_to_mas(0.5/(max(max(abs(DATA_OBS['VIS2']['U'])/DATA_OBS['VIS2']['WAVEL']),max(abs(DATA_OBS['VIS2']['V'])/DATA_OBS['VIS2']['WAVEL']))))

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    # Préparation des données pour chaque tâche
    data = [
        (DATA_OBS, path_data, path_filtered[i], pixelsize, min_FoV, hyperparameter[i], param[i],  regularization, path_save, maxeval, num_cores, obs_res)
        for i in range(len(path_filtered))
    ]
        
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:  # Ajustez max_workers
        future_results = [executor.submit(parallel_resampling, args) for args in data]
    
        for future in concurrent.futures.as_completed(future_results):
            try:
                new_path, image, chi2 = future.result()
                new_path_all.append(new_path)
                image_all.append(image)
                chi2_all.append(chi2)
            except Exception as e:
                print(f"Error in task: {e}")
    
    
    new_path_all = np.array(new_path_all)
    
    # Associer chaque image à son chi²
    images_with_chi2 = list(zip(image_all, chi2_all))

    # Trier les images par chi² croissant
    images_with_chi2.sort(key=lambda x: x[1])  # Trie par chi²

    # Extraire les images triées
    sorted_images = [Image.fromarray(np.array(img[0])) for img in images_with_chi2]

    # Fonction pour ajuster le canvas de chaque image
    def adjust_canvas_to_image(img):
        # Créer une nouvelle image avec la taille exacte de l'image originale
        canvas = Image.new("RGB", img.size, (255, 255, 255))  # Fond blanc
        canvas.paste(img, (0, 0))
        return canvas

    # Ajuster le canvas de chaque image
    adjusted_images = [adjust_canvas_to_image(img) for img in sorted_images]

    # Enregistrer le PDF en s'assurant que chaque image occupe toute la page
    if adjusted_images:
        adjusted_images[0].save(
            path_save+"all_images.pdf",
            save_all=True,
            append_images=adjusted_images[1:],
            resolution=150.0,  # Résolution standard pour PDF
            quality=75         # Haute qualité pour JPEG
        )
        print("PDF generated : all_outputs.pdf")
    else:
        print("No image available to generate the PDF.")

    return new_path_all
