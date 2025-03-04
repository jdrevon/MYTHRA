# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:34:53 2024

@author: jdrevon
"""

import os
import re
from astropy.io import fits

# Extraction of the values from the data filename
def extract_values_from_filename(filename):
    pattern = r"reg_(\w+)_FoV_(\d+\.\d+)_pixsize_(\d+\.\d+)_param_(\d+\.\d+)_mu_(\d+\.?\d*E?[+-]?\d*)"
    match = re.search(pattern, filename)
    
    if match == None:
      pattern = r"reg_(\w+)_FoV_(\d+\.\d+)_pixsize_(\d+\.\d+)_param_(\d+\.?\d*E?[+-]?\d*)_mu_(\d+\.?\d*E?[+-]?\d*)"
      match = re.search(pattern, filename)
    
    
    if match:
        return {
            "reg": match.group(1),
            "FoV": float(match.group(2)),
            "pixsize": float(match.group(3)),
            "param": float(match.group(4)),
            "mu": float(match.group(5))
        }
    return None

# Function to dig into the directories and extract the info from the .fits files
def process_fits_files(directory, headers_data):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".fits"):
                filepath = os.path.join(root, file)
                
                # Extraction des valeurs du nom du fichier
                extracted_values = extract_values_from_filename(file)
                
                if extracted_values:
                    headers_data["reg"].append(extracted_values["reg"])
                    headers_data["FoV"].append(extracted_values["FoV"])
                    headers_data["pixsize"].append(extracted_values["pixsize"])
                    headers_data["param"].append(extracted_values["param"])
                    headers_data["mu"].append(extracted_values["mu"])
                    headers_data["filepath"].append(filepath)  # Chemin complet du fichier

                    # Lecture des informations du header
                    with fits.open(filepath) as hdul:
                        # Lecture de chi2 depuis le header de l'extension 2
                        if len(hdul) > 2 and hdul[2].header.get("EXTNAME") == "IMAGE-OI OUTPUT PARAM":
                            chi2_value = hdul[2].header.get("CHISQ", None)
                            headers_data["chi2"].append(chi2_value)
                        else:
                            headers_data["chi2"].append(None)  # Aucun chi2 trouvé

                        if len(hdul) > 2 and hdul[2].header.get("EXTNAME") == "IMAGE-OI OUTPUT PARAM":
                            chi2_value = hdul[2].header.get("NEVAL", None)
                            headers_data["NEVAL"].append(chi2_value)
                        else:
                            headers_data["NEVAL"].append(None)  # Aucun chi2 trouvé

                        if len(hdul) > 2 and hdul[2].header.get("EXTNAME") == "IMAGE-OI OUTPUT PARAM":
                            chi2_value = hdul[2].header.get("CONVERGE", None)
                            headers_data["CONVERGE"].append(chi2_value)
                        else:
                            headers_data["CONVERGE"].append(None)  # Aucun chi2 trouvé

                        if len(hdul) > 2 and hdul[2].header.get("EXTNAME") == "IMAGE-OI OUTPUT PARAM":
                            chi2_value = hdul[2].header.get("GPNORM", None)
                            headers_data["GPNORM"].append(chi2_value)
                        else:
                            headers_data["GPNORM"].append(None)  # Aucun chi2 trouvé


                        # Lecture et stockage du header de la première table
                        if len(hdul) > 1:
                            first_header = dict(hdul[0].header)  # Conversion en dictionnaire pour un accès plus facile
                            headers_data["first_header"].append(first_header)
                        else:
                            headers_data["first_header"].append(None)  # Aucun header trouvé pour la première table


def selected_data(directory, chi2_boundaries, chi2_boundaries_up, chi2_boundaries_down, num_bins, offset_percentage, factor, threshold=None):
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # Initialisation du dictionnaire des données
    headers_data = {"reg": [], "GPNORM": [], "CONVERGE":[], "NEVAL":[], "FoV": [], "pixsize": [], "mu": [], "param": [], "chi2": [], "filepath": [], "first_header": []}
    process_fits_files(directory, headers_data)

    # Charger les données
    mu = np.array(headers_data["mu"], dtype=float)
    chi2 = np.array(headers_data["chi2"], dtype=float)
    pixsize = np.array(headers_data["pixsize"], dtype=float)
    path = np.array(headers_data["filepath"], dtype=str)
    FoV = np.array(headers_data["FoV"], dtype=float)
    first_header = np.array(headers_data["first_header"])
    gpnorm = np.array(headers_data["GPNORM"])
    converge = np.array(headers_data["CONVERGE"]) 
    param = np.array(headers_data["param"]) 
    
    
    
    # Tri des données
    sorted_indices = np.argsort(mu)
    mu_sorted = mu[sorted_indices]*pixsize[sorted_indices]**2
    chi2_sorted = chi2[sorted_indices]

    # Création des bins sur l'axe mu
    mu_bins = np.logspace(np.log10(mu_sorted.min()), np.log10(mu_sorted.max()), num_bins)
    mu_bin_centers = (mu_bins[:-1] + mu_bins[1:]) / 2

    # Calcul de la médiane du chi2
    chi2_median = []
    for i in range(len(mu_bins) - 1): 
        mask = (mu_sorted >= mu_bins[i]) & (mu_sorted < mu_bins[i + 1])
        if np.any(mask):
            chi2_median.append(np.nanmedian(chi2_sorted[mask]))
        else:
            chi2_median.append(np.nan)
    chi2_median = np.array(chi2_median)

    # Suppression des valeurs NaN
    valid_indices = ~np.isnan(chi2_median)
    mu_bin_centers_valid = mu_bin_centers[valid_indices]
    chi2_median_valid = chi2_median[valid_indices]

    # Calcul du gradient en échelle logarithmique
    dlog_chi2_dlog_mu = np.gradient(np.log(chi2_median_valid), np.log(mu_bin_centers_valid))

    # Ajustement du seuil
    if threshold is None:
        # Calcul adaptatif basé sur les percentiles
        threshold = np.percentile(dlog_chi2_dlog_mu, 80)
    print(f"Threshold utilisé : {threshold}")

    # Détection de divergence
    divergence_index = np.where(dlog_chi2_dlog_mu > threshold)[0]
    # print(threshold)
    # print(dlog_chi2_dlog_mu)
    if divergence_index.size > 0:
        if divergence_index[0] > 0:
            divergence_index = divergence_index[0] - 1
            mu_divergence = mu_bin_centers_valid[divergence_index]
            chi2_divergence = chi2_median_valid[divergence_index]
        else:
            mu_divergence = None
            chi2_divergence = None
    else:
        mu_divergence = None
        chi2_divergence = None
    
    # Plot the initila curve, the median one and the divergence point
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(mu*pixsize**2, chi2, c=pixsize, cmap="rainbow", edgecolor="k", s=50, label="Data")
    plt.plot(mu_bin_centers, chi2_median, color="black", linewidth=2, label="Median curve")
    
    # Hilghlight the divergence point
    
    if mu_divergence is not None:
        plt.scatter(mu_divergence, chi2_divergence, color="red", label="Point of divergence", s=100, zorder=5)
        plt.axvline(mu_divergence, color="red", linestyle="--", linewidth=1)
        plt.axhline(chi2_divergence, color="red", linestyle="--", linewidth=1)
    
    plt.colorbar(sc, label="Pixsize")
    
    # Add labels etc...
    plt.xlabel("Mu")
    plt.ylabel("Chi2")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    figure_path = os.path.join(directory, "figure_log_tot.png")  # Nom du fichier
    plt.savefig(figure_path, bbox_inches='tight')  # Sauvegarde avec ajustement des bordures
    
    plt.close()
    
    if mu_divergence is not None:
        point_divergence_mu = mu_divergence
    else:
        point_divergence_mu = np.median(mu)  # If no divergence use the median
    
    # Offset value regarding the divergence point
    # offset_percentage = 0.05  
    
    # Compute the new divergence point with the offset using log10
    log10_offset = np.log10(point_divergence_mu) * (1 - offset_percentage)
    point_central_mu = 10 ** log10_offset  # Using base 10 exponentiation
    
    # Define the window around the offseted divergence point in which we keep the data 
    # factor = 0.060  # 6 % seems to be fine
    
    # Compute the logarithmic boundaries using log10
    log10_mu = np.log10(point_central_mu)
    lower_limit_log10 = log10_mu - factor * np.abs(log10_mu)  # Lower limit
    upper_limit_log10 = log10_mu + factor * np.abs(log10_mu)  # Upper limit
    
    # Convert from log10 to linear
    lower_limit = 10 ** lower_limit_log10
    upper_limit = 10 ** upper_limit_log10
    
    # Add the convergence constraints:
    gpnorm = np.array(headers_data["GPNORM"])
    converge = np.array(headers_data["CONVERGE"]) 

    #GREEN if converge, if num eval < max eval -1 , and if GPNORM < 1 
    note = []

    for i in range(len(gpnorm)):

        if np.logical_and(converge[i] == True, gpnorm[i]<1) == True:

            note.append('C')

        #ORANGE, WARNING if converge and if num eval >= max eval -1 or if GPNORM > 1 

        elif np.logical_and(converge[i] == True, gpnorm[i]>1) == True:

            note.append('W')

        #RED, non-converged if converge and if num eval >= max eval -1 or if GPNORM > 1 

        elif converge[i] == False :

            note.append('NC')
    
    note = np.array(note)
    # print(note=='C')
    # Filter the data
    
    if chi2_boundaries == True:
        mask = (mu*pixsize**2 >= lower_limit) & (mu*pixsize**2 <= upper_limit) & (chi2 <chi2_boundaries_up) & (chi2 > chi2_boundaries_down) 
    else:
        mask = (mu*pixsize**2 >= lower_limit) & (mu*pixsize**2 <= upper_limit)  
    mu_filtered = mu[mask]
    chi2_filtered = chi2[mask]
    pixsize_filtered = pixsize[mask]
    path_filtered = path[mask]
    FoV_filtered  = FoV[mask]
    param_filtered  = param[mask]
    first_header_filtered = first_header[mask]
    # New figure with the filtered data highlited
    plt.figure(figsize=(10, 6))
    
    plt.scatter(mu*pixsize**2, chi2, color='gray', alpha=0.5, edgecolor='k', s=50, label="Non-selected data")
    sc = plt.scatter(mu_filtered*pixsize_filtered**2, chi2_filtered, c=pixsize_filtered, cmap="rainbow", edgecolor='k', s=50, label="Filtered data")
    
    plt.axvline(point_divergence_mu, color="red", linestyle="--", linewidth=1, label="Point of divergence")
    plt.axvline(point_central_mu, color="blue", linestyle="--", linewidth=1, label="Central point with offset")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mu")
    plt.ylabel("Chi2")
    plt.legend()
    plt.colorbar(sc, label="Pixsize")
    plt.grid(True)
    plt.show()
    
    # Save figure
    figure_path = os.path.join(directory, "figure_log_region.png")  # File name
    plt.savefig(figure_path, bbox_inches='tight')  # Save
    plt.close()
    return path_filtered, FoV_filtered, pixsize_filtered, first_header_filtered, chi2_filtered, param_filtered, mu_filtered
