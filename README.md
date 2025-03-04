# MYTHRA: Mean Astrophysical Images with PYRA

**MYTHRA** is a Python code and extension of **PYRA** used to create a mean image from all the MiRA outputs generated by PYRA.
**PYRA** (https://github.com/jdrevon/PYRA) is a Python wrapper designed to facilitate and optimize the use of **MiRA** (Multi-aperture Image Reconstruction Algorithm), developed by the [JMMC](https://github.com/emmt/MiRA?tab=readme-ov-file). This tool enables efficient and random scanning of the parameter space for image reconstruction. 


**Requirements:**
- MiRA
- Python 3.8+
- Linux/Ubuntu environment

---

## Overview
The purpose of **MYTHRA** is to create a mean image and the associated error maps.


## How to Use

### Main Script: `after_reconstruction2_v2.py`
The primary script contains variables you need to configure before running. Below is a list of key variables and their purposes:

| **Variable**             | **Description**                                                                                                                                                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `path_MiRA`              | Full path to the regularization directory where the PYRA outputs are stored.                                                                                                                                                                |                                                                              
| `path_data`              | Full path to the `.fits` file containing observations. If multiple `.fits` files exist, merge them before running the script (e.g., using Python or [OIFits Explorer](https://www.jmmc.fr/)).                                               |
| `num_cores`              | Number of CPU cores dedicated to the routine.                                                                                                                                                                                               |
| `bin_num`                | Number of bins used to model the l-curve trend for the script to spot the divergence point. Needs to be adjusted regarding the shape of the l-curve provided by PYRA and the number of image reconstructed (default for 1000 images and a clean l-curve: 10).|
| `negative_offset`        | Purcentage used to shift the detected divergence point to a lower value in the logarithmic scale (default: 0.1)                                                                                                                                                            |
| `window`                 | Width of the window around the divergence point to select data. (default: 0.04). You can increase or decrease this value if you want to include more data. A plot with the data taken is saved inside path_MiRA.|     
| `chi2_boundaries`        | True or False, set a threshol constraint on the chi2 for the selection of the datasubset on the l-curve. This parameter is useful in case you do not have a clean l-curve with a lot of outlayers to make also a chi2 selection on the good data you want to keep. |
| `chi2_boundaries_up`     | Superior threshold on the chi2 values that will define the best data subset. |
| `chi2_boundaries_down`   | Inferior threshold on the chi2 values that will define the best data subset. |
| `maxeval`                | Maximum number of evaluations for MiRA to stop if convergence is not achieved (default: 50). This parameter is used to operate a common resampling of the pixelsize for all the images using MiRA.                                          |
| `chi2_treshold`          | Threshold on the chi2 of the data-subset. In case the median chi2 of the datasubset selected from the l-curve is below the chi2_threshold, the routine use the function optimize_centring2.py to optimize the chi2 of the final image. The images are rejected if the image increases the chi2 of the mean image above the thresold chi2_mean_V2 for the chi2 on the squared visibilities and chi2_mean_CP for the closure phase. 
| `chi2_mean_V2`           | Threshold on the visibility chi2 of the mean image used to select the outlayers from the image reconstructions subset in case the median chi2 of the datasubset is below chi2_threshold.                                                                                                                                                    |
| `chi2_mean_CP`           | Threshold on the closure phase chi2 used to select the outlayers from the image reconstructions subset in case the median chi2 of the datasubset is below chi2_threshold. |
---

### Running the Script
1. Configure the above variables in the script.
2. Run the script:
 ```bash
 python3 after_reconstruction2_v2.py
```
## Outputs
Results are stored directly in path_MiRA.
The user is also able to check the results from the resampling by MiRA which are stored in the same main folder than the path_MiRA folder.
