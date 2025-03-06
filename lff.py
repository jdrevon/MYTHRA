import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits

rad2as = 206265

def load_fits_data(dir, filename):
    hdu = fits.open(dir + filename)
    binary_table_names = np.array([hdi.name for hdi in hdu if isinstance(hdi, fits.BinTableHDU)])
    return hdu, binary_table_names

######################################################

def extract_vis2_data(hdu, binary_table_names):
    OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid = (np.array([]) for _ in range(9))
    count = 0
    for ibin, bnam in enumerate(binary_table_names):
        if bnam == "OI_VIS2":
            oivis2 = hdu[ibin+1]
            vis2, vis2e = oivis2.data['VIS2DATA'], oivis2.data['VIS2ERR']
            u, v = hdu[ibin+1].data['UCOORD'], hdu[ibin+1].data['VCOORD']
            instru = hdu[ibin+1].header['INSNAME']
            Wavelength, Bandwidth = find_wavelength_table(hdu, binary_table_names, instru)
            b = np.sqrt(u**2 + v**2)
            freq = b[:,None]/Wavelength[None,:] / rad2as
            for i in range(vis2.shape[0]):
                count += 1
                wlen, band = np.append(wlen, Wavelength), np.append(band, Bandwidth)
                OIVIS2, OIVIS2e = np.append(OIVIS2, vis2[i,:]), np.append(OIVIS2e, vis2e[i,:])
                FREQ, U, V, B, Bid = (np.append(arr, np.ones(Wavelength.shape[0])*val) for arr, val in zip([FREQ, U, V, B, Bid], [freq[i,:], u[i], v[i], b[i], count]))
    return OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid, count

######################################################

def find_wavelength_table(hdu, binary_table_names, instru):
    for wbin, wnam in enumerate(binary_table_names):
        oiwlen = hdu[wbin+1]
        if wnam == "OI_WAVELENGTH" and oiwlen.header['INSNAME'] == instru:
            return hdu[wbin+1].data['EFF_WAVE'], hdu[wbin+1].data['EFF_BAND']
    raise ValueError(f"OI_WAVELENGTH table not found for instrument {instru}")

######################################################

def plot_vis2_data(freq, vis2, vis2e, color="blue", label=None):
    for ibase in range(vis2.shape[0]):
        plt.errorbar(freq[ibase], vis2[ibase], yerr=vis2e[ibase], color=color, label=label)

######################################################

def filter_visibilities(OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, v2thresh):
    flag = (OIVIS2 - OIVIS2e > v2thresh) & (B < 5 * min(B))
    return (arr[flag] for arr in [OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band])

######################################################

def sort_visibilities(freqfl, Vis2fl, Vis2fle):
    sorted_indices = np.argsort(freqfl)
    return (arr[sorted_indices] for arr in [freqfl, Vis2fl, Vis2fle])

######################################################

def gaussian_centered(x, sigma, a=1, x0=0):
    return a * np.exp(-(x - x0)**2 / (2 * (sigma+(sigma==0))**2))

######################################################

def plot_gaussian_fit(freq_wl, vis2_wl, vis2e_wl, popt, wl, colors, unique_wavelengths):
    x_fit = np.linspace(0, max(freq_wl), 1000)
    y_fit = gaussian_centered(x_fit, *popt)
    plt.plot(x_fit, y_fit, linestyle='--', label=f'Fit for wl={wl:.2e}', color=colors[np.where(unique_wavelengths == wl)[0][0]])
    plt.legend()

######################################################

def gaussian_2d(xy, sigma_x, sigma_y, theta):
    x, y = xy
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    return np.exp(-((x_rot)**2 / (2 * sigma_x**2) + (y_rot)**2 / (2 * sigma_y**2)))

######################################################

def plot_gaussian_2d_fit(freq_wl, vis2_wl, vis2e_wl, popt, wl, colors, unique_wavelengths):
    x_fit = np.linspace(0, max(freq_wl), 1000)
    y_fit = gaussian_centered(x_fit, *popt)
    plt.plot(x_fit, y_fit, linestyle='--', label=f'Fit for wl={wl:.2e}', color=colors[np.where(unique_wavelengths == wl)[0][0]])
    plt.legend()

######################################################

def main():
    #dir = '/Users/fmillour/Documents/ARTICLES/HD62623_3/lpup_LMdata_cal_Chop_2020&2024/'
    #filename = 'OiXP_l_Pup_MATISSE_A0-B2-C1-D0_2020-02-16_L.fits'
    #filename = 'OiXP_l_Pup_MATISSE_A0-B2-C1-D0_2020-02-16_M.fits'
    #filename = 'OiXP_l_Pup_MATISSE_A0-B2-C1-D0_2020-02-16_L.fits'
    
    dir = '/Users/fmillour/Documents/ARTICLES/HD62623_3/Re_LFF_l_Pup/'
    filename = 'l_Pup_newNband_MATISSE_IR-N_LOW_noChop_cal_merged_oifits_0.fits'
    #filename = 'l_Pup_newLband_MATISSE_IR-LM_LOW_noChop_cal_merged_oifits_0.fits'
    #filename = 'l_Pup_newMband_MATISSE_IR-LM_LOW_noChop_cal_merged_oifits_0.fits'

    v2thresh = 0.3**2
    csym = 1

    hdu, binary_table_names = load_fits_data(dir, filename)
    OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, Bid, count = extract_vis2_data(hdu, binary_table_names)
    print(f'Data file contained {count} visibilities')
    
    plt.figure(figsize=(12, 9))
    plot_vis2_data(FREQ, OIVIS2, OIVIS2e)

    Vis2fl, Vis2fle, freqfl, Ufl, Vfl, Bfl, wlenfl, bandfl = filter_visibilities(OIVIS2, OIVIS2e, FREQ, U, V, B, wlen, band, v2thresh)
    unique_wavelengths, index = np.unique(wlenfl, return_index=True)
    unique_band = bandfl[index]
    freqfl_sorted, Vis2fl_sorted, Vis2fle_sorted = sort_visibilities(freqfl, Vis2fl, Vis2fle)
    
    print(unique_wavelengths)

    plt.axhline(y=v2thresh, color="red", linestyle=':')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_wavelengths)))
    for i, wl in enumerate(unique_wavelengths):
        mask = wlenfl == wl
        plt.errorbar(freqfl_sorted[mask], Vis2fl_sorted[mask], yerr=Vis2fle_sorted[mask], color=colors[i], label=f'wl={wl:.2e}')
    plt.legend()

    sigma_fits = []
    for wl in unique_wavelengths:
        mask = wlenfl == wl
        freq_wl, vis2_wl, vis2e_wl = freqfl_sorted[mask], Vis2fl_sorted[mask], Vis2fle_sorted[mask]
        popt, _ = curve_fit(gaussian_centered, freq_wl, vis2_wl, sigma=vis2e_wl, p0=[np.std(freq_wl)])
        sigma_fits.append(popt[0])
        plot_gaussian_fit(freq_wl, vis2_wl, vis2e_wl, popt, wl, colors, unique_wavelengths)

    plt.xlabel(r'$B / \lambda$ (cycles/arcsec)')
    plt.ylabel(r'$V^2$')
    plt.title(r'$V^2$ vs $B / \lambda$')
    plt.yscale('log')
    plt.xlim(0, 3*max(freqfl_sorted))
    plt.ylim(v2thresh/1000, 1.1)
    #output_filename = filename.replace('.fits', '_v2_fit1D.png')
    #plt.savefig(output_filename)
    #plt.show()

    # 2D Gaussian fit
    sigma_x_fits = []
    sigma_y_fits = []
    theta_fits = []
    reduced_chi2s_2d = []

    for wl in unique_wavelengths:
        mask = wlen == wl
        xy = np.vstack((U[mask] / wl, V[mask] / wl)) / rad2as
        vis2_wl  = OIVIS2[mask]
        vis2e_wl = OIVIS2e[mask]

        initial_guess = [np.std(U[mask] / wl / rad2as), np.std(V[mask] / wl / rad2as), 0.1]
        popt2, pcov2  = curve_fit(gaussian_2d, xy, vis2_wl, sigma=vis2e_wl, p0=initial_guess)
        sigma_x_fits.append(popt2[0])
        sigma_y_fits.append(popt2[1])
        theta_fits.append(popt2[2])

        # Calculate reduced chi-squared for 2D fit
        residuals_2d = vis2_wl - gaussian_2d(xy, *popt2)
        chi2_2d = np.sum((residuals_2d / vis2e_wl) ** 2)
        dof_2d = len(vis2_wl) - len(popt2)  # degrees of freedom
        reduced_chi2_2d = chi2_2d / dof_2d
        reduced_chi2s_2d.append(reduced_chi2_2d)

    print(f"Fit parameters for 2D Gaussian (sigma_x, sigma_y, theta) for each wavelength: {list(zip(sigma_x_fits, sigma_y_fits, theta_fits))}")
    print(f"Reduced chi-squared for 2D fit for each wavelength: {reduced_chi2s_2d}")

    # Generate new visibilities from the fit parameters
    num_points = 30
    max_radius = min(freqfl_sorted) * rad2as * min(unique_wavelengths) * 0.9
    U_new = []
    V_new = []
    while len(U_new) < num_points:
        r = max_radius * np.random.uniform(0, 1)
        theta = np.random.uniform(0, np.pi)
        u = r * np.cos(theta)
        v = r * np.sin(theta)
        U_new.append(u)
        V_new.append(v)
    U_new = np.array(U_new)
    V_new = np.array(V_new)

    wlen_new = unique_wavelengths
    band_new = unique_band

    new_visibilities = []
    new_B = []
    for iwl, wl in enumerate(wlen_new):
        B_new = np.sqrt(U_new**2 + V_new**2)
        new_B.append(B_new)
        freq = B_new / wl / rad2as
        if csym == 1:
            new_vis = gaussian_centered(freq, sigma_fits[iwl])
        else:
            xy = np.vstack((U_new, V_new)) / wl / rad2as
            new_vis = gaussian_2d(xy, sigma_x_fits[iwl], sigma_y_fits[iwl], theta_fits[iwl])
        new_visibilities.append(new_vis)
    new_visibilities = np.array(new_visibilities).T
    new_B = np.array(new_B).T

    new_vis2err = np.median(Vis2fle)/np.median(Vis2fl) * new_visibilities * new_B / np.max(new_B)

    for ibase in range(new_visibilities.shape[0]):
        new_freq = new_B[ibase,:] / wlen_new / rad2as
        plt.plot(new_freq, new_visibilities[ibase,:], color="black")
        plt.errorbar(new_freq, new_visibilities[ibase,:], yerr=new_vis2err[ibase,:], color="black")

    nwl = wlen_new.shape[0]
    new_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='UCOORD', format='D', array=U_new),
        fits.Column(name='VCOORD', format='D', array=V_new),
        fits.Column(name='VIS2DATA', format=f'{nwl}D', array=new_visibilities),
        fits.Column(name='FLAG', format=f'{nwl}L', array=np.zeros(new_visibilities.shape, dtype=bool)),
        fits.Column(name='INT_TIME', format='D', array=np.zeros(U_new.shape)),
        fits.Column(name='MJD', format='D', array=hdu['OI_VIS2'].data['MJD'][0]*np.ones(U_new.shape)),
        fits.Column(name='STA_INDEX', format='2I', array=np.zeros(U_new.shape, dtype=int)),
        fits.Column(name='TARGET_ID', format='I', array=np.ones(U_new.shape, dtype=int)),
        fits.Column(name='TIME', format='D', array=np.zeros(U_new.shape)),
        fits.Column(name='VIS2ERR', format=f'{nwl}D', array=new_vis2err)
    ])
    new_hdu.header['EXTNAME'] = 'OI_VIS2'
    new_hdu.header['INSNAME'] = hdu['OI_VIS2'].header['INSNAME']
    new_hdu.header['OI_REVN'] = 1
    new_hdu.header['DATE-OBS'] = '2023-01-01T00:00:00.000'
    new_hdu.data['STA_INDEX'][:,1] = 1

    new_hdul = fits.HDUList([fits.PrimaryHDU(), new_hdu])

    wavelength_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='EFF_WAVE', format='E', array=wlen_new),
        fits.Column(name='EFF_BAND', format='E', array=band_new)
    ])
    wavelength_hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
    wavelength_hdu.header['INSNAME'] = hdu['OI_WAVELENGTH'].header['INSNAME']
    wavelength_hdu.header['OI_REVN'] = 1

    new_hdul.append(wavelength_hdu)

    for hdi in hdu:
        if 'EXTNAME' in hdi.header and hdi.header['EXTNAME'] == 'OI_ARRAY':
            new_hdul.append(hdi)
            new_hdul[-1].header['OI_REVN'] = 1
    for hdi in hdu:
        if 'EXTNAME' in hdi.header and hdi.header['EXTNAME'] == 'OI_TARGET':
            new_hdul.append(hdi)
            new_hdul[-1].header['OI_REVN'] = 1
            break

    if csym == 1:
        new_filename = filename.replace('.fits', '_new_v2_fit1D_LFF.fits')
        output_filename = filename.replace('.fits', '_v2_fit1D.png')
    else:
        new_filename = filename.replace('.fits', '_new_v2_fit2D_LFF.fits')
        output_filename = filename.replace('.fits', '_v2_fit2D.png')
    new_hdul.writeto(dir + new_filename, overwrite=True)

    plt.savefig(dir + output_filename)
    plt.show()

if __name__ == "__main__":
    main()
