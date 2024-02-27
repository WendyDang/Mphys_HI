#This code will generate all spectrum in the list 
import numpy as np
import matplotlib.pyplot as plt
import radio_beam
from radio_beam import Beam
from spectral_cube import SpectralCube
from matplotlib import cm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from matplotlib.patches import Ellipse
from astropy import units as u 
from astropy import coordinates
from astropy.utils import data
from astropy.cosmology import LambdaCDM
#import pylab as pl
from astropy.io import ascii
from matplotlib import rcParams
#import aplpy
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mtick
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm import tqdm

#read the catalogue ---------------------------------------
tab1 = '~/Desktop/HI/COSMOS_32k_asy.txt'
# Reading the data
data1 = pd.read_csv(tab1, comment='#', sep='\t')
# Setting the index of data1 to the 'Name' column
data1['Name'] = data1['Name_copy']
data1.set_index('Name', inplace=True)


#constants ----------------------------------------------
restfr=1420.405*10**6 #HI rest frequency in HZ 
H0=67.5 #Hubble constant
c=299792 #speed of light
sigma=6.8e-05
#list_sigma=[]

def generate_spectrum(cube_name):
    freq=data1.loc[cube_name, 'freq']*10**9 # central frequency of the source in the cubelet
    cubelet=SpectralCube.read('/Users/apple/Documents/GitHub/Mphys_HI/COSMOS_r0p5/'+str(cube_name)+'.r0p5.fits')
    bmaj=cubelet.header['BMAJ']*3600 # values in the header are in degrees, so we need to convert them to arcsec
    bmin=cubelet.header['BMIN']*3600
    bpa=cubelet.header['BPA']
    dpix=cubelet.header['CDELT2']*3600 #amount of arcsec per pixel
    df=cubelet.header['CDELT3'] #hz

    #Parameters to convert your flux from Jy/beam to Jy
    omega = (np.pi*(bmaj/206265)*(bmin/206265))/(4*np.log(2))*4.25*10**10 # beam area in arcsec
    bmpix=dpix**2/omega # amount of pixels per beam area- convertion factor, we will see it used later
    #Formulas from the lecture
    V=(restfr**2-freq**2)/(restfr**2+freq**2)*c # systemic velocity
    dist=V/H0           #distance Mpc
    z=(restfr-freq)/freq # redshift

    #Convert cahnnel width from HZ to km/s
    dv=(((restfr**2-(freq-df)**2)/(restfr**2+(freq-df)**2)*c))-V #channel width in km/s

    cubelet = cubelet.with_spectral_unit(u.km/u.s,
                                       velocity_convention='radio',
                                       rest_value=1.420405*u.GHz)
    
    #Convolve with new beam---------------------------------------
    # we define the new beam, which is 20 x 20 arcsec
    # I didn't convolve here because of ValueError: Beam could not be deconvolved
    beam = radio_beam.Beam(major=bmaj*u.arcsec, minor=bmin*u.arcsec, pa=0*u.deg)
    #then we convolve our original cubelet with the new beam
    cube_20 = cubelet.convolve_to(beam)

    #create mask cube -------------------------------------------
    subcube_20=cube_20[:,0:15,0:15]

    rms = subcube_20.std(axis=(1,2))
    sigma=np.mean(rms.value)
    include_mask = cube_20 > 3*sigma*cube_20.unit

    mask_cube = cube_20.with_mask(include_mask)
    #you can save the mask and examine it in ds9 
    print(1)
    mask_cube.write('/Users/apple/Documents/GitHub/Mphys_HI/COSMOS_r0p5/'+str(cube_name)+'_revised_masked_cube.fits', format='fits',overwrite='True') 

    #generate spectrum -----------------------------------------
    spectrum_new_jy = mask_cube.sum(axis=(1,2))*bmpix
    spectrum_new_jy_converted = np.nan_to_num(spectrum_new_jy.value)
    v_arr1 = cubelet.spectral_axis.value

    #calculate asymmetry and store --------------------------------------
    left_sum=0
    right_sum=0
    spec_center=int(data1.loc[cube_name, 'spec_center'])
    i_begin = 0
    i_mid = int(len(v_arr1)/2)-spec_center
    i_end = int(len(v_arr1))
    

    for i in range(i_begin,i_mid):
        left_sum=spectrum_new_jy_converted[i]+left_sum
    for i in range(i_mid,i_end):    
        right_sum=spectrum_new_jy_converted[i]+right_sum

        
    asymmetry = round(left_sum/right_sum,2)
    #data1.loc[cube_name, 'asy'] = asymmetry
    #data1.to_csv("~/Desktop/HI/COSMOS_32k_asy.txt", index=False, sep='\t')

    #draw spectrum -------------------------------------------
    
    plt.figure(figsize=(9,6))
    plt.plot(v_arr1, spectrum_new_jy_converted)
    #plt.plot(v_arr1, spectrum_jy, label='Spectrum_unmasked')
    #plt.plot([800, 25700], [0, 0],'k--')
    plt.plot(v_arr1[i_mid], spectrum_new_jy_converted[i_mid], 'r|', markersize=50)

    plt.xlabel('Vel (km/s)', fontsize = 12)
    plt.ylabel('Jy', fontsize = 12)

    x_pos = 0.97 * (v_arr1[72] - v_arr1[0]) + v_arr1[0]  # 10% into the x-range
    y_pos = 0.95 * (max(spectrum_new_jy_converted) - min(spectrum_new_jy_converted))  # 10% into the y-range

    x_pos_ = 0.97 * (v_arr1[72] - v_arr1[0]) + v_arr1[0]  # 10% into the x-range
    y_pos_ = 0.85 * (max(spectrum_new_jy_converted) - min(spectrum_new_jy_converted))  # 10% into the y-range

    plt.text(x_pos, y_pos, 'Asymmetry ' +str(asymmetry), fontsize=14, bbox=dict(facecolor='white', alpha=0.5))


    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().tick_params(axis='both', which='major', direction='in', length=10, top=True, right=True)
    plt.gca().tick_params(axis='both', which='minor', direction='in', length=5, top=True, right=True)

    #plt.legend(loc=2, numpoints=1, prop={'size':12})

    plt.savefig('/Users/apple/Documents/GitHub/Mphys_HI/COSMOS_r0p5_spectrum/'+str(cube_name)+'_spectrum.pdf',format = 'pdf', bbox_inches = 'tight', transparent=True)

#to be finished difficulty:  different central frequencies for a list of cubes, need to be checked by eye
#check whether we want to make this result first
def generate_Al_from_mask(cube_name):
    freq=data1.loc[cube_name, 'freq']*10**9 # central frequency of the source in the cubelet
    cubelet=SpectralCube.read('~/Desktop/HI/COSMOS_r0p5/'+str(cube_name)+'_revised_masked_cube.fits')
    bmaj=cubelet.header['BMAJ']*3600 # values in the header are in degrees, so we need to convert them to arcsec
    bmin=cubelet.header['BMIN']*3600
    bpa=cubelet.header['BPA']
    dpix=cubelet.header['CDELT2']*3600 #amount of arcsec per pixel
    df=cubelet.header['CDELT3'] #hz

    #Parameters to convert your flux from Jy/beam to Jy
    omega = (np.pi*(bmaj/206265)*(bmin/206265))/(4*np.log(2))*4.25*10**10 # beam area in arcsec
    bmpix=dpix**2/omega # amount of pixels per beam area- convertion factor, we will see it used later
    #Formulas from the lecture
    V=(restfr**2-freq**2)/(restfr**2+freq**2)*c # systemic velocity
    dist=V/H0           #distance Mpc
    z=(restfr-freq)/freq # redshift

    #Convert cahnnel width from HZ to km/s
    dv=(((restfr**2-(freq-df)**2)/(restfr**2+(freq-df)**2)*c))-V #channel width in km/s

    mask_cube = cubelet.with_spectral_unit(u.km/u.s,
                                       velocity_convention='radio',
                                       rest_value=1.420405*u.GHz)
    
    #generate spectrum -----------------------------------------
    spectrum_new_jy = mask_cube.sum(axis=(1,2))*bmpix
    spectrum_new_jy_converted = np.nan_to_num(spectrum_new_jy.value)
    v_arr1 = cubelet.spectral_axis.value

    #calculate asymmetry and store --------------------------------------
    left_sum=0
    right_sum=0

    i_begin = 0
    i_mid = int(len(v_arr1)/2)+1
    i_end = int(len(v_arr1))
    

    for i in range(i_begin,i_mid):
        left_sum=spectrum_new_jy_converted[i]+left_sum
    for i in range(i_mid,i_end):    
        right_sum=spectrum_new_jy_converted[i]+right_sum

        
    asymmetry = round(left_sum/right_sum,2)
    data1.loc[cube_name, 'asy'] = asymmetry
    data1.to_csv("~/Desktop/HI/COSMOS_32k_asy.txt", index=False, sep='\t')


def generate_morphology(cube_name):

    freq=data1.loc[cube_name, 'freq']*10**9 # central frequency of the source in the cubelet
    cubelet=SpectralCube.read('~/Desktop/HI/COSMOS_r0p5/'+str(cube_name)+'.r0p5.fits')
    bmaj=cubelet.header['BMAJ']*3600 # values in the header are in degrees, so we need to convert them to arcsec
    bmin=cubelet.header['BMIN']*3600
    bpa=cubelet.header['BPA']
    dpix=cubelet.header['CDELT2']*3600 #amount of arcsec per pixel
    df=cubelet.header['CDELT3'] #hz

    #Parameters to convert your flux from Jy/beam to Jy
    omega = (np.pi*(bmaj/206265)*(bmin/206265))/(4*np.log(2))*4.25*10**10 # beam area in arcsec
    bmpix=dpix**2/omega # amount of pixels per beam area- convertion factor, we will see it used later
    #Formulas from the lecture
    V=(restfr**2-freq**2)/(restfr**2+freq**2)*c # systemic velocity
    dist=V/H0           #distance Mpc
    z=(restfr-freq)/freq # redshift

    #Convert cahnnel width from HZ to km/s
    dv=(((restfr**2-(freq-df)**2)/(restfr**2+(freq-df)**2)*c))-V #channel width in km/s

    cubelet = cubelet.with_spectral_unit(u.km/u.s,
                                       velocity_convention='radio',
                                       rest_value=1.420405*u.GHz)
    
    #Convolve with new beam---------------------------------------
    # we define the new beam, which is 20 x 20 arcsec
    # I didn't convolve here because of ValueError: Beam could not be deconvolved
    beam = radio_beam.Beam(major=bmaj*u.arcsec, minor=bmin*u.arcsec, pa=0*u.deg)
    #then we convolve our original cubelet with the new beam
    cube_20 = cubelet.convolve_to(beam)
    subcube_20=cube_20[:,0:10,0:10]
    rms = subcube_20.std(axis=(1,2))
    sigma=np.mean(rms.value)
    #create mask cube -------------------------------------------
    
    include_mask = cube_20 > 3*sigma*cube_20.unit

    mask_cube = cube_20.with_mask(include_mask)
    #you can save the mask and examine it in ds9 
    mask_cube.write('~/Desktop/HI/COSMOS_r0p5/'+str(cube_name)+'_revised_masked_cube.fits', format='fits',overwrite='True') 

    #get moment 0 -----------------------------------------
    moment_0 = mask_cube.moment(order=0) 
    moment_0.hdu  #it is important so you can use coordinates for the plots
    atoms=(1.1*10**24*((moment_0)/(bmaj*bmin)))


    #plot----------------------------------------------
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111,projection=atoms.wcs)
    cmap = cm.plasma
    cmap.set_bad('white')
    im = ax.imshow(atoms.value,origin='lower', cmap=cmap)
    contours = ax.contour(atoms.value,levels=( 1e20,3e20,5e20,1e21,2E21), colors=['white'], linewidths=1)

    ax.set_xlabel("RA ", fontsize=10)
    ax.set_ylabel("Dec", fontsize=10)
    cb = plt.colorbar(mappable=im)
    cb.set_label(r'${\rm atoms}\,{\rm cm}^{-2}$')
    cb.add_lines(contours)
    el = Ellipse((7, 6), bmaj, bmin,angle=bpa, edgecolor='black',hatch='//', facecolor='none')
    ax.add_patch(el)
    plt.savefig('/Users/apple/Desktop/HI/COSMOS_r0p5_morphology/'+str(cube_name)+'_morphology.pdf',format = 'pdf',\
                 bbox_inches = 'tight', transparent=True)



def generate_Aloop(cube_name): 
    freq=data1.loc[cube_name, 'freq']*10**9 # central frequency of the source in the cubelet
    cubelet=SpectralCube.read('~/Desktop/HI/COSMOS_r0p5/'+str(cube_name)+'_revised_masked_cube.fits')
    bmaj=cubelet.header['BMAJ']*3600 # values in the header are in degrees, so we need to convert them to arcsec
    bmin=cubelet.header['BMIN']*3600
    bpa=cubelet.header['BPA']
    dpix=cubelet.header['CDELT2']*3600 #amount of arcsec per pixel
    df=cubelet.header['CDELT3'] #hz

    #Parameters to convert your flux from Jy/beam to Jy
    omega = (np.pi*(bmaj/206265)*(bmin/206265))/(4*np.log(2))*4.25*10**10 # beam area in arcsec
    bmpix=dpix**2/omega # amount of pixels per beam area- convertion factor, we will see it used later
    #Formulas from the lecture
    V=(restfr**2-freq**2)/(restfr**2+freq**2)*c # systemic velocity
    dist=V/H0           #distance Mpc
    z=(restfr-freq)/freq # redshift

    #Convert cahnnel width from HZ to km/s
    dv=(((restfr**2-(freq-df)**2)/(restfr**2+(freq-df)**2)*c))-V #channel width in km/s

    cubelet = cubelet.with_spectral_unit(u.km/u.s,
                                       velocity_convention='radio',
                                       rest_value=1.420405*u.GHz)
    

    #generate spectrum -----------------------------------------
    spectrum_new_jy = cubelet.sum(axis=(1,2))*bmpix
    spectrum_new_jy_converted = np.nan_to_num(spectrum_new_jy.value)
    v_arr1 = cubelet.spectral_axis.value
        #calculate asymmetry and store --------------------------------------
    left_sum=0
    right_sum=0
    spec_center=int(data1.loc[cube_name, 'spec_center'])
    i_begin = 0
    i_mid = int(len(v_arr1)/2)-spec_center
    i_end = int(len(v_arr1))
    

    for i in range(i_begin,i_mid):
        left_sum=spectrum_new_jy_converted[i]+left_sum
    for i in range(i_mid,i_end):    
        right_sum=spectrum_new_jy_converted[i]+right_sum

        
    Asy_l = round((left_sum-right_sum)/(left_sum+right_sum),2)
    data1.loc[cube_name, 'Asy_l'] = Asy_l
    data1.to_csv("~/Desktop/HI/COSMOS_32k_asy.txt", index=False, sep='\t')


for name in tqdm(data1.index.tolist(), desc="Generating Spectra"):
    
    generate_spectrum(name)
# for name in tqdm(list_sigma, desc="Generating Spectra"):
#     generate_spectrum(name)
#     generate_morphology(name)
