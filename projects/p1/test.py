import matplotlib.pyplot as plt
import numpy as np
import astropy
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from astropy.utils.data import download_file
from scipy.stats import lognorm
# from matplotlib import interactive


def plot_fits(data, vmin, vmax, title, norm = None):
    #plot the data...
    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis', origin='lower', norm=norm, aspect='auto')
    ax.set_xlabel('x pixels')
    ax.set_ylabel('y pixels')
    ax.set_title(title)
    fig.colorbar(im)

def fits_to_data(file_name):
    source_file = fits.open(file_name)
    source_data = source_file[0].data
    return source_data, source_file
parent = './fit_files/'
bias = './bias/'
dark = './dark/'
flat = './flat/'
source_fits = parent + 'source.fits'
cal_lamp_fits = parent + 'calibration.fits'
bias_frame_fits = parent + bias + 'bias0008.fits'
dark_frame_fits = parent + dark + 'dark0001.fits'

s_data, s_file = fits_to_data(source_fits)
cal_lamp_data, cal_lamp_file = fits_to_data(cal_lamp_fits)
bias_frame_data, bias_frame_file = fits_to_data(bias_frame_fits)
dark_frame_data, dark_frame_file = fits_to_data(dark_frame_fits)
dark_frame_fits = parent + dark + 'dark0001.fits'


source_header = s_file[0].header
print(repr(source_header))

print(repr(source_header))
#%%
#Now let's plot the different images:
#we need to establish a minimum and maximum range to plot the intensity over
vmin = 1e3
vmax = 4e3

plot_fits(s_data, vmin, vmax, title=source_fits, norm=None)
plot_fits(cal_lamp_data, vmin, vmax, title=cal_lamp_fits, norm=None)
plot_fits(bias_frame_data, vmin, vmax, title=bias_frame_fits, norm=None)
plot_fits(dark_frame_data, vmin, vmax, title=dark_frame_fits, norm=None)
plt.show()