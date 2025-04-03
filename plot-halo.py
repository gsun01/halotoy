# plot halo from photon_data.csv
# Usage: python plot-halo.py
# Output: halo.png

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import imageio_ffmpeg as ffmpeg

data = np.abs(np.loadtxt('data_v1.csv', skiprows=1, delimiter=','))
sorted_indices = np.argsort(data[:,3])
sorted_data = data[sorted_indices]

T = sorted_data[:,3]
x = np.abs(sorted_data[:,1])*180/np.pi*np.cos(sorted_data[:,2])
x_max = np.max(x)*1.1

# define bin edges and bin data by T_delay
# linear bins in T because we want to make a movie where linear time makes sense
num_bins = 1000
T_min, T_max = np.min(T), np.max(T)
bins = np.linspace(T_min, T_max, num_bins)
bin_assignment = np.digitize(T, bins)
unique_bins = np.unique(bin_assignment)

# plot halo for each bin
halo_dir = 'halo_imgs'
if not os.path.exists(halo_dir):
    os.makedirs(halo_dir)

for bin in unique_bins:
    rows_in_bin = sorted_data[bin_assignment == bin]
    E_in_bin = rows_in_bin[:,0]
    T_in_bin = rows_in_bin[:,3]
    # convert theta and phi to screen coordinates
    x = np.abs(rows_in_bin[:,1])*180/np.pi*np.cos(rows_in_bin[:,2])
    y = np.abs(rows_in_bin[:,1])*180/np.pi*np.sin(rows_in_bin[:,2])
    # plot halo
    plt.figure(figsize=(6,6))
    # if 5 < E < 7, color = 'red'
    plt.scatter(x[(E_in_bin>5) & (E_in_bin<7)], y[(E_in_bin>5) & (E_in_bin<7)], s=5.0, color='red', label='5 TeV < E < 7 TeV')
    # if 7 < E < 10, color = 'green'
    plt.scatter(x[(E_in_bin>7) & (E_in_bin<10)], y[(E_in_bin>7) & (E_in_bin<10)], s=5.0, color='green', label='7 TeV < E < 10 TeV')
    # if 10 < E < 20, color = 'blue'
    plt.scatter(x[(E_in_bin>10) & (E_in_bin<20)], y[(E_in_bin>10) & (E_in_bin<20)], s=5.0, color='blue', label='10 TeV < E < 20 TeV')
    # show labels
    plt.xlabel('degrees')
    plt.ylabel('degrees')
    plt.xlim(-x_max, x_max)
    plt.ylim(-x_max, x_max)
    # scientific notation for T_delay
    plt.title(f'T_delay = {T_in_bin[0]:.2e} s')
    plt.legend()
    plt.grid()
    # make bin number 4 digits
    bin_str = str(bin).zfill(4)
    plt.savefig(os.path.join(halo_dir, f'halo_{bin_str}.png'), dpi=300)
    plt.close()

# make a movie from halo/
ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
framerate = str(int(num_bins/10))
command = [
    ffmpeg_exe,
    '-framerate', framerate,
    '-pattern_type', 'glob',
    '-i', os.path.join(halo_dir,'halo_*.png'),
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    'halo.mp4'
]

# Run the command
try:
    subprocess.run(command, check=True)
    print("Movie created successfully!")
except subprocess.CalledProcessError as e:
    print("An error occurred while creating the movie:", e)