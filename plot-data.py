#!usr/bin/env python

# This script is used to plot histograms and halos from data.csv
# and make a movie from the halos

# Usage: python plot-data.py z B th_j th_v
# Output: halo.png, halo_imgs/*.png, halo.mp4, histograms/*.png

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import subprocess
import imageio_ffmpeg as ffmpeg

def sort_data(path_to_data):
    data = np.abs(np.loadtxt(path_to_data, skiprows=1, delimiter=','))
    sorted_indices = np.argsort(data[:,3])
    sorted_data = data[sorted_indices]

    # plot all photons from all time
    E = sorted_data[:,0]
    th = sorted_data[:,1]*180/np.pi
    phi = sorted_data[:,2]
    T = sorted_data[:,3]

    x = np.abs(th)*np.cos(phi)
    y = np.abs(th)*np.sin(phi)

    return E, th, phi, x, y, T

def plot_histogram(E, th, phi, T, hist_dir):
    # plot histogram of E
    plt.figure(figsize=(6,6))
    plt.hist(E, bins=100, color='black', alpha=0.7)
    plt.xlabel('E')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'E-hist.png'), dpi=300)
    plt.close()

    # plot histogram of th_obs
    plt.figure(figsize=(6,6))
    plt.hist(th, bins=100, color='black', alpha=0.7)
    plt.xlabel('theta_obs')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'th-hist.png'), dpi=300)
    plt.close()

    # plot histogram of phi_obs
    plt.figure(figsize=(6,6))
    plt.hist(phi, bins=100, color='black', alpha=0.7)
    plt.xlabel('phi_obs')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'phi-hist.png'), dpi=300)
    plt.close()

    # plot histogram of T
    plt.figure(figsize=(6,6))
    plt.hist(T, bins=100, color='black', alpha=0.7)
    plt.xlabel('T')
    # plt.xscale('log')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'T-hist.png'), dpi=300)
    plt.close()

def plot_all_photons(E, x, y, run_dir):
    x_max = np.max(x)*1.1

    plt.figure(figsize=(6,6))
    # if 5 < E < 7, color = 'red'
    plt.scatter(x[(E>5) & (E<7)], y[(E>5) & (E<7)], s=5.0, color='red', label='5 TeV < E < 7 TeV')
    # if 7 < E < 10, color = 'green'
    plt.scatter(x[(E>7) & (E<10)], y[(E>7) & (E<10)], s=5.0, color='green', label='7 TeV < E < 10 TeV')
    # if 10 < E < 20, color = 'blue'
    plt.scatter(x[(E>10) & (E<20)], y[(E>10) & (E<20)], s=5.0, color='blue', label='10 TeV < E < 20 TeV')
    # show labels
    plt.xlabel('degrees')
    plt.ylabel('degrees')
    plt.xlim(-x_max, x_max)
    plt.ylim(-x_max, x_max)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(run_dir,'halo.png'), dpi=300)
    plt.close()

def make_movie(E, x, y, T, movie_img_dir, nbins=1000):
    num_bins = nbins
    x_max = np.max(x)*1.1

    # linear bins in T because we want to make a movie where linear time makes sense
    T_min, T_max = np.min(T), np.max(T)
    bins = np.linspace(T_min, T_max, num_bins)
    # define bin edges and bin data by T_delay
    bin_assignment = np.digitize(T, bins)
    unique_bins = np.unique(bin_assignment)

    for bin in unique_bins:
        E_in_bin = E[bin_assignment == bin]
        T_in_bin = T[bin_assignment == bin]
        x_in_bin = x[bin_assignment == bin]
        y_in_bin = y[bin_assignment == bin]
        # plot halo
        plt.figure(figsize=(6,6))
        # if 5 < E < 7, color = 'red'
        plt.scatter(x_in_bin[(E_in_bin>5) & (E_in_bin<7)], y_in_bin[(E_in_bin>5) & (E_in_bin<7)], s=5.0, color='red', label='5 TeV < E < 7 TeV')
        # if 7 < E < 10, color = 'green'
        plt.scatter(x_in_bin[(E_in_bin>7) & (E_in_bin<10)], y_in_bin[(E_in_bin>7) & (E_in_bin<10)], s=5.0, color='green', label='7 TeV < E < 10 TeV')
        # if 10 < E < 20, color = 'blue'
        plt.scatter(x_in_bin[(E_in_bin>10) & (E_in_bin<20)], y_in_bin[(E_in_bin>10) & (E_in_bin<20)], s=5.0, color='blue', label='10 TeV < E < 20 TeV')
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
        plt.savefig(os.path.join(movie_img_dir, f'halo_{bin_str}.png'), dpi=300)
        plt.close()

    # make a movie from halo/
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    framerate = str(int(num_bins/10))
    # framerate = '24'
    command = [
        ffmpeg_exe,
        '-framerate', framerate,
        '-pattern_type', 'glob',
        '-i', os.path.join(movie_img_dir,'halo_*.png'),
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

def main():
    args = sys.argv
    if len(args) != 5:
        print("Usage: python plot-data.py z B th_j th_v")
        sys.exit(1)
    z = args[1]
    B = args[2]
    th_j = args[3]
    th_v = args[4]

    run_dir = os.path.join('runs', f'z{z}_B{B}_j{th_j}_v{th_v}')
    if not os.path.exists(run_dir):
        print(f"Run directory {run_dir} does not exist.")
        sys.exit(1)
    movie_img_dir = os.path.join(run_dir, 'halo_imgs')
    hist_dir = os.path.join(run_dir, 'histograms')
    if not os.path.exists(movie_img_dir):
        os.makedirs(movie_img_dir)
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)
    
    path_to_data = os.path.join(run_dir, 'data.csv')
    if not os.path.exists(path_to_data):
        print(f"Data file {path_to_data} does not exist.")
        sys.exit(1)
    

    E, th, phi, x, y, T = sort_data(path_to_data)
    plot_histogram(E, th, phi, T, hist_dir)
    plot_all_photons(E, x, y, run_dir)
    # make_movie(E, x, y, T, movie_img_dir)

if __name__ == '__main__':
    main()