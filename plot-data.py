#!usr/bin/env python

# This script is used to plot histograms and halos from data.csv
# and make a movie from the halos

# Usage: python plot-data.py z B th_j th_v
# Output: halo.png, halo_imgs/*.png, halo.mp4, histograms/*.png

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
import os
import subprocess
import imageio_ffmpeg as ffmpeg
from scipy.signal import savgol_filter

def sort_data(path_to_data):
    # data header is:
    # E,theta_obs,phi_obs,T,w,th_emj,th_emi,delta
    data = np.loadtxt(path_to_data, skiprows=1, delimiter=',')
    sorted_indices = np.argsort(data[:,3])
    sorted_data = data[sorted_indices]

    # plot all photons from all time
    E = sorted_data[:,0]
    th = sorted_data[:,1]*180/np.pi
    phi = sorted_data[:,2]
    T = sorted_data[:,3]
    w = sorted_data[:,4]

    x = th*np.cos(phi)
    y = th*np.sin(phi)

    # mask out NaN values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    E = E[mask]
    th = th[mask]
    phi = phi[mask]
    T = T[mask]

    return E, th, phi, x, y, T, w

def plot_E2dNdE_obs(E, w, hist_dir, nbins=100):
    # observed spectrum
    E_obs = 0.32*(E/20)**2  # observed photon energy in TeV
    hist, edges = np.histogram(E_obs, bins=nbins, weights=w, density=False)
    centers = 0.5*(edges[1:]+edges[:-1])
    widths = edges[1:] - edges[:-1]
    dNdE = hist / widths
    E2dNdE = dNdE * centers**2
    hist_sg = savgol_filter(E2dNdE, window_length=11, polyorder=3)

    plt.figure(figsize=(6,6))
    plt.plot(centers, hist_sg, lw=2, color='orange', label='observed spectrum')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E [TeV]')
    plt.ylabel(r'$E^2dN/dE$ [arb. u]')
    plt.legend()
    plt.savefig(os.path.join(hist_dir,'E2dNdE.png'), dpi=300)
    plt.close()

def plot_E2dNdE_inj(run_dir, nbins=100):
    E_inj = np.loadtxt(os.path.join(run_dir, 'E_inj.csv'))
    hist, edges = np.histogram(E_inj, bins=nbins, density=False)
    centers = 0.5*(edges[1:]+edges[:-1])
    widths = edges[1:] - edges[:-1]
    dNdE = hist / widths
    E2dNdE = dNdE * centers**2
    hist_sg = savgol_filter(E2dNdE, window_length=11, polyorder=3)

    plt.figure(figsize=(6,6))
    plt.plot(centers, hist_sg, lw=2, color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E [TeV]')
    plt.ylabel(r'$E^2dN/dE$ [arb. u]')
    plt.savefig(os.path.join(run_dir,'E2dNdE_inj.png'), dpi=300)
    plt.close()

def plot_histogram(th, phi, T, w, hist_dir, nbins=100):
    # plot histogram of th_obs
    plt.figure(figsize=(6,6))
    plt.hist(th, bins=nbins, color='black', alpha=0.7, weights=w)
    plt.xlabel('theta_obs')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'th-hist.png'), dpi=300)
    plt.close()

    # plot histogram of phi_obs
    plt.figure(figsize=(6,6))
    plt.hist(phi, bins=nbins, color='black', alpha=0.7, weights=w)
    plt.xlabel('phi_obs')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'phi-hist.png'), dpi=300)
    plt.close()

    # plot histogram of T
    plt.figure(figsize=(6,6))
    plt.hist(T, bins=nbins, color='black', alpha=0.7, weights=w)
    plt.xlabel('T')
    plt.xscale('log')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(hist_dir,'T-hist.png'), dpi=300)
    plt.close()

def plot_density_map(B, x, y, w, run_dir, nbins=100):
    x_max = np.max(x)*1.1
    y_max = np.max(y)*1.1
    xy_max = np.max([x_max, y_max])
    hist, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[[-xy_max, xy_max], [-xy_max, xy_max]], weights=w)

    # mask 0 counts for log colorbar and set bad color to black
    hist = np.ma.masked_where(hist == 0, hist)
    hist = hist / np.sum(hist)
    hist = hist / np.max(hist)
    cmap = copy.copy(plt.cm.hot)
    cmap.set_bad(color='black')

    # Plot the histogram as an image
    plt.figure(figsize=(6,6))
    plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='equal', cmap=cmap, norm=LogNorm())
    cbar = plt.colorbar(label='Counts [arbitrary u]')
    # cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    plt.xlabel(r'$\theta \cos\phi$ [degrees]')
    plt.ylabel(r'$\theta \sin\phi$ [degrees]')
    plt.title(f'B = 1.0e-{B} G')
    plt.savefig(os.path.join(run_dir,'halo_density_map.png'), dpi=300)
    plt.close()

def make_movie(B, x, y, T, w, run_dir, tbins=100, nbins=100):
    movie_img_dir = os.path.join(run_dir, 'halo_imgs')
    if not os.path.exists(movie_img_dir):
        os.makedirs(movie_img_dir)
    
    x_max = np.max(x)*1.1
    y_max = np.max(y)*1.1
    xy_max = np.max([x_max, y_max])

    T_min, T_max = np.min(T), np.max(T)
    # linear bins in T because we want to make a movie where linear time makes sense
    Tbins = np.linspace(T_min, T_max, tbins)
    # actually let's use log bins
    # Tbins = np.logspace(np.log10(T_min), np.log10(T_max), tbins)
    # define bin edges and bin data by T_delay
    bin_assignment = np.digitize(T, Tbins)
    unique_Tbins = np.unique(bin_assignment)

    # global min max for color normalization
    global_min = np.inf
    global_max = 0.0

    for b in unique_Tbins:
        mask = bin_assignment == b
        if not np.any(mask):
            continue

        # raw 2D histogram with weights
        hist, xedges, yedges = np.histogram2d(
            x[mask], y[mask],
            bins=nbins,
            range=[[-xy_max, xy_max], [-xy_max, xy_max]],
            weights=w[mask]
        )
        hist = hist / np.sum(hist)

        pos = hist[hist > 0]
        if pos.size:
            global_min = min(global_min, pos.min())
            global_max = max(global_max, pos.max())

    norm = LogNorm(vmin=global_min, vmax=global_max)

    bin_num = 0
    for bin in unique_Tbins:
        T_in_bin = T[bin_assignment == bin]
        x_in_bin = x[bin_assignment == bin]
        y_in_bin = y[bin_assignment == bin]
        w_in_bin = w[bin_assignment == bin]
        
        hist, xedges, yedges = np.histogram2d(x_in_bin, y_in_bin, bins=nbins, range=[[-xy_max, xy_max], [-xy_max, xy_max]], weights=w_in_bin)

        # mask 0 counts for log colorbar and set bad color to black
        hist = np.ma.masked_where(hist == 0, hist)
        hist = hist / np.sum(hist)
        hist = hist / np.max(hist)
        cmap = copy.copy(plt.cm.hot)
        cmap.set_bad(color='black')

        # Plot the histogram as an image
        plt.figure(figsize=(6,6))
        plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='equal', cmap=cmap, norm=norm)
        cbar = plt.colorbar(label='Counts [arbitrary u]')
        # cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
        plt.xlabel(r'$\theta \cos\phi$ [degrees]')
        plt.ylabel(r'$\theta \sin\phi$ [degrees]')
        plt.title(f'T0 = {T_min:.2e} s, T - T0 = {T_in_bin[0]-T_min:.2e} s, B = 1.0e-{B} G')
        plt.savefig(os.path.join(movie_img_dir, f'halo_{str(bin_num).zfill(4)}.png'), dpi=300)
        plt.close()
        bin_num += 1

    # make a movie from halo/
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    framerate = str(int(tbins/10))
    # framerate = '24'
    command = [
        ffmpeg_exe, '-y',   # overwrite output file
        '-framerate', framerate,
        '-pattern_type', 'glob',
        '-i', os.path.join(movie_img_dir,'halo_*.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        os.path.join(os.path.join(run_dir,'halo.mp4'))
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print("Movie created successfully!")
    except subprocess.CalledProcessError as e:
        print("An error occurred while creating the movie:", e)

def main():
    nbins = 100
    group_dir = 'runs_VHE'
    for dir in os.listdir(group_dir):
        run_dir = os.path.join(group_dir, dir)
        if not os.path.isdir(run_dir):
            print(f"{run_dir} is not a directory, skipping.")
            continue
        hist_dir = os.path.join(run_dir, 'histograms')
        if not os.path.exists(hist_dir):
            os.makedirs(hist_dir)


        B = dir.split('_')[1][1:]
        E, th, phi, x, y, T, w = sort_data(os.path.join(run_dir, 'data.csv'))
        # plot_histogram(th, phi, T, w, run_dir, nbins=nbins)
        plot_density_map(B, x, y, w, run_dir, nbins=nbins)
        plot_E2dNdE_inj(run_dir, nbins=nbins)
        plot_E2dNdE_obs(E, w, run_dir, nbins=nbins)
        make_movie(B, x, y, T, w, run_dir, nbins=nbins)
        print(f"Plotting {run_dir} done.")
    print("All plots done.")

if __name__ == '__main__':
    main()