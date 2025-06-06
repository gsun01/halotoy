#!usr/bin/env python

"""
This script plots histograms and halos from data.csv for multiple run directories in parallel
using joblib. It makes a movie from the halos in each run.

Usage:
    python plot-data.py

Assumes a directory structure like:
    group_dir/
        run_dir1/
            data.csv
            E_inj.csv
        run_dir2/
            data.csv
            E_inj.csv
        ...
Each run directory will produce:
    run_dir/halo_density_map.png
    run_dir/E2dNdE_inj.png
    run_dir/E2dNdE_obs.png
    run_dir/halo_imgs/halo_*.png  (movie frames)
    run_dir/halo.mp4
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
import os
import subprocess
import imageio_ffmpeg as ffmpeg
from scipy.signal import savgol_filter
from joblib import Parallel, delayed

def sort_data(path_to_data):
    # data header is:
    # E,theta_obs,phi_obs,T
    data = np.loadtxt(path_to_data, skiprows=1, delimiter=',')
    sorted_indices = np.argsort(data[:,3])
    sorted_data = data[sorted_indices]

    # plot all photons from all time
    E = sorted_data[:,0]
    th = sorted_data[:,1]*180/np.pi
    phi = sorted_data[:,2]
    T = sorted_data[:,3]
    w = np.ones_like(E) # weights are all 1 for now

    x = th*np.cos(phi)
    y = th*np.sin(phi)

    # mask out NaN values
    # mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    # x = x[mask]
    # y = y[mask]
    # w = w[mask]
    # E = E[mask]
    # th = th[mask]
    # phi = phi[mask]
    # T = T[mask]

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

def make_movie(run_dir, B, x, y, T, w, tbins=100, nbins=100, n_jobs=None):
    """
    Create a series of snapshots binned in time and compile into a movie.
    Within each run, frames are generated in parallel using joblib.
    Saves frames to run_dir/halo_imgs and final movie to run_dir/halo.mp4
    """

    movie_img_dir = os.path.join(run_dir, 'halo_imgs')
    os.makedirs(movie_img_dir, exist_ok=True)

    x_max = np.max(x) * 1.1
    y_max = np.max(y) * 1.1
    xy_max = np.max([x_max, y_max])

    T_min, T_max = np.min(T), np.max(T)
    Tbins = np.linspace(T_min, T_max, tbins)
    bin_assignment = np.digitize(T, Tbins)
    unique_Tbins = np.unique(bin_assignment)

    # Determine global normalization (serial pass)
    global_min = np.inf
    global_max = 0.0
    for b in unique_Tbins:
        mask = (bin_assignment == b)
        if not np.any(mask):
            continue
        hist, _, _ = np.histogram2d(
            x[mask], y[mask],
            bins=nbins,
            range=[[-xy_max, xy_max], [-xy_max, xy_max]],
            weights=w[mask]
        )
        nonzero = hist[hist > 0]
        if nonzero.size:
            global_min = min(global_min, nonzero.min())
            global_max = max(global_max, nonzero.max())

    norm = LogNorm(vmin=global_min, vmax=global_max)

    def make_frame(idx, b):
        """
        Generate and save a single frame for time bin b at index idx.
        """
        mask = (bin_assignment == b)
        if not np.any(mask):
            return

        hist, xedges, yedges = np.histogram2d(
            x[mask], y[mask],
            bins=nbins,
            range=[[-xy_max, xy_max], [-xy_max, xy_max]],
            weights=w[mask]
        )
        hist = np.ma.masked_where(hist == 0, hist)
        hist = hist / np.sum(hist)
        hist = hist / np.max(hist)
        cmap = copy.copy(plt.cm.hot)
        cmap.set_bad(color='black')

        plt.figure(figsize=(6, 6))
        plt.imshow(
            hist.T,
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect='equal',
            cmap=cmap,
            norm=norm
        )
        plt.colorbar(label='Normalized counts')
        plt.xlabel(r'$\theta \cos\phi$ [degrees]')
        plt.ylabel(r'$\theta \sin\phi$ [degrees]')
        dt = T[mask][0] - T_min
        plt.title(f'T - T0 = {dt:.2e} s, B = 1.0e-{B} G')

        frame_path = os.path.join(movie_img_dir, f'halo_{str(idx).zfill(4)}.png')
        plt.savefig(frame_path, dpi=300)
        plt.close()

    Parallel(n_jobs=n_jobs)(
        delayed(make_frame)(idx, b)
        for idx, b in enumerate(unique_Tbins)
        if np.any(bin_assignment == b)
    )

    # Compile frames into a movie with ffmpeg
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    framerate = str(max(1, tbins // 10))
    command = [
        ffmpeg_exe,
        '-y',
        '-framerate', framerate,
        '-pattern_type', 'glob',
        '-i', os.path.join(movie_img_dir, 'halo_*.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        os.path.join(run_dir, 'halo.mp4')
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"[{run_dir}] Error creating movie: {e}")

def process_run(run_dir, nbins=100, tbins=100):
    """
    Process a single run directory: 
      - sort data
      - generate density map
      - plot injected and observed spectra
      - build time-binned frames and movie
    """
    run_dir = os.path.abspath(run_dir)
    hist_dir = os.path.join(run_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)

    dir_name = os.path.basename(run_dir.rstrip('/'))
    try:
        B_token = [token for token in dir_name.split('_') if token.startswith('B')][0]
        B = B_token[1:]
    except IndexError:
        B = 'unknown'

    data_path = os.path.join(run_dir, 'data.csv')
    if not os.path.isfile(data_path):
        print(f"[{run_dir}] data.csv not found, skipping.")
        return

    E, th, phi, x, y, T, w = sort_data(data_path)

    plot_density_map(B, x, y, w, run_dir, nbins=nbins)
    plot_E2dNdE_inj(run_dir, hist_dir, nbins=nbins)
    plot_E2dNdE_obs(E, w, run_dir, hist_dir, nbins=nbins)

    inner_jobs = max(1, (os.cpu_count() or 1) // 2)
    make_movie(run_dir, B, x, y, T, w, tbins=100, nbins=100, n_jobs=inner_jobs)

    print(f"[{run_dir}] Done.")

def main():
    data_dir = '/data/sguotong/data/halotoy'
    group_dir = os.path.join(data_dir, 'test_0606_01')
    if not os.path.isdir(group_dir):
        print(f"Group directory '{group_dir}' not found. Exiting.")
        return

    run_dirs = [
        os.path.join(group_dir, entry)
        for entry in os.listdir(group_dir)
        if os.path.isdir(os.path.join(group_dir, entry))
    ]

    # Parallelize with joblib: use all available cores by default (n_jobs=-1)
    Parallel(n_jobs=-1)(
        delayed(process_run)(run_dir)
        for run_dir in run_dirs
    )

    print("All runs processed.")

if __name__ == '__main__':
    main()