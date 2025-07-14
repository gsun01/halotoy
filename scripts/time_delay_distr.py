import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Read config path from CLI
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <config.json>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}")
        sys.exit(2)

    # Load config
    cfg = json.loads(config_path.read_text())
    output_base_dir = Path(cfg["output_base_dir"])
    if not output_base_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_base_dir}")
    
    # list all subdirs in the output base directory
    subdirs = [d for d in output_base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subdirectories found in {output_base_dir}")
        sys.exit(3)
    print(f"Found {len(subdirs)} subdirectories in {output_base_dir}")
    
    T_max_arr_lo, T_max_arr_hi = np.array([]), np.array([])
    E_min_lo, E_max_lo = 8.00e-2, 5.60e-1
    E_min_hi, E_max_hi = 1.52, 2.00
    no_data = 0
    for subdir in subdirs:
        data_file = subdir / "data.csv"
        # find max and min of T values
        if not data_file.is_file():
            print(f"Data file not found in {subdir}: {data_file}")
            continue
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        try:
            T = data[:, 3]
        except IndexError:
            print(f"Data file {data_file} is probably empty.")
            continue
        # select lo/hi energy bins
        E = data[:, 0]
        E_obs = 0.32*(E/20)**2
        mask_lo = (E_obs >= E_min_lo) & (E_obs <= E_max_lo)
        mask_hi = (E_obs >= E_min_hi) & (E_obs <= E_max_hi)
        if not np.any(mask_lo) or not np.any(mask_hi):
            print(f"No data in energy range for {subdir}: {data_file}")
            no_data += 1
            continue
        T_lo = T[mask_lo]
        T_hi = T[mask_hi]
        T_max_lo, T_max_hi = np.max(T_lo), np.max(T_hi)
        T_max_arr_lo = np.append(T_max_arr_lo, T_max_lo)
        T_max_arr_hi = np.append(T_max_arr_hi, T_max_hi)

    print("no_data:", no_data)
    # plot histogram of T_max_lo and T_max_hi in top-bottom subplots
    # plt.figure(figsize=(10, 10))

    # plt.subplot(2, 1, 1)
    # plt.hist(T_max_arr_lo, bins=30, alpha=0.5, label='T_max_lo', color='blue')
    # plt.xlabel('Time Delay [seconds]')
    # plt.ylabel('Counts')
    # plt.xscale('log')
    # plt.title(f'Time Delay Distribution [{E_min_lo:.2e}, {E_max_lo:.2e}] GeV, {output_base_dir.name}')

    # plt.subplot(2, 1, 2)
    # plt.hist(T_max_arr_hi, bins=30, alpha=0.5, label='T_max_hi', color='red')
    # plt.xlabel('Time Delay [seconds]')
    # plt.ylabel('Counts')
    # plt.xscale('log')
    # plt.title(f'Time Delay Distribution [{E_min_hi:.2e}, {E_max_hi:.2e}] GeV, {output_base_dir.name}')
    # plt.tight_layout()

    # plot histogram of T_max_lo and T_max_hi in one plot
    plt.figure(figsize=(10, 5))
    plt.hist(T_max_arr_lo, bins=30, alpha=0.5, label=f'[{E_min_lo:.2e}, {E_max_lo:.2e}] GeV', color='blue', histtype='step')
    plt.hist(T_max_arr_hi, bins=30, alpha=0.5, label=f'[{E_min_hi:.2e}, {E_max_hi:.2e}] GeV', color='red', histtype='step')
    plt.xlabel('Time Delay [seconds]')
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.legend()
    plt.title(f'Time Delay Distribution in two energy bins; {output_base_dir.name}')


    plt.savefig(output_base_dir / "time_delay_distribution.svg", dpi=300)
    print(f"Saved time delay distribution plot to {output_base_dir / 'time_delay_distribution.svg'}")
    

if __name__ == "__main__":
    main()