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
    
    T_max_arr_lo, T_max_arr_hi = np.array([]), np.array([])
    E_min_lo, E_max_lo = 8.00e-2, 5.60e-1
    E_min_hi, E_max_hi = 1.52, 2.00
    no_data = 0
    for data_csv in (output_base_dir / "photon_data").iterdir():
        # find max and min of T values
        data = np.loadtxt(data_csv, delimiter=',', skiprows=1)
        try:
            T = data[:, 3]
        except IndexError:
            print(f"Data file {data_csv} is probably empty.")
            continue
        # select lo/hi energy bins
        E = data[:, 0]
        E_obs = 0.32*(E/20)**2
        mask_lo = (E_obs >= E_min_lo) & (E_obs <= E_max_lo)
        mask_hi = (E_obs >= E_min_hi) & (E_obs <= E_max_hi)
        if not np.any(mask_lo) or not np.any(mask_hi):
            print(f"No data in energy range for {data_csv}")
            no_data += 1
            continue
        T_lo = T[mask_lo]
        T_hi = T[mask_hi]
        T_max_lo, T_max_hi = np.max(T_lo), np.max(T_hi)
        T_max_arr_lo = np.append(T_max_arr_lo, T_max_lo)
        T_max_arr_hi = np.append(T_max_arr_hi, T_max_hi)

    print("# of files with no data: ", no_data)

    # plot histogram of T_max_lo and T_max_hi in one plot
    plt.figure(figsize=(10, 5))
    plt.hist(T_max_arr_lo, bins=30, alpha=0.5, label=f'[{E_min_lo:.2e}, {E_max_lo:.2e}] TeV', color='blue', histtype='step')
    plt.hist(T_max_arr_hi, bins=30, alpha=0.5, label=f'[{E_min_hi:.2e}, {E_max_hi:.2e}] TeV', color='red', histtype='step')
    plt.xlabel('Time Delay [seconds]')
    plt.ylabel('Counts')
    plt.xscale('log')
    plt.legend()
    plt.title(f'Time Delay Distribution in two energy bins; {output_base_dir.name}')
    outfile = output_base_dir / "time_delay_distribution.svg"
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved time delay distribution plot to {outfile}")

if __name__ == "__main__":
    main()