import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

def get_grb_data_from_id(output_base_dir, grb_id):
    # returns observed and injected energy arrays for a given GRB ID
    grb_data_file = output_base_dir / "photon_data" / f'data_GRB_{grb_id}.csv'
    if not grb_data_file.exists():
        raise FileNotFoundError(f"GRB data file {grb_data_file} does not exist.")
    grb_Einj_file = output_base_dir / "Einj" / f'Einj_GRB_{grb_id}.csv'
    if not grb_Einj_file.exists():
        raise FileNotFoundError(f"GRB Einj file {grb_Einj_file} does not exist.")
    
    E = np.loadtxt(grb_data_file, delimiter=',', skiprows=1)[:,0]
    E_obs = 0.32*(E/20)**2          # TeV
    E_inj = np.loadtxt(grb_Einj_file, delimiter=',')
    
    return E_obs, E_inj

def main():
    if len(sys.argv) != 3:
        print("Usage: python energetics.py <config.json> grb_id")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}")
        sys.exit(2)
    cfg = json.loads(config_path.read_text())

    output_base_dir = Path(cfg["output_base_dir"])
    if not output_base_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_base_dir}")

    grb_id = sys.argv[2]
    E_obs, E_inj = get_grb_data_from_id(output_base_dir, grb_id)
    
    nbins = 100

    # injection spectrum
    hist_inj, edges_inj = np.histogram(E_inj, bins = nbins, density=False)
    centers_inj = 0.5*(edges_inj[1:]+edges_inj[:-1])
    widths_inj = edges_inj[1:] - edges_inj[:-1]
    dNdE_inj = hist_inj / widths_inj
    E2dNdE_inj = dNdE_inj * centers_inj**2
    hist_sg_inj = savgol_filter(E2dNdE_inj, window_length=11, polyorder=3)

    # observed spectrum
    hist_obs, edges_obs = np.histogram(E_obs, bins=nbins, density=False)
    centers_obs = 0.5*(edges_obs[1:]+edges_obs[:-1])
    widths_obs = edges_obs[1:] - edges_obs[:-1]
    dNdE_obs = hist_obs / widths_obs
    E2dNdE_obs = dNdE_obs * centers_obs**2
    hist_sg_obs = savgol_filter(E2dNdE_obs, window_length=11, polyorder=3)

    plt.figure(figsize=(10,5))
    plt.plot(centers_inj, hist_sg_inj, lw=2, color='blue', label='injected spectrum')
    plt.plot(centers_obs, hist_sg_obs, lw=2, color='orange', label='observed spectrum')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E [TeV]')
    plt.ylabel(r'$E^2dN/dE$ [arb. u]')
    plt.legend()
    outfile = output_base_dir / f'E2dNdE_{grb_id}.svg'
    plt.savefig(outfile, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()