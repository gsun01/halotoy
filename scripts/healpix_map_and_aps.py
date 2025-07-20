#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def get_all_grb_data(output_base_dir):
    """Aggregate E_obs, theta_obs, phi_obs from all GRB runs."""
    E = np.zeros(0, dtype=np.float64)
    theta = np.zeros(0, dtype=np.float64)
    phi   = np.zeros(0, dtype=np.float64)

    for data_csv in (output_base_dir / "photon_data").iterdir():
        data = np.loadtxt(data_csv, delimiter=',', skiprows=1)
        if data.size <= 4:
            # make-shift filter for files with less than 2 rows
            continue
        E     = np.concatenate((E, data[:,0]))
        theta = np.concatenate((theta, data[:,1]))
        phi   = np.concatenate((phi, data[:,2]))
    E = 0.32 * (E/20)**2  # observed E in TeV
    return E, theta, phi

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
    nside           = cfg.get("NSIDE", 128)

    if not output_base_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_base_dir}")

    # Aggregate E_obs, theta_obs, phi_obs from all GRB runs
    E, theta, phi = get_all_grb_data(output_base_dir)

    # define E bin edges
    n_bins = 4
    edges = np.linspace(0.08, 2, n_bins+1)
    print(f"E bin edges: {edges}")

    # Prepare HEALPix map
    npix   = hp.nside2npix(nside)
    ell_max = 3*nside - 1
    pix_size = hp.nside2pixarea(nside)  # sr
    pix_size_deg2 = np.sqrt(hp.nside2pixarea(nside, degrees=True))  # (x deg)^2

    for i in range(n_bins):
        E_lo, E_hi = edges[i], edges[i+1]
        delta_E = E_hi - E_lo

        mask = (E >= E_lo) & (E < E_hi)

        th_in_bin = theta[mask]
        ph_in_bin = phi[mask]
        E_in_bin  = E[mask]

        pix = hp.ang2pix(nside, th_in_bin, ph_in_bin)
        # m is a map of \sum{E} in each pixel
        # m = np.bincount(pix, weights=E_in_bin, minlength=npix)
        # m is a count map
        m = np.bincount(pix, weights=None, minlength=npix)

        # correct for exposure & pixel solid angle:
        # I = m / (exposure * pix_size)
        # exposure is the exposure map of size npix
        # now I is intensity map of unit ph*cm^{-2}*s^{-1}*sr^{-1}
        intensity = m / pix_size

        beam = np.deg2rad(0.05)
        I_sm = hp.smoothing(intensity, fwhm=beam, verbose=False)

        # power spectrum: I(n) = \isum_{lm} a_lm Y_lm(n), Cl = 1/(2l+1) \sum_{m=-l}^{l} |a_lm|^2
        cl = hp.anafast(I_sm, lmax=ell_max)   # returns array length ell_max+1
        # pixwin = hp.pixwin(nside, lmax=ell_max)
        # cl /= pixwin**2  # correct for pixel window function

        fits_out = output_base_dir / f"Healpix_map_{E_lo:.2e}-{E_hi:.2e}TeV_NSIDE{nside}.fits"
        hp.write_map(fits_out, intensity, overwrite=True)
        print(f"Saved Healpix map fits to {fits_out}")

        txt_out = output_base_dir / f"Cl_{E_lo:.2e}TeV_NSIDE{nside}.txt"
        np.savetxt(txt_out, cl)
        print(f"Saved Cl txt to {txt_out}")

        plt.figure(figsize=(8, 6))
        logI_sm = np.log10(I_sm, where=(I_sm>0))
        logI_sm[I_sm <= 0] = hp.UNSEEN
        hp.mollview(
            logI_sm,
            title=f"[{E_lo:.2e}, {E_hi:.2e}] TeV, NSIDE={nside}, pix size=({pix_size_deg2:.2f} [deg])^2",
            unit=r"Log(Intensity [$cm^{-2} s^{-1} sr^{-1}$])",
            norm="hist",
            badcolor="black",
        )
        svg_out = output_base_dir / f"Healpix_map_{E_lo:.2e}TeV_NSIDE{nside}.svg"
        plt.savefig(svg_out, dpi=300)
        plt.close()
        print(f"Saved Healpix map to {svg_out}")

        plt.figure(figsize=(6, 4))
        ell = np.arange(ell_max + 1)
        plt.loglog(ell[1:], np.abs(cl[1:]), label=r"$C_\ell$")
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell [cm^{-4}\,s^{-2}\,sr^{-2}\,sr]$")
        plt.title(f"Auto-APS [{E_lo:.2e}, {E_hi:.2e}] TeV")
        # plt.legend()
        plt.tight_layout()
        svg_out = output_base_dir / f"Auto_APS_{E_lo:.2e}TeV_NSIDE{nside}.svg"
        plt.savefig(svg_out, dpi=300)
        plt.close()
        print(f"Saved Auto-APS plot to {svg_out}")

if __name__ == "__main__":
    main()
