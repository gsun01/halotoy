#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np
import healpy as hp
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
    # catalog_file    = Path(cfg["catalog_file"])
    output_base_dir = Path(cfg["output_base_dir"])
    nside           = cfg.get("NSIDE", 128)

    if not output_base_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_base_dir}")

    # Prepare HEALPix map
    npix   = hp.nside2npix(nside)
    sky_map = np.zeros(npix, dtype=np.int64)

    # Accumulate from each GRB run
    for run_dir in sorted(output_base_dir.glob("GRB_*")):
        data_csv = run_dir / "data.csv"
        if not data_csv.is_file():
            print(f"Warning: missing {data_csv}, skipping")
            continue

        # load theta_obs, phi_obs
        data = np.genfromtxt(
            data_csv, delimiter=",", names=True,
            usecols=("theta_obs","phi_obs")
        )
        if data.size == 0:
            continue

        theta = data["theta_obs"].ravel()
        phi   = data["phi_obs"].ravel()
        pix   = hp.ang2pix(nside, theta, phi, nest=False)

        sky_map += np.bincount(pix, minlength=npix)

    # Write out FITS
    fits_out = output_base_dir / f"halo_map_nside{nside}.fits"
    hp.write_map(str(fits_out), sky_map, nest=False, overwrite=True)
    print(f"Wrote HEALPix map to {fits_out}")

    # Plot and save Mollweide projection (RING scheme)
    plt.figure(figsize=(8, 6))
    hp.mollview(
        sky_map,
        title=f"GRB Pair‐Halo Sky Map (nside={nside})",
        unit="counts",
        norm="hist"
    )
    svg_out = output_base_dir / f"halo_map_nside{nside}.svg"
    plt.savefig(str(svg_out), dpi=300)
    plt.close()
    print(f"Saved map plot to {svg_out}")

    # cl = hp.anafast(sky_map, lmax=512)
    cl = hp.anafast(sky_map)

    #  D_ell = l(l+1) C_ell / (2π)
    ell = np.arange(len(cl))
    D_ell = ell * (ell + 1) * cl / (2 * np.pi)

    # Plot D_ell
    plt.figure(figsize=(6,4))
    plt.loglog(ell[1:], np.abs(D_ell[1:]), label=r"$D_\ell$")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell = \frac{\ell(\ell+1)}{2\pi}C_\ell$")
    # plt.title("Angular Power Spectrum of GRB Pair-Halo Sky Map")
    plt.legend()
    plt.tight_layout()
    svg_out = output_base_dir / f"aps_nside{nside}.svg"
    plt.savefig(str(svg_out), dpi=300)
    plt.close()
    print(f"Saved Cl plot to {svg_out}")

if __name__ == "__main__":
    main()
