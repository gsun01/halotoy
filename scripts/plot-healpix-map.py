#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def main():
    # 1) Read config path from CLI
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <config.json>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}")
        sys.exit(2)

    # 2) Load config
    cfg = json.loads(config_path.read_text())
    catalog_file    = Path(cfg["catalog_file"])
    output_base_dir = Path(cfg["output_base_dir"])
    nside           = cfg.get("NSIDE", 128)

    if not output_base_dir.is_dir():
        raise RuntimeError(f"Output directory not found: {output_base_dir}")

    # 3) Prepare HEALPix map
    npix   = hp.nside2npix(nside)
    sky_map = np.zeros(npix, dtype=np.int64)

    # 4) Accumulate from each GRB run
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

    # 5) Write out FITS
    fits_out = output_base_dir / f"halo_map_nside{nside}.fits"
    hp.write_map(str(fits_out), sky_map, nest=False, overwrite=True)
    print(f"Wrote HEALPix map to {fits_out}")

    # 6) Plot and save Mollweide projection
    plt.figure(figsize=(8, 6))
    hp.mollview(
        sky_map,
        title=f"GRB Pair‐Halo Sky Map (nside={nside})",
        unit="counts",
        norm="hist"
    )
    svg_out = output_base_dir / f"halo_map_nside{nside}.svg"
    plt.savefig(str(svg_out), dpi=300)
    print(f"Saved map plot to {svg_out}")

if __name__ == "__main__":
    main()
