#!usr/bin/env python

import sys
import json
from pathlib import Path
import numpy as np

def sample_redshift(z_min, z_max):
    z = np.random.uniform(z_min, z_max)
    return z

def sample_alpha(mean=2.3, sd=0.3, min_val=1.0, max_val=3.5):
    while True:
        a = np.random.normal(mean, sd)
        if min_val <= a <= max_val:
            return a

def sample_loguniform(low, high):
    """Sample from a log-uniform distribution between low and high."""
    return 10**np.random.uniform(np.log10(low), np.log10(high))

def sample_spectral_params():
    alpha = sample_alpha()
    Ec = sample_loguniform(10.0, 1000.0)          # in keV
    E_trunc = np.random.uniform(0.1 * Ec, 0.9 * Ec)
    return alpha, Ec, E_trunc

def sample_angles():
    """Uniformly sample source and viewing angles on the sphere."""
    th_src = np.random.uniform(0, np.pi)
    phi_src = np.random.uniform(0, 2 * np.pi)
    th_v = 0.0
    phi_v = 0.0
    jet_opening = np.random.uniform(0.1, 2.0) * np.pi/180.0  # in radians
    # th_v = np.random.uniform(0, np.pi)
    # phi_v = np.random.uniform(0, 2 * np.pi)
    return th_src, phi_src, th_v, phi_v, jet_opening

def generate_population(B0, NUM_GRB, NUM_E, NUM_SAMPLES_PER_E, z_min, z_max):
    pop = []
    for i in range(NUM_GRB):
        z = sample_redshift(z_min, z_max)
        th_src, phi_src, th_v, phi_v, jet_opening = sample_angles()
        grb = {
            "GRB_id":             f"GRB_{i+1:06d}",
            "z":                  z,
            "th_src":             th_src,
            "phi_src":            phi_src,
            "th_v":               th_v,
            "phi_v":              phi_v,
            "jet_opening":        jet_opening,
            "alpha":              2.5,
            "Ec":                 20.0,
            "E_trunc":            10.0,
            "B0":                 B0,
            "NUM_E":              NUM_E,
            "NUM_SAMPLES_PER_E":  NUM_SAMPLES_PER_E
        }
        pop.append(grb)
    return pop

def load_config(config_file):
    with open(config_file, "r") as cf:
        cfg = json.load(cf)
    return cfg

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <config.json>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}")
        sys.exit(2)
    # Load parameters from config.json
    cfg = load_config(config_path)
    output_file = cfg.get("catalog_file", "/grad/sguotong/projects/halotoy/catalogs/default_catalog.json")
    B0 = cfg.get("B0", 1.0e-15)
    NUM_GRB = cfg.get("NUM_GRB", 1000)
    NUM_E = cfg.get("NUM_E", 100000)
    NUM_SAMPLES_PER_E = cfg.get("NUM_SAMPLES_PER_E", 10000)
    z_min = cfg.get("MIN_z", 0.03)
    z_max = cfg.get("MAX_z", 0.15)

    population = generate_population(B0, NUM_GRB, NUM_E, NUM_SAMPLES_PER_E, z_min, z_max)

    with open(output_file, "w") as f:
        json.dump(population, f, indent=2)
    print(f"Generated {NUM_GRB} GRBs -> {output_file}")

if __name__ == "__main__":
    main()
