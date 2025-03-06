#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "solve_z.h"

class Photon {
public:
  Photon(double energy, double th_p, double phi_p, double z) {
    E = energy;
    phi = phi_p;
    z = z;
    z_s = calc_z(E, z, 0.8*z, 1.2*z);
    mfp = calc_mfp(z_s);
  };

  ~Photon() {};

  // comoving mean free path of photon before scattering
  double calc_mfp(double z) {
    return 0.0;
  }

  // draw from distribution the actual comoving distance traveled 
  // by photon before scattering
  double calc_d() {
    std::random_device rd;
    unsigned int seed = rd();
    std::mt19937 rng(seed);
    std::exponential_distribution<double> exp_dist(1.0/mfp);
    d = exp_dist(rng);
    return d;
  }

  // scattering angle
  double calc_delta(double E) {
    return 0.0;
  }

  double calc_theta() {
    return 0.0;
  }

  bool is_observed() {
    return false;
  }

  double E, theta, phi; // energy of primary photon in TeV and the arrival direction at the observer (screen)
  double z; // redshift of GRB
  double z_s; // redshift of scattering event
  double mfp; // comoving mean free path of photon
  double d; // comoving distance traveled by photon before scattering
  double dE; // comoving distance to GRB
  double T; // time delay due to scattering
};