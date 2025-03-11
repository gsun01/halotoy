#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "solve_z.h"

class Photon {
public:
  Photon(double energy, double th_p, double phi_p, double z, double B) {
    E = energy;
    B0 = B;
    phi = phi_p;
    theta_p = th_p;
    z = z;
    z_s = calc_z(E, z, 0.8*z, 1.2*z);
    calc_mfp();
    calc_d();
    calc_delta(E);
    calc_theta();
    calc_T();
  };

  ~Photon() {};

  // comoving mean free path of photon before scattering
  double calc_mfp() {
    auto f = [=](double z) {
      return -1/(70*std::sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7));
    };
    mfp = simpsonsRule(f, z, z_s, 1000);
    return mfp;
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

  // scattering angle in radians
  double calc_delta(double E) {
    delta = 3.0e-6 * pow((1+z_s),-2) * (B0/1.0e-18) * pow((0.5*E/10),-2);
    return delta;
  }

  double calc_theta() {
    theta = delta - theta_p;
    return theta;
  }

  double calc_T() {
    T = (d+dE*std::sin(theta_p)/std::sin(M_PI-delta)-dE) / 3.0e5;
    return T;
  }

  bool is_observed() {
    return std::abs(
      d * std::cos(theta) + dE * std::sin(theta_p) / std::sin(M_PI-delta) - d
    ) < 1.0e-6;
  }

  double E, theta, phi; // energy of primary photon in TeV and the arrival direction at the observer (screen)
  double theta_p;
  double z; // redshift of GRB
  double z_s; // redshift of scattering event
  double mfp; // comoving mean free path of photon
  double d; // comoving distance traveled by photon before scattering
  double delta; // scattering angle
  double dE; // comoving distance to GRB
  double T; // time delay due to scattering

  double B0; // strength of IGMF in G (free param)
};