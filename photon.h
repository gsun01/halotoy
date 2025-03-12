#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "helper.h"

class Photon {
public:
  Photon(double energy, double th_p, double phi_p, double redshift, double B) {
    E = energy;
    B0 = B;
    phi = phi_p;  // azimuthal angle in radians
    theta_p = th_p; // polar angle in radians
    z = redshift;
    is_obs = false;

    calc_dE();
    z_s = calc_z(E, z, 0.8*z, 1.2*z);
    calc_mfp();
    calc_d();
    calc_delta();
    calc_theta();

    is_observed();
    if (is_obs) {
      calc_T();
    }
  };

  ~Photon() {};

  void calc_dE() {
    auto f = [=](double z) {
      return 3.0e5/(70*std::sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7));
    };
    dE = adaptiveGaussQuadrature(f, 0, z);
  }

  // comoving mean free path of photon before scattering
  void calc_mfp() {
    auto f = [=](double z) {
      return -3.0e5/(70*std::sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7));
    };
    mfp = adaptiveGaussQuadrature(f, z, z_s);
  }

  // draw from distribution the actual comoving distance traveled 
  // by photon before scattering
  void calc_d() {
    std::random_device rd;
    unsigned int seed = rd();
    std::mt19937 rng(seed);
    std::exponential_distribution<double> exp_dist(1.0/mfp);
    d = exp_dist(rng);
  }

  // scattering angle in radians
  void calc_delta() {
    delta = 3.0e-6 /(1+z_s)/(1+z_s) * (B0/1.0e-18) /(0.5*E/10)/(0.5*E/10);
  }

  // polar angle on screen in radians
  void calc_theta() {
    theta = delta - theta_p;
  }

  void is_observed() {
    // if (delta < theta_p) {return;}
    double x = d*std::sin(delta)/std::sin(theta);
    if (std::abs(x-dE) < 1.0e-1) {
      is_obs = true;
    }
  }

  void calc_T() {
    T = (d+dE*std::sin(theta_p)/std::sin(M_PI-delta)-dE) / 3.0e5;
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
  bool is_obs; // is the photon observed?

  double B0; // strength of IGMF in G (free param)
};