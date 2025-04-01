#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "helper.h"

class Photon {
public:
  Photon(double energy, double th_emi, double phi_emi, double redshift, double B) {
    E = energy;
    B0 = B;
    theta_emi = std::abs(th_emi); // polar angle in radians
    phi_obs = phi_emi;
    if (th_emi < 0) {
      phi_obs = fmod(phi_obs+M_PI, 2*M_PI);
    }

    z = redshift;
    is_obs = false;

    calc_dE();
    z_s = calc_z(E, z, 0.0, z);
    calc_mfp();
    calc_d();
    calc_delta();
    calc_theta_obs();

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
    d_E = adaptiveGaussQuadrature(f, 0, z);
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
    d_gamma = exp_dist(rng);
  }

  // scattering angle in radians
  void calc_delta() {
    delta = 3.0e-6 /(1+z_s)/(1+z_s) * (B0/1.0e-18) /(0.5*E/10)/(0.5*E/10);
  }

  // polar angle on screen in radians
  void calc_theta_obs() {
    theta_obs = delta - theta_emi;
  }

  void is_observed() {
    if (delta < theta_emi) {return;}
    double x = d_gamma*std::sin(delta)/std::sin(theta_obs);
    if (std::abs(x-d_E) < (1.0e-6*d_E+1.0e-3)) {
      is_obs = true;
    }
  }

  void calc_T() {
    // convert from Mpc to km and divide by c
    T = (d_gamma+d_E*std::sin(theta_emi)/std::sin(M_PI-delta)-d_E) * (3.086e19 / 3.0e5);
  }

  double E, theta_obs, phi_obs; // energy of primary photon in TeV and the arrival direction at the observer (screen)
  double theta_emi;
  double z; // redshift of GRB
  double z_s; // redshift of scattering event
  double mfp; // comoving mean free path of photon
  double d_gamma; // comoving distance traveled by photon before scattering
  double delta; // scattering angle
  double d_E; // comoving distance to GRB
  double T; // time delay due to scattering
  bool is_obs; // is the photon observed?

  double B0; // strength of IGMF in G (free param)
};