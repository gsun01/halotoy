#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "helper.h"

struct GRB_params {
  double z; // redshift of GRB
  double E; // energy of primary photon in TeV
  double B0; // strength of IGMF in Gauss
  double th_emj; // emission angle w.r.t. the jet axis in radians
  double phi_emj; // emission angle w.r.t. the line of sight in radians
  double th_jet; // half opening angle of jet in radians
  double th_view; // viewing angle in radians

  unsigned int mfp_seed;
};

class Photon {
private:
  void calc_phi_emi() {
    // phi_emi
    if (std::abs(phi_emj-M_PI/2.0) < 1.0e-6) {
      phi_emi = M_PI/2.0;
      return;
    }
    if (std::abs(phi_emj-1.5*M_PI) < 1.0e-6) {
      if (std::abs(th_emj-th_view) < 1.0e-6) {
        // photon comes from the jet axis
        phi_emi = 0.0;
        return;
      }
      else if (th_emj < th_view) {
        phi_emi = M_PI/2.0;
        return;
      }
      else {
        phi_emi = M_PI*1.5;
        return;
      }
    }
    double a = std::atan(std::tan(phi_emj) + th_view/th_emj/std::cos(phi_emj));
    if (0 <= phi_emj && phi_emj < M_PI/2.0) { phi_emi = a;}
    else if (M_PI/2.0 < phi_emj && phi_emj < 1.5*M_PI) { phi_emi = M_PI+a;}
    else if (1.5*M_PI < phi_emj && phi_emj < 2.0*M_PI) { phi_emi = fmod(2.0*M_PI + a, 2.0*M_PI);}
    return;
  }

  void calc_th_emi() {
    // th_emi
    if (th_emj < 1.0e-6) {
      // photon comes from the jet axis
      th_emi = th_view;
      return;
    }
    th_emi = (th_emj * std::sin(phi_emj) + th_view) / std::sin(phi_emi);
  }

  void calc_emitting_angles() {
    calc_phi_emi();
    calc_th_emi();
  }

  void calc_distances() {
    // d_E: comoving distance to GRB
    auto f1 = [=](double z) {
      return 3.0e5/(70*std::sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7));
    };
    d_E = adaptiveGaussQuadrature(f1, 0, z);

    // z_s: redshift of scattering event
    z_s = calc_z(E, z, 0.0, z);

    // mfp: comoving mean free path of photon before scattering
    auto f2 = [=](double z) {
      return -3.0e5/(70*std::sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7));
    };
    mfp = adaptiveGaussQuadrature(f2, z, z_s);

    // d_gamma: comoving distance traveled by photon before scattering
    std::mt19937 rng(mfp_seed);
    std::exponential_distribution<double> exp_dist(1.0/mfp);
    d_gamma = exp_dist(rng);
  }

  // scattering angle in radians
  void calc_delta() {
    delta = 3.0e-6 /(1+z_s)/(1+z_s) * (B0/1.0e-18) /(0.5*E/10)/(0.5*E/10);
  }

  void calc_obs_angles() {
    th_obs = delta - th_emi;
    phi_obs = phi_emi;
  }

  void is_observed() {
    if (delta < th_emi) {return;}
    double x = d_gamma*std::sin(delta)/std::sin(th_obs);
    if (std::abs(x-d_E) < (1.0e-6*d_E+1.0e-3)) {
      is_obs = true;
    }
  }

  void calc_T() {
    // convert from Mpc to km and divide by c
    T = (d_gamma+d_E*std::sin(th_emi)/std::sin(M_PI-delta)-d_E) * (3.086e19 / 3.0e5);
  }

public:
  Photon(GRB_params &params)
    : E(params.E),
      th_emj(params.th_emj),
      phi_emj(params.phi_emj),
      z(params.z),
      B0(params.B0),
      th_jet(params.th_jet),
      th_view(params.th_view),
      is_obs(false),
      mfp_seed(params.mfp_seed) {};

  ~Photon() {};

  void propagate_photon() {    
    calc_emitting_angles();
    calc_distances();
    calc_delta();
    calc_obs_angles();
    is_observed();
    if (is_obs) {
      calc_T();
    }
  }

  double th_emj, phi_emj; // emission angles w.r.t. the jet axis in radians
  double th_emi, phi_emi; // emission angles w.r.t. the line of sight in radians
  double th_jet, th_view; // half opening angle of jet and viewing angle in radians
  double E, th_obs, phi_obs; // energy of primary photon in TeV and the arrival direction 
                             // at the observer in radians (screen)

  double z; // redshift of GRB
  double z_s; // redshift of scattering event
  double mfp; // comoving mean free path of photon
  double d_gamma; // comoving distance traveled by photon before scattering
  double delta; // scattering angle in radians
  double d_E; // comoving distance to GRB
  double T; // time delay due to scattering
  bool is_obs; // is the photon observed?

  double B0; // strength of IGMF in G (free param)
  unsigned int mfp_seed;
};