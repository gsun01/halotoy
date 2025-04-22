#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <limits>

#include "helper.h"

struct GRB_params {
  double d_E; // comoving distance to GRB in Mpc
  double mfp; // comoving mfp of photon before scattering in Mpc
  double delta; // scattering angle in radians

  double th_emj; // emission angle w.r.t. the jet axis in radians
  double phi_emj; // emission angle w.r.t. the line of sight in radians
  double th_jet; // half opening angle of jet in radians
  double th_view; // viewing angle in radians

  std::mt19937& rng;
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
    double sin_phi_emi = std::sin(phi_emi);
    if (sin_phi_emi < 1.0e-6) {
      th_emi = std::sqrt(th_emj*th_emj - th_view*th_view);
      return;
    }
    th_emi = (th_emj * std::sin(phi_emj) + th_view) / sin_phi_emi;
  }

  void calc_emitting_angles() {
    calc_phi_emi();
    calc_th_emi();
  }

  void importance_sample_d_gamma() {
    // d_gamma_hit: given E and th_emi there is a unique d_gamma_hit for the photon to cross the Earth
    double d_gamma_hit = std::sin(delta-th_emi) * d_E / std::sin(delta);

    // narrow Gaussian centered on d_gamma_hit
    double d_gamma_sigma = 1.0e-3*d_gamma_hit;
    double min_sigma = std::numeric_limits<double>::epsilon() * std::abs(d_gamma_hit);
    d_gamma_sigma = std::max(d_gamma_sigma, min_sigma);
    std::normal_distribution<double> prop_dist(d_gamma_hit, d_gamma_sigma);
    // sample from proposal distribution
    double trial = prop_dist(rng);
    d_gamma = (trial>0 ? trial : -trial);    // only accept positive d_gamma

    // weight of the sample = P(d|E,th) / N(d_hit, sig_d)
    double p_d = std::exp(-d_gamma/mfp) / mfp;    // true distribution: exponential
    double D = (d_gamma-d_gamma_hit)/d_gamma_sigma;
    double N_d = 1/(std::sqrt(2*M_PI)*d_gamma_sigma) * std::exp(-0.5*D*D);
    w = p_d / N_d;
  }

  void calc_obs_angles() {
    th_obs = delta - th_emi;
    phi_obs = phi_emi;
  }

  void is_observed() {
    if (delta < th_emi) {return;}
    // x: "correct" distance traveled by photon after scattering
    double x = std::sqrt(d_E*d_E + d_gamma*d_gamma - 2*d_E*d_gamma*std::cos(th_emi));
    // delta_hit: "correct" scattering angle for the photon to cross the Earth
    double cos_delta_hit = (d_E*std::cos(th_emi) - d_gamma) / x;
    double tol = 2.0e-16 / x;     // angle subtended by Earth at the scattering point
    // double tol = 10*std::numeric_limits<double>::epsilon(); // 10*machine precision (3 orders higher than the above)
    if (std::abs(std::cos(delta)-cos_delta_hit) < tol) {
      // photon crosses Earth
      is_obs = true;
    }
  }

  void calc_T() {
    // convert from Mpc to km and divide by c
    T = (d_gamma+d_E*std::sin(th_emi)/std::sin(M_PI-delta)-d_E) * (3.086e19 / 3.0e5);
  }

public:
  Photon(GRB_params &params)
    : d_E(params.d_E),
      mfp(params.mfp),
      delta(params.delta),
      d_gamma(0.0),

      th_emj(params.th_emj),
      phi_emj(params.phi_emj),
      th_jet(params.th_jet),
      th_view(params.th_view),
      is_obs(false),
      
      rng(params.rng) {};

  ~Photon() {};

  void propagate_photon() {    
    calc_emitting_angles();
    importance_sample_d_gamma();
    calc_obs_angles();
    is_observed();
    if (is_obs) {
      calc_T();
    }
  }

  double th_emj, phi_emj; // emission angles w.r.t. the jet axis in radians
  double th_emi, phi_emi; // emission angles w.r.t. the line of sight in radians
  double th_jet, th_view; // half opening angle of jet and viewing angle in radians
  double th_obs, phi_obs; // the arrival direction at the observer in radians (screen)

  double mfp; // comoving mfp of photon before scattering in Mpc
  double d_gamma; // comoving distance traveled by photon before scattering
  double delta; // scattering angle in radians
  double d_E; // comoving distance to GRB
  double T; // time delay due to scattering
  bool is_obs; // is the photon observed?

  double w; // weight of the photon (importance sampling)

  std::mt19937& rng; // random number generator
};