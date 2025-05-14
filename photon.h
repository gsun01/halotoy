#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

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

struct Vec3 { double x,y,z;
  Vec3 operator+(Vec3 o) const {return {x+o.x,y+o.y,z+o.z};}
  Vec3 operator-(Vec3 o) const {return {x-o.x,y-o.y,z-o.z};}
  Vec3 operator*(double s) const {return {x*s,y*s,z*s};}
};
double dot(Vec3 a, Vec3 b){return a.x*b.x + a.y*b.y + a.z*b.z;}

class Photon {
private:
  void calc_emitting_angles() {
    double sin_th_view = std::sin(th_view);
    double cos_th_view = std::cos(th_view);
    double sin_th_emj = std::sin(th_emj);
    double cos_th_emj = std::cos(th_emj);
    double sin_phi_emj = std::sin(phi_emj);
    double cos_phi_emj = std::cos(phi_emj);
    th_emi = std::acos(-sin_th_view*sin_th_emj*cos_phi_emj + cos_th_view*cos_th_emj);
    // (–π, +π]
    phi_emi = std::atan2(sin_th_emj * sin_phi_emj, cos_th_view*sin_th_emj*cos_phi_emj-sin_th_view*cos_th_emj);
    phi_emi -= M_PI/2.0; // [–π/2, +π/2]
    if (phi_emi < 0) phi_emi += 2.0*M_PI; // [0, 2π)
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

  void flat_sample_d_gamma() {
    std::exponential_distribution<double> exp_dist(1.0/mfp);
    // sample from the exponential distribution
    double trial = exp_dist(rng);
    d_gamma = (trial>0 ? trial : -trial);    // only accept positive d_gamma
    w = 1.0;
  }

  void calc_obs_angles() {
    th_obs = std::asin(d_gamma/d_E * std::sin(delta));
    phi_obs = phi_emi;
  }

  void calc_T(double x) {
    // convert from Mpc to km and divide by c
    T = (d_gamma + x - d_E) * (3.086e19 / 3.0e5);
  }

  void is_observed() {
    if (delta < th_emi) {return;}

    // x: "correct" distance traveled by photon after scattering
    double x = std::sqrt(d_E*d_E + d_gamma*d_gamma - 2*d_E*d_gamma*std::cos(th_emi));
    // delta_hit: "correct" scattering angle for the photon to cross the Earth
    double cos_delta_hit = (d_E*std::cos(th_emi) - d_gamma) / x;

    // double tol = std::cos(2.0e-16 / x);     // angle subtended by Earth at the scattering point
    double tol = std::cos(2.4e-12 / x);     // angle subtended by 1 AU at the scattering point
    // double tol = 1.0e-10;   // unrealistically large tolerance

    if (std::abs(std::cos(delta)-cos_delta_hit) < tol) {
      // photon crosses Earth
      is_obs = true;
      calc_T(x);
    }
  }

public:
  Photon(GRB_params &params):
      d_E(params.d_E),
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
    // importance_sample_d_gamma();
    flat_sample_d_gamma();
    calc_obs_angles();
    is_observed();
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