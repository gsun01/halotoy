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

  ~Photon();

  // Simulate the reverse scattering event.
  // distanceToSource: the known distance from the observer to the GRB.
  // rng: a random number generator for sampling distributions.
  void simulateScattering(double distanceToSource, std::mt19937 &rng);

  // Get the computed post-scattering (final) direction toward the GRB.
  auto getArrivalDirection() const;

  // Get the computed time delay due to the extra path length (compared to a direct path).
  double getTimeDelay() const;

  // Get the scattering distance (drawn from an exponential distribution).
  double getScatterDistance() const;

  // Get the scattering angle (drawn from its distribution, may depend on energy).
  double getScatterAngle() const;

  // Get the photon energy.
  double getEnergy() const;

  // (Optional) Utility: Rotate a vector around an axis by a given angle (in radians).
  // static auto rotateVector(const std::vector<double> &vec, const std::vector<double> &axis, double angle);

private:
  double calc_mfp(double z) {
    return 0.0;
  }

  // Sample a distance from an exponential distribution with the specified mean.
  double sampleExponential(double mean, std::mt19937 &rng);
  // Sample a scattering angle based on the photon energy (this function can be tuned as needed).
  double sampleScatteringAngle(double energy, std::mt19937 &rng);

  double E, theta, phi; // energy of primary photon in TeV and the arrival direction at the observer (screen)
  double z; // redshift of GRB
  double z_s; // redshift of scattering event
  double mfp; // comoving mean free path of photon
  double d; // comoving distance traveled by photon before scattering
  double dE; // comoving distance to GRB
  double T; // time delay due to scattering
};