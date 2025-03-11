#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

#include "photon.h"

int main() {
    // uniformly sample 10000 photons in energy, theta_p, phi_p, with fixed z = 0.151, B = 1e-15
    double z = 0.151;
    double B = 1.0e-15;
    std::random_device rd;
    unsigned int seed = rd();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> energy_dist(2, 20);
    std::uniform_real_distribution<double> thetap_dist(-M_PI/2.0, M_PI/2.0);
    std::uniform_real_distribution<double> phi_dist(-M_PI, M_PI);

    std::ofstream file("photon_data.csv");
    file << "E,theta_p,phi_p,z,z_s,mfp,d,delta,theta,T\n";
    for (int i = 0; i < 10000; ++i) {
        if (i % 1000 == 0) {std::cout << "Sampling photon " << i << std::endl;}
        double E = energy_dist(rng);
        double theta_p = thetap_dist(rng);
        double phi_p = phi_dist(rng);
        Photon photon(E, theta_p, phi_p, z, B);
        if (photon.is_observed()) {
            file << photon.E << "," << photon.theta << "," << photon.phi << "," <<  photon.T << "\n";
        }
    }
    file.close();
}