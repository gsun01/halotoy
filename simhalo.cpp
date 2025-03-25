#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <sstream>
#include <omp.h>

#include "photon.h"

int main() {
    double z = 0.151;
    double B = 1.0e-15;
    // jet and viewing angles of GRB221009A: https://arxiv.org/pdf/2301.01798
    double theta_jet = 1.5 * M_PI/180.0;
    // double theta_view = 2.0/3.0 * theta_jet;
    double theta_view = 0.0;

    // // if theta_jet is the full opening angle
    // double theta_emi_min = theta_view - 0.5*theta_jet;
    // double theta_emi_max = theta_view + 0.5*theta_jet;

    // if theta_jet is the half opening angle
    double theta_emi_min = theta_view - theta_jet;
    double theta_emi_max = theta_view + theta_jet;
    
    std::ofstream file("photon_data.csv");
    file << "E,theta_obs,phi_obs,T\n";
    
    #pragma omp parallel
    {
        // thread-local random number generator.
        std::random_device rd;
        int thread_id = omp_get_thread_num();
        std::mt19937 rng(rd() + thread_id);
        std::uniform_real_distribution<double> energy_dist(2, 20);
        std::uniform_real_distribution<double> thetap_dist(theta_emi_min, theta_emi_max);
        std::uniform_real_distribution<double> phi_dist(-M_PI, M_PI);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < 100000; ++i) {
            std::stringstream localBuffer; // thread-local buffer for photon data
            
            // if (i % 1000 == 0) {
            //     #pragma omp critical
            //     std::cout << "Simulating photon " << i << std::endl;
            // }
            
            double E = energy_dist(rng);
            
            for (int j = 0; j < 1000; ++j) {
                double theta_p = thetap_dist(rng);
                double phi_p = phi_dist(rng);
                Photon photon(E, theta_p, phi_p, z, B);
                if (photon.is_obs) {
                    localBuffer << photon.E << ","
                                << photon.theta_obs << ","
                                << photon.phi_obs << ","
                                << photon.T << "\n";
                }
            }

            #pragma omp critical
            {
                file << localBuffer.str();
            }
        }
    }
    
    file.close();
    return 0;
}
