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
    double th_j = 1.5 * M_PI/180.0;
    double th_v = 2.0/3.0 * th_j;
    
    std::ofstream file("data.csv");
    file << "E,theta_obs,phi_obs,T\n";
    
    #pragma omp parallel
    {
        // thread-local random number generator.
        std::random_device rd;
        int thread_id = omp_get_thread_num();
        std::mt19937 rng(rd() + thread_id);
        std::uniform_real_distribution<double> energy_dist(2, 20);
        std::uniform_real_distribution<double> theta_dist(-th_j, th_j);
        std::uniform_real_distribution<double> phi_dist(0, 2.0*M_PI);
        
        #pragma omp for schedule(static)
        for (int i = 0; i < 10000; ++i) {
            std::stringstream localBuffer; // thread-local buffer for photon data

            double E = energy_dist(rng);
            
            for (int j = 0; j < 100000; ++j) {
                double th_emj = theta_dist(rng);
                for (int k = 0; k < 1000; ++k) {
                    // Generate a random azimuthal angle
                    double phi_emj = phi_dist(rng);
                    // Create a Photon object and propagate it
                    GRB_params params;
                    params.z = z;
                    params.E = E;
                    params.B0 = B;
                    params.th_emj = th_emj;
                    params.phi_emj = phi_emj;
                    params.th_jet = th_j;
                    params.th_view = th_v;
                    params.mfp_seed = rd() + thread_id;
                    Photon photon(params);
                    photon.propagate_photon();
                    // Check if the photon is observed
                    if (photon.is_obs) {
                        localBuffer << photon.E << ","
                                    << photon.th_obs << ","
                                    << photon.phi_obs << ","
                                    << photon.T << "\n";
                    }
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
