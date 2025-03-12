#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <sstream>
#include <omp.h>

#include "photon.h"

int main() {
    // fixed parameters
    double z = 0.151;
    double B = 1.0e-15;
    
    // open output file
    std::ofstream file("photon_data.csv");
    file << "E,theta,phi,T\n";
    
    // Parallel region starts here.
    #pragma omp parallel
    {
        // Create thread-local random number generator.
        std::random_device rd;
        int thread_id = omp_get_thread_num();
        std::mt19937 rng(rd() + thread_id);
        std::uniform_real_distribution<double> energy_dist(2, 20);
        std::uniform_real_distribution<double> thetap_dist(-M_PI/5000.0, M_PI/5000.0);
        std::uniform_real_distribution<double> phi_dist(-M_PI, M_PI);
        
        // Parallelize the outer loop with OpenMP.
        #pragma omp for schedule(static)
        for (int i = 0; i < 1000000; ++i) {
            std::stringstream localBuffer; // thread-local buffer for photon data
            
            // Optionally, print progress (protected by critical section).
            if (i % 1000 == 0) {
                #pragma omp critical
                std::cout << "Simulating photon " << i << std::endl;
            }
            
            // Sample energy once per outer iteration.
            double E = energy_dist(rng);
            
            // Inner loop: simulate 1000 photons.
            for (int j = 0; j < 1000; ++j) {
                double theta_p = thetap_dist(rng);
                double phi_p = phi_dist(rng);
                Photon photon(E, theta_p, phi_p, z, B);
                if (photon.is_obs) {
                    localBuffer << photon.E << ","
                                << photon.theta << ","
                                << photon.phi << ","
                                << photon.T << "\n";
                }
            }
            // Write the buffered results to the file in a critical section.
            #pragma omp critical
            {
                file << localBuffer.str();
            }
        }
    }
    
    file.close();
    return 0;
}
