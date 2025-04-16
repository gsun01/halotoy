#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <random>
#include <sstream>
#include <omp.h>

#include "photon.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " z B theta_jet theta_view" << std::endl;
        return 1;
    }
    // create run directory
    std::string run_dir = "runs/z" + std::string(argv[1]) + "_B" + std::string(argv[2]) + "_j" + std::string(argv[3]) + "_v" + std::string(argv[4]) + "/";
    std::cout << "Creating run directory: " << run_dir << std::endl;
    if (fs::exists(run_dir)) {
        std::cerr << "Run directory already exists. Please remove it or choose a different name." << std::endl;
        return 1;
    }
    std::string command = "mkdir " + run_dir;
    system(command.c_str());

    double z = atof(argv[1]);
    double B = pow(10, -atof(argv[2]));
    double th_j = atof(argv[3])*M_PI/180.0; // convert to radians
    double th_v = atof(argv[4])*M_PI/180.0; // convert to radians
    // jet and viewing angles of GRB221009A: https://arxiv.org/pdf/2301.01798
    
    std::ofstream file(run_dir+"data.csv");
    file << "E,theta_obs,phi_obs,T,d_E,d_gamma,delta\n";
    
    #pragma omp parallel
    {
        // thread-local random number generator.
        std::random_device rd;
        int thread_id = omp_get_thread_num();
        std::mt19937 rng(rd() + thread_id);
        std::uniform_real_distribution<double> energy_dist(2, 20);
        std::uniform_real_distribution<double> theta_dist(0, th_j);
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
                                    // << photon.T << "\n";
                                    << photon.T << ","
                                    << photon.d_E << ","
                                    << photon.d_gamma << ","
                                    << photon.delta << "\n";
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
    std::cout << "Data written to " << run_dir + "data.csv" << std::endl;
    return 0;
}
