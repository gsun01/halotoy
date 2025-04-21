#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <random>
#include <sstream>
#include <omp.h>

#include "photon.h"

namespace fs = std::filesystem;
int NUM_E = 10000;
int NUM_SAMPLES_PER_E = 100000000;

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
    // create tmp directory
    std::string tmp_dir = run_dir + "tmp/";
    std::cout << "Creating temporary directory: " << tmp_dir << std::endl;
    command = "mkdir " + tmp_dir;
    system(command.c_str());

    double z = atof(argv[1]);
    double B = pow(10, -atof(argv[2]));
    double th_j = atof(argv[3])*M_PI/180.0; // convert to radians
    double th_v = atof(argv[4])*M_PI/180.0; // convert to radians
    // jet and viewing angles of GRB221009A: https://arxiv.org/pdf/2301.01798
    


    // serial seeding before parallel region
    std::random_device rd;
    int max_threads = omp_get_max_threads();
    std::cout << "Using " << max_threads << " threads." << std::endl;
    std::cout << "Generating " << NUM_E << " energy samples and " << NUM_SAMPLES_PER_E << " samples per energy." << std::endl;
    std::vector<uint32_t> thread_seeds(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        thread_seeds[t] = rd();
    }


    std::uniform_real_distribution<double> energy_dist(2, 20);
    std::uniform_real_distribution<double> theta_dist(0, th_j);
    std::uniform_real_distribution<double> phi_dist(0, 2.0*M_PI);

    // --------------------------------------------------------------------------------------
    // Parallel region
    // --------------------------------------------------------------------------------------
    omp_set_num_threads(max_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 rng(thread_seeds[thread_id]);
        
        // thread-local output files
        std::ofstream thread_file(tmp_dir + "data_thread_" + std::to_string(thread_id) + ".csv");
        // write header once per file
        thread_file << "E,theta_obs,phi_obs,T,th_emj,th_emi,delta" << std::endl;
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < NUM_E; ++i) {
            std::stringstream localBuffer; // thread-local buffer for photon data

            double E = energy_dist(rng);
            
            for (int j = 0; j < NUM_SAMPLES_PER_E; ++j) {
                double th_emj = theta_dist(rng);
                double phi_emj = phi_dist(rng);
                GRB_params params {z, E, B, th_j, th_v, th_emj, phi_emj};

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
                                << photon.th_emj << ","
                                << photon.th_emi << ","
                                << photon.delta << "\n";
                }
            // end of th-phi loop
            }
            // flush buffer to thread-local file once per E
            thread_file << localBuffer.str();
        // end of E loop
        }
        thread_file.close();
    // end of parallel region
    }


    // --------------------------------------------------------------------------------------
    // Merge thread-local files into the main file
    // --------------------------------------------------------------------------------------
    std::string main_file_path = run_dir + "data.csv";
    std::ofstream main_file(main_file_path);
    main_file << "E,theta_obs,phi_obs,T,th_emj,th_emi,delta\n";
    std::cout << "Merging thread-local files into main file: " << main_file_path << std::endl;

    for (int t = 0; t < max_threads; ++t) {
        std::ifstream thread_file(tmp_dir + "data_thread_" + std::to_string(t) + ".csv");
        std::string line;
        // skip header
        std::getline(thread_file, line);
        while (std::getline(thread_file, line)) {
            main_file << line << "\n";
        }
        thread_file.close();
        // remove thread-local file
        fs::remove(tmp_dir + "data_thread_" + std::to_string(t) + ".csv");
    }
    
    main_file.close();
    std::cout << "Data written to " << main_file_path << std::endl;
    return 0;
}
