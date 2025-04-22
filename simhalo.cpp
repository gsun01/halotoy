#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <random>
#include <sstream>
#include <omp.h>

#include "photon.h"

namespace fs = std::filesystem;
constexpr int NUM_E = 10'000;
constexpr int NUM_SAMPLES_PER_E = 100'000'000;

std::vector<double> calc_per_E_params(double z, double E, double B0) {
    // d_E: comoving distance to GRB
    auto f1 = [=](double zz) {
      return 3.0e5/(70*std::sqrt(0.3*(1+zz)*(1+zz)*(1+zz)+0.7));
    };
    double d_E = adaptiveGaussQuadrature(f1, 0, z);

    // z_s: redshift of scattering event
    double z_s = calc_z(E, z, 0.0, z);

    // mfp: comoving mean free path of photon before scattering
    auto f2 = [=](double zz) {
      return -3.0e5/(70*std::sqrt(0.3*(1+zz)*(1+zz)*(1+zz)+0.7));
    };
    double mfp = adaptiveGaussQuadrature(f2, z, z_s);

    // delta: scattering angle in radians
    double delta = 3.0e-6 /(1+z_s)/(1+z_s) * (B0/1.0e-18) /(0.5*E/10)/(0.5*E/10);

    std::vector<double> res = {d_E, mfp, delta};

    return res;
}

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
    fs::create_directories(run_dir);
    // create tmp directory
    std::string tmp_dir = run_dir + "tmp/";
    std::cout << "Creating temporary directory: " << tmp_dir << std::endl;
    fs::create_directories(tmp_dir);

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
        uint32_t t_seed = thread_seeds[thread_id];
        std::mt19937 rng(t_seed);
        
        // thread-local output files
        std::ofstream thread_file(tmp_dir + "data_thread_" + std::to_string(thread_id) + ".csv");
        // write header once per file
        thread_file << "E,theta_obs,phi_obs,T,w,th_emj,th_emi,delta" << std::endl;
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < NUM_E; ++i) {
            std::stringstream localBuffer; // thread-local buffer for photon data

            double E = energy_dist(rng);
            std::vector<double> per_E_params = calc_per_E_params(z, E, B);
            double d_E = per_E_params[0];
            double mfp = per_E_params[1];
            double delta = per_E_params[2];
            
            for (int j = 0; j < NUM_SAMPLES_PER_E; ++j) {
                double th_emj = theta_dist(rng);
                double phi_emj = phi_dist(rng);

                GRB_params params {d_E, mfp, delta, th_emj, phi_emj, th_j, th_v, rng};

                Photon photon(params);
                photon.propagate_photon();
                // Check if the photon is observed
                if (photon.is_obs) {
                    localBuffer << E << ","
                                << photon.th_obs << ","
                                << photon.phi_obs << ","
                                // << photon.T << "\n";
                                << photon.T << ","
                                << photon.w << ","
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
    main_file << "E,theta_obs,phi_obs,T,w,th_emj,th_emi,delta\n";
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
    fs::remove(tmp_dir);
    std::cout << "Data written to " << main_file_path << std::endl;
    return 0;
}
