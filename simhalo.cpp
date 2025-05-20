#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <cmath>
#include <vector>
#include <random>
#include <omp.h>

#include "helper.h"
#include "photon.h"

namespace fs = std::filesystem;
constexpr int NUM_E = 10'000;
constexpr int NUM_SAMPLES_PER_E = 1'000;
constexpr double alpha = 2.5; // GRB intrinsic spectral index, i.e. dN/dE ~ E^-alpha exp(-E/Ec)
constexpr double Ec = 20.0; // GRB intrinsic cutoff energy; TeV
constexpr double E_trunc = 10.0; // lower bound on energy sampling; TeV
std::string group_dir = "test_0515/";

std::vector<double> calc_per_E_params(double z, double E, double B0, std::mt19937& rng) {
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

void merge_thread_files(std::string& run_dir, std::string& tmp_dir) {
    std::ofstream out_data(run_dir + "data.csv");
    out_data << "E,theta_obs,phi_obs,T,w\n";

    std::ofstream out_einj(run_dir + "E_inj.csv");

    std::cout << "Merging thread files from: " << tmp_dir << "\n";

    for (auto& entry : fs::directory_iterator(tmp_dir)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path();
        auto fn   = path.filename().string();

        std::ifstream in(path);
        if (!in) continue;

        std::string line;
        if (fn.rfind("data_thread_", 0) == 0) {
            // skip header in each thread-data file
            std::getline(in, line);
            while (std::getline(in, line)) {
                out_data << line << "\n";
            }
        }
        else if (fn.rfind("Einj_thread_", 0) == 0) {
            while (std::getline(in, line)) {
                out_einj << line << "\n";
            }
        }
        else {
            continue;
        }

        in.close();
        fs::remove(path);
    }

    out_data.close();
    out_einj.close();
    fs::remove(tmp_dir);
    std::cout << "data.csv and E_inj.csv written in " << run_dir << std::endl;

    // std::string main_file_path = run_dir + "data.csv";
    // std::ofstream main_file(main_file_path);
    // main_file << "E,theta_obs,phi_obs,T,w\n";
    // std::cout << "Merging thread data files into: " << main_file_path << std::endl;

    // for (int t = 0; t < max_threads; ++t) {
    //     std::ifstream thread_file(tmp_dir + "data_thread_" + std::to_string(t) + ".csv");
    //     std::string line;
    //     // skip header
    //     std::getline(thread_file, line);
    //     while (std::getline(thread_file, line)) {
    //         main_file << line << "\n";
    //     }
    //     thread_file.close();
    //     // remove thread-local file
    //     fs::remove(tmp_dir + "data_thread_" + std::to_string(t) + ".csv");
    // }
    // main_file.close();
    // std::cout << "Data written to " << main_file_path << std::endl;

    // // merge injection energy files
    // std::string main_Einj_path = run_dir + "E_inj.csv";
    // std::ofstream main_Einj_file(main_Einj_path);
    // std::cout << "Merging thread E_inj files into: " << main_Einj_path << std::endl;
    // for (int t = 0; t < max_threads; ++t) {
    //     std::ifstream thread_Einj(tmp_dir + "Einj_thread_" + std::to_string(t) + ".csv");
    //     std::string line;
    //     while (std::getline(thread_Einj, line)) {
    //         main_file << line << "\n";
    //     }
    //     thread_Einj.close();
    //     // remove thread-local file
    //     fs::remove(tmp_dir + "Einj_thread_" + std::to_string(t) + ".csv");
    // }
    // main_Einj_file.close();
    // std::cout << "Data written to " << main_Einj_path << std::endl;
}

int main(int argc, char* argv[]) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " z B theta_jet theta_view" << std::endl;
        return 1;
    }

    // create run directory
    std::string run_dir = group_dir + "z" + std::string(argv[1]) + "_B" + std::string(argv[2]) + "_j" + std::string(argv[3]) + "_v" + std::string(argv[4]) + "/";
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

    // f(E) := dN/dE ~ E^-alpha * exp(-E/Ec)
    // envelope function for rejection sampling: g(E) = Ec^{-1} * exp(-E/Ec)
    // ratio M := max[f(E)/g(E)] = max[Ec * E^-alpha] = Ec * E_trunc^{-alpha}, s.t. f(E) <= M*g(E) for all E >= E_trunc
    // acceptance probability: p_accept(E) = f(E)/[M*g(E)] = (E/E_trunc)^(-alpha)
    std::exponential_distribution<double> envelop_dist(1.0/Ec);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    // uniform distribution for jet-frame emission angles
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
        thread_file << "E,theta_obs,phi_obs,T,w" << std::endl;

        // injection energy file
        std::ofstream thread_Einj(tmp_dir + "Einj_thread_" + std::to_string(thread_id) + ".csv");
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < NUM_E; ++i) {
            std::stringstream localBuffer; // thread-local buffer for photon data

            double E;
            while (true) {
                E = envelop_dist(rng);
                // sample from the envelope function
                if (E < E_trunc) { continue;}
                double accept_prob = std::pow(E/E_trunc, -alpha);
                if (uni(rng) < accept_prob) {
                    break;
                }
            }

            std::vector<double> per_E_params = calc_per_E_params(z, E, B, rng);
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
                                << photon.T << ","
                                << photon.w << "\n";
                }
            } // end of th-phi loop

            // flush buffer to thread-local file once per E
            thread_file << localBuffer.str();
            thread_Einj << E << "\n";
        } // end of E loop

        thread_file.close();
        thread_Einj.close();
    } // end of parallel region

    merge_thread_files(run_dir, tmp_dir);
    return 0;
}
