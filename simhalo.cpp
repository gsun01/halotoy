#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <cmath>
#include <tuple>
#include <random>
#include <omp.h>

#include "helper.h"

namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Simulation config
constexpr int NUM_E = 10'000'000;
constexpr int NUM_SAMPLES_PER_E = 100'000;
fs::path data_dir = "/data/sguotong/data/halotoy";
fs::path group_dir = data_dir / "test_0606_02/";
// GRB parameters
constexpr double z = 0.151;                         // redshift of GRB
constexpr double jet_opening = 1.5*M_PI/180.0;      // jet half-opening angle, radians
constexpr double th_v = 0.0*M_PI/180.0;             // viewing angle in the galactic frame, polar angle
constexpr double phi_v = 00.0*M_PI/180.0;           // viewing angle in the galactic frame, azimuthal angle
constexpr double th_src = 0.0;                      // source direction in the galactic frame, polar angle
constexpr double phi_src = 0.0;                     // source direction in the galactic frame, azimuthal angle
                                                    // jet and viewing angles of GRB221009A: https://arxiv.org/pdf/2301.01798
// Source spectrum, dN/dE ~ E^-alpha exp(-E/Ec)
constexpr double alpha = 2.5;                       // intrinsic spectral index
constexpr double Ec = 20.0;                         // intrinsic cutoff energy; TeV
constexpr double E_trunc = 10.0;                    // lower bound on energy sampling; TeV
// IGMF parameters
constexpr double B0 = 1.0e-15;                      // current epoch IGMF strength, Gauss
///////////////////////////////////////////////////////////////////////////////////////////////////////////


std::tuple<std::string,std::string> create_run_dirs(fs::path const& group_dir, double z, 
                                                     double jet_opening, double th_v, double phi_v) {
    std::ostringstream oss;
    oss << "z" << to_string(z, 3)
        << "_B"   << to_string(-std::log10(B0), 0)
        << "_j"   << to_string(jet_opening * 180.0 / M_PI, 1)
        << "_v"   << to_string(th_v * 180.0 / M_PI, 1)
        << "_phv" << to_string(phi_v * 180.0 / M_PI, 1);

    // create run directory
    fs::path run_dir = group_dir / oss.str();
    std::cout << "Creating run directory: " << run_dir.string() << std::endl;
    if (fs::exists(run_dir)) {
        std::cerr << "Run directory already exists." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    try {
        fs::create_directories(run_dir);
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating run directory " << run_dir.string()
                  << ": " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // create tmp subdir 
    fs::path tmp_dir = run_dir / "tmp";
    std::cout << "Creating temporary directory: " << tmp_dir << std::endl;
    try {
        fs::create_directory(tmp_dir);
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating tmp directory " << tmp_dir.string()
                  << ": " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return {run_dir, tmp_dir};
}

void serial_seeding(std::vector<uint32_t>& thread_seeds, int max_threads) {
    std::random_device rd;
    for (int t = 0; t < max_threads; ++t) {
        thread_seeds[t] = rd();
    }
    return;
}

std::tuple<double,Vec3,Vec3,Mat3> calc_per_GRB_params(double z, double th_src, double phi_src, 
                                            double th_v, double phi_v) {
    // d_E: comoving distance to GRB
    auto f1 = [=](double zz) {
      return 3.0e5/(70*std::sqrt(0.3*(1+zz)*(1+zz)*(1+zz)+0.7));
    };
    double d_E = adaptiveGaussQuadrature(f1, 0, z);
    
    // n_src: source direction in the galactic frame
    Vec3 n_src_hat = sph2cart(th_src, phi_src);
    Vec3 n_src = d_E * n_src_hat;


    // R: rotation matrix from jet frame to galactic frame
    Vec3 n_view_hat = sph2cart(th_v, phi_v);
    Mat3 R = rot2gal(n_view_hat, n_src_hat);

    return {d_E, n_src_hat, n_src, R};
}

std::tuple<double,double> calc_per_E_params(double z, double E, double B0) {
    // z_s: redshift of scattering event
    double z_s = calc_z(E, z, 0.0, z);

    // mfp: comoving mean free path of photon before scattering
    auto f2 = [=](double zz) {
      return -3.0e5/(70*std::sqrt(0.3*(1+zz)*(1+zz)*(1+zz)+0.7));
    };
    double mfp = adaptiveGaussQuadrature(f2, z, z_s);

    // delta: scattering angle in radians
    double delta = 3.0e-6 /(1+z_s)/(1+z_s) * (B0/1.0e-18) /(0.5*E/10)/(0.5*E/10);

    return {mfp, delta};
}

double flat_sample_d_gamma(double mfp, std::mt19937& rng) {
    std::exponential_distribution<double> exp_dist(1.0/mfp);
    // sample from the exponential distribution
    double trial = exp_dist(rng);
    double d_gamma = (trial>0 ? trial : -trial);    // only accept positive d_gamma
    return d_gamma;
}

// void importance_sample_d_gamma() {
//     // d_gamma_hit: given E and th_emi there is a unique d_gamma_hit for the photon to cross the Earth
//     double d_gamma_hit = std::sin(delta-th_emi) * d_E / std::sin(delta);

//     // narrow Gaussian centered on d_gamma_hit
//     double d_gamma_sigma = 1.0e-3*d_gamma_hit;
//     double min_sigma = std::numeric_limits<double>::epsilon() * std::abs(d_gamma_hit);
//     d_gamma_sigma = std::max(d_gamma_sigma, min_sigma);
//     std::normal_distribution<double> prop_dist(d_gamma_hit, d_gamma_sigma);
//     // sample from proposal distribution
//     double trial = prop_dist(rng);
//     d_gamma = (trial>0 ? trial : -trial);    // only accept positive d_gamma

//     // weight of the sample = P(d|E,th) / N(d_hit, sig_d)
//     double p_d = std::exp(-d_gamma/mfp) / mfp;    // true distribution: exponential
//     double D = (d_gamma-d_gamma_hit)/d_gamma_sigma;
//     double N_d = 1/(std::sqrt(2*M_PI)*d_gamma_sigma) * std::exp(-0.5*D*D);
//     w = p_d / N_d;

// }

void merge_thread_files(fs::path const& run_dir, fs::path const& tmp_dir) {
    std::ofstream out_data(run_dir / "data.csv");
    out_data << "E,theta_obs,phi_obs,T\n";

    std::ofstream out_einj(run_dir / "E_inj.csv");

    std::cout << "Merging thread files from: " << tmp_dir.string() << "\n";

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
    std::cout << "data.csv and E_inj.csv written in " << run_dir.string() << std::endl;
}

int main() {
    auto [run_dir, tmp_dir] = create_run_dirs(group_dir, z, jet_opening, th_v, phi_v);
    auto [d_E, n_src_hat, n_src, R] = calc_per_GRB_params(z, th_src, phi_src, th_v, phi_v);

    // serialize seeding before parallel region
    int max_threads = omp_get_max_threads();
    std::vector<uint32_t> thread_seeds(max_threads);
    serial_seeding(thread_seeds, max_threads);

    // f(E) := dN/dE ~ E^-alpha * exp(-E/Ec)
    // envelope function for rejection sampling: g(E) = Ec^{-1} * exp(-E/Ec)
    // ratio M := max[f(E)/g(E)] = max[Ec * E^-alpha] = Ec * E_trunc^{-alpha}, s.t. f(E) <= M*g(E) for all E >= E_trunc
    // acceptance probability: p_accept(E) = f(E)/[M*g(E)] = (E/E_trunc)^(-alpha)
    std::exponential_distribution<double> envelop_dist(1.0/Ec);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    // uniform distribution for jet-frame emission angles
    std::uniform_real_distribution<double> theta_dist(0, jet_opening);
    std::uniform_real_distribution<double> phi_dist(0, 2.0*M_PI);


    // --------------------------------------------------------------------------------------
    // Parallel region
    // --------------------------------------------------------------------------------------
    omp_set_num_threads(max_threads);
    std::cout << "Using " << max_threads << " threads to generate " << NUM_E << " energy samples and " 
                << NUM_SAMPLES_PER_E << " samples per energy." << std::endl;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        uint32_t t_seed = thread_seeds[thread_id];
        std::mt19937 rng(t_seed);
        
        // thread-local output files
        std::ofstream thread_file(tmp_dir / ("data_thread_" + std::to_string(thread_id) + ".csv"));
        thread_file << "E,theta_obs,phi_obs,T" << std::endl;
        // injection energy file
        std::ofstream thread_Einj(tmp_dir / ("Einj_thread_" + std::to_string(thread_id) + ".csv"));


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

            auto [mfp, delta] = calc_per_E_params(z, E, B0);
            double d_gamma = flat_sample_d_gamma(mfp, rng);
            
            for (int j = 0; j < NUM_SAMPLES_PER_E; ++j) {
                double th_emj = theta_dist(rng);
                double phi_emj = phi_dist(rng);
                Vec3 n_emj_hat = sph2cart(th_emj, phi_emj); // emission direction in jet frame

                Vec3 n1_hat = R * n_emj_hat;
                Vec3 n1 = d_gamma * n1_hat;
                Vec3 n2_hit = -n_src - n1;          // correct secondary direction
                Vec3 n2_hit_hat = n2_hit.normalized();
                double n2_norm = n2_hit.norm();

                // rotate n1_hat by delta towards observer
                Vec3 ax = n1_hat.cross(-n_src_hat).normalized();
                Mat3 R_delta = rotation_matrix(ax, delta);
                Vec3 n2_hat = R_delta * n1_hat;

                // check if photon is observed
                double tol = std::cos(2.4e-12 / n2_norm);     // angle subtended by 1 AU at the scattering point
                if (std::abs(n2_hat.dot(n2_hit_hat)) >= tol) {
                    auto [th_obs, phi_obs] = cart2sph(-n2_hat);
                    double T = (d_gamma + n2_norm - d_E) * (3.086e19 / 3.0e5); // time in seconds

                    // store photon data in local buffer
                    localBuffer << E << ","
                                << th_obs << ","
                                << phi_obs << ","
                                << T << "\n";
                }
            } // end of th-phi loop

            // flush buffer to thread-local file once per E
            thread_file << localBuffer.str();
            thread_Einj << E << "\n";
        } // end of E loop

        thread_file.close();
        thread_Einj.close();
    }
    
    // --------------------------------------------------------------------------------------
    // End parallel region
    // --------------------------------------------------------------------------------------

    merge_thread_files(run_dir, tmp_dir);
    return 0;
}
