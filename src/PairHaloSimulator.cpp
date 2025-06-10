#include "PairHaloSimulator.h"
#include "helper.h"
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

PairHaloSimulator::PairHaloSimulator(SimulationParams const& params, fs::path const& out_dir)
    : _params(params),
      _out_dir(out_dir),
      _tmp_dir(out_dir / "tmp")
{ /* nothing else here */ }

bool PairHaloSimulator::run() {
    if (fs::exists(_out_dir)) {
        std::cerr << "ERROR: run directory " << _out_dir.string()
             << " already exists. Aborting.\n";
        return false;
    }
    fs::create_directories(_out_dir);
    fs::create_directory(_tmp_dir);

    compute_per_GRB_params();

    initialize_thread_seeds();

    generate_halo_photons();

    merge_thread_files();

    return true;
}

void PairHaloSimulator::compute_per_GRB_params() {
    auto [z, th_src, phi_src, th_v, phi_v] 
        = std::make_tuple(_params.z, _params.th_src, _params.phi_src,
                          _params.th_v, _params.phi_v);

    // comoving distance to source
    auto f1 = [=](double zz) {
        return 3.0e5/(70*std::sqrt(0.3*(1+zz)*(1+zz)*(1+zz)+0.7));
    };
    d_E = adaptiveGaussQuadrature(f1, 0, z);

    // n_src (unit) in galactic frame:
    n_src_hat = sph2cart(th_src, phi_src);
    n_src = d_E * n_src_hat;

    // rotation matrix from jet frame to galactic frame:
    Vec3 n_view_hat = sph2cart(th_v, phi_v);
    R = jet2gal(n_view_hat, n_src_hat);
}

void PairHaloSimulator::initialize_thread_seeds() {
    int max_threads = omp_get_max_threads();
    _thread_seeds.resize(max_threads);
    std::random_device rd;
    for (int t = 0; t < max_threads; ++t) {
        _thread_seeds[t] = rd();
    }
}

std::tuple<double,double> PairHaloSimulator::compute_per_E_params(double E) {
    // exactly your calc_per_E_params:
    double z = _params.z;
    // ... compute z_s, mfp, delta as before ...
    double z_s = calc_z(E, z, 0.0, z);
    auto f2 = [=](double zz) {
        return -3.0e5/(70*std::sqrt(0.3*(1+zz)*(1+zz)*(1+zz)+0.7));
    };
    double mfp = adaptiveGaussQuadrature(f2, z, z_s);
    double delta = 3.0e-6 /(1+z_s)/(1+z_s) * (_params.B0/1.0e-18) 
                   / (0.5*E/10)/(0.5*E/10);
    return {mfp, delta};
}

double PairHaloSimulator::flat_sample_d_gamma(double mfp, std::mt19937& rng) {
    // Sample d_gamma from exponential distribution with mean mfp:
    std::exponential_distribution<double> exp_dist(1.0/mfp);
    return exp_dist(rng);
}

void PairHaloSimulator::generate_halo_photons() {
    auto&  P        = _params;
    double z        = P.z;
    double Ec       = P.Ec;
    double E_trunc  = P.E_trunc;
    double alpha    = P.alpha;
    double jet_open = P.jet_opening;
    int    NUM_E    = P.NUM_E;
    int    NUM_SAMP = P.NUM_SAMPLES_PER_E;
    double B0       = P.B0;

    std::exponential_distribution<double> envelop_dist(1.0/Ec);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> theta_dist(0, jet_open);
    std::uniform_real_distribution<double> phi_dist(0, 2.0*M_PI);

    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);

    std::cout << "Running per‐GRB simulation (z=" << z << ", B0=" << B0
         << ") using " << max_threads 
         << " threads and NUM_E=" << NUM_E 
         << ", samples/E=" << NUM_SAMP << "\n";

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937   rng(_thread_seeds[thread_id]);

        // Create thread‐local files:
        std::ostringstream fname_data, fname_einj;
        fname_data << "data_thread_" << thread_id << ".csv";
        fname_einj << "Einj_thread_" << thread_id << ".csv";
        std::ofstream thread_file  (_tmp_dir / fname_data.str());
        std::ofstream thread_Einj  (_tmp_dir / fname_einj.str());
        thread_file  << "E,theta_obs,phi_obs,T\n";

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < NUM_E; ++i) {
            std::stringstream localBuffer;
            double E;
            // Rejection sampling from f(E) ~ E^{-alpha} exp(−E/Ec):
            while (true) {
                E = envelop_dist(rng);
                if (E < E_trunc) continue;
                double accept_prob = std::pow(E/E_trunc, -alpha);
                if (uni(rng) < accept_prob) break;
            }

            auto [mfp, delta] = compute_per_E_params(E);
            double d_gamma = flat_sample_d_gamma(mfp, rng);

            for (int j = 0; j < NUM_SAMP; ++j) {
                double th_emj = theta_dist(rng);
                double phi_emj = phi_dist(rng);
                Vec3 n_emj_hat = sph2cart(th_emj, phi_emj);
                Vec3 n1_hat = R * n_emj_hat;
                Vec3 n1 = d_gamma * n1_hat;
                Vec3 n2_hit = -n_src - n1;
                Vec3 n2_hit_hat = n2_hit.normalized();
                double n2_norm = n2_hit.norm();

                // rotate n1_hat by delta towards observer:
                Vec3 ax = n1_hat.cross(-n_src_hat).normalized();
                Mat3 R_delta = rotation_matrix(ax, delta);
                Vec3 n2_hat = R_delta * n1_hat;

                double tol = std::cos(2.4e-12 / n2_norm);
                if (std::abs(n2_hat.dot(n2_hit_hat)) >= tol) {
                    auto [th_obs, phi_obs] = cart2sph(-n2_hat);
                    double T = (d_gamma + n2_norm - d_E)*(3.086e19/3.0e5);
                    localBuffer << E << "," 
                                << th_obs << "," 
                                << phi_obs << "," 
                                << T << "\n";
                }
            }
            // Flush once per energy:
            thread_file << localBuffer.str();
            thread_Einj << E << "\n";
        }

        thread_file.close();
        thread_Einj.close();
    } // end parallel
}

void PairHaloSimulator::merge_thread_files() {
    std::ofstream out_data(_out_dir / "data.csv");
    out_data << "E,theta_obs,phi_obs,T\n";
    std::ofstream out_einj(_out_dir / "E_inj.csv");

    std::cout << "Merging thread files in " << _tmp_dir.string() << "\n";
    for (auto& entry : fs::directory_iterator(_tmp_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fn = entry.path().filename().string();
        if (fn.rfind("data_thread_", 0) == 0) {
            std::ifstream in(entry.path());
            std::string line;
            getline(in, line);  // skip header
            while (getline(in, line)) {
                out_data << line << "\n";
            }
            in.close();
            fs::remove(entry.path());
        }
        else if (fn.rfind("Einj_thread_", 0) == 0) {
            std::ifstream in(entry.path());
            std::string line;
            while (getline(in, line)) {
                out_einj << line << "\n";
            }
            in.close();
            fs::remove(entry.path());
        }
    }
    out_data.close();
    out_einj.close();
    fs::remove(_tmp_dir);
    std::cout << "Wrote data.csv and E_inj.csv into " << _out_dir.string() << "\n";
}