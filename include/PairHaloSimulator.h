#pragma once

#include <filesystem>
#include <tuple>
#include <random>
#include "helper.h"

namespace fs = std::filesystem;

struct SimulationParams {
    // Simulation config
    int NUM_E;
    int NUM_SAMPLES_PER_E;
    // GRB parameters
    double z;                                   // redshift
    double jet_opening = 1.5*M_PI/180.0;        // jet half-opening angle, radians
    double th_v, phi_v;                         // viewing angle (rad)
    double th_src, phi_src;                     // source sky coords (rad)
                                                // jet and viewing angles of GRB221009A: https://arxiv.org/pdf/2301.01798
    // Source spectrum, dN/dE ~ E^-alpha exp(-E/Ec)
    double alpha;                               // intrinsic spectral index
    double Ec;                                  // intrinsic cutoff energy; TeV
    double E_trunc;                             // lower bound on energy sampling; TeV
    // IGMF parameters
    double B0;                                  // current epoch IGMF strength, Gauss
};

class PairHaloSimulator {
public:
    PairHaloSimulator(SimulationParams const& params, 
                      fs::path const& out_dir);

    PairHaloSimulator() = delete;

    bool run();

private:
    SimulationParams _params;
    fs::path _out_dir;
    fs::path _tmp_dir;

    // thread-local RNG seeds
    std::vector<uint32_t> _thread_seeds;

    // cached per-GRB params
    double d_E;
    Vec3 n_src_hat;
    Vec3 n_src;
    Mat3 R;

    void compute_per_GRB_params();

    std::tuple<double,double> compute_per_E_params(double E);

    void initialize_thread_seeds();

    double flat_sample_d_gamma(double mfp, std::mt19937& rng);

    void generate_halo_photons();

    void merge_thread_files();

    PairHaloSimulator(PairHaloSimulator const&) = delete;
    PairHaloSimulator& operator=(PairHaloSimulator const&) = delete;
};