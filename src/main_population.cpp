#include "PairHaloSimulator.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>\n";
        return 1;
    }

    // 1) Load configuration
    fs::path config_path = argv[1];
    std::ifstream cfg_in(config_path);
    if (!cfg_in) {
        std::cerr << "Error: cannot open config file " << config_path << "\n";
        return 2;
    }
    json cfg;
    cfg_in >> cfg;

    // Extract paths from config
    fs::path catalog_file    = cfg["catalog_file"].get<std::string>();
    fs::path output_base_dir = cfg["output_base_dir"].get<std::string>();
    fs::path log_dir         = cfg["log_dir"].get<std::string>();

    // Ensure output and log directories exist
    fs::create_directories(output_base_dir);
    fs::create_directories(log_dir);

    // 2) Read GRB catalog (JSON array)
    std::ifstream in(catalog_file);
    if (!in) {
        std::cerr << "Error: cannot open catalog file " << catalog_file << "\n";
        return 3;
    }
    json bursts;
    in >> bursts;

    // 3) Loop over bursts and run simulation
    for (auto const& item : bursts) {
        SimulationParams params;
        params.z                  = item["z"].get<double>();
        params.th_src             = item["th_src"].get<double>();
        params.phi_src            = item["phi_src"].get<double>();
        params.th_v               = item["th_v"].get<double>();
        params.phi_v              = item["phi_v"].get<double>();
        params.jet_opening        = item["jet_opening"].get<double>();
        params.alpha              = item["alpha"].get<double>();
        params.Ec                 = item["Ec"].get<double>();
        params.E_trunc            = item["E_trunc"].get<double>();
        params.B0                 = item["B0"].get<double>();
        params.NUM_E              = item["NUM_E"].get<int>();
        params.NUM_SAMPLES_PER_E  = item["NUM_SAMPLES_PER_E"].get<int>();

        std::string grb_id        = item["GRB_id"].get<std::string>();

        // Construct output and log file names
        std::ostringstream dir_name;
        dir_name << "GRB_" << grb_id;
        fs::path run_dir = output_base_dir / dir_name.str();
        fs::path out_log = log_dir / (dir_name.str() + ".stdout.log");
        fs::path err_log = log_dir / (dir_name.str() + ".stderr.log");

        // Open log files
        std::ofstream log_out(out_log);
        std::ofstream log_err(err_log);
        if (!log_out || !log_err) {
            std::cerr << "Error: cannot open log files for " << grb_id << "\n";
            continue;
        }

        // Redirect cout and cerr
        auto cout_buf = std::cout.rdbuf(log_out.rdbuf());
        auto cerr_buf = std::cerr.rdbuf(log_err.rdbuf());

        // Run the simulation
        PairHaloSimulator simulator(params, run_dir);
        bool success = simulator.run();

        // Restore output streams
        std::cout.rdbuf(cout_buf);
        std::cerr.rdbuf(cerr_buf);

        // Report overall status
        if (!success) {
            std::cerr << "▶ Skipping " << grb_id << ": run failed\n";
        } else {
            std::cout << "✔ Completed GRB " << grb_id << "\n";
        }
    }

    return 0;
}