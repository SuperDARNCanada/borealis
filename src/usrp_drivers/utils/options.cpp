/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "options.hpp"

namespace pt = boost::property_tree;

/**
 * @brief      The base class opens the config file and parses the JSON into a boost property tree.
 */
void Options::parse_config_file() {
    char *radar_code = std::getenv("RADAR_CODE");
    std::string str(radar_code);
    std::string config_path = "config/";
    config_path = config_path + radar_code + "/" + radar_code + "_config.ini";
    std::ifstream json_file(config_path);
    pt::read_json(json_file, config_pt);
}
