/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include "options.hpp"

namespace pt = boost::property_tree;

/**
 * @brief      The base class opens the config file and parses the JSON into a boost property tree.
 */
void Options::parse_config_file() {
    std::ifstream json_file("config.ini");
    pt::read_json(json_file, config_pt);
}