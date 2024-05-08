/*Copyright 2016 SuperDARN*/
#include "options.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

/**
 * @brief      The base class opens the config file and parses the JSON into a
 * boost property tree.
 */
void Options::parse_config_file() {
  char *radar_id = std::getenv("RADAR_ID");
  std::string str(radar_id);
  std::string config_path = "config/";
  config_path = config_path + radar_id + "/" + radar_id + "_config.ini";
  std::ifstream json_file(config_path);
  pt::read_json(json_file, config_pt);
}
