/*Copyright 2016 SuperDARN*/
#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;

class Options {
 protected:
  //! property tree to hold parsed fields.
  pt::ptree config_pt;

  void parse_config_file();
};

#endif
