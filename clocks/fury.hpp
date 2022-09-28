/*
Copyright 2022 SuperDARN

See LICENSE for details.

  @file fury.hpp
  This file contains class declarations for the Fury GPSDO clock

*/
#ifndef FURY_H
#define FURY_H

#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <unistd.h>
#include "utils/shared_macros/shared_macros.hpp"

/**
 * @brief      Contains an abstract wrapper for the Fury object.
 */
class Fury{
  public:
    explicit Fury(std::string serial_port);
    bool get_status();
    bool is_locked();

  private:
};

#endif
