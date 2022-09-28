/*
Copyright 2022 SuperDARN

See LICENSE for details.

  @file fury.cpp
  This file contains driver code to interface with a Jackson Labs Fury GPSDO over serial

*/
#include <string>
#include "clocks/fury.hpp"


/**
 * @brief      Creates the Fury GPSDO object given a serial port
 *
 * @param[in]  serial_port      A serial port to access the Fury GPSDO (i.e. /dev/ttyUSB0)
 */
clocks::Fury(std::string serial_port)
{
  // Open the serial port file for read-write
  int port = open(serial_port, O_RDWR);
  if (port < 0) {
    RUNTIME_MSG("Error: " << COLOR_RED(strerror(errno)) << " opening serial port: "
        << COLOR_RED(serial_port));
    return;
  }

  // Change terminal settings
  struct termios fury_port;
  if(tcgetattr(port, &fury_port) != 0) {
    RUNTIME_MSG("Error: " << COLOR_RED(strerror(errno)));
    return;
  }
  // 8-N-1 at 115200 baud, no hardware flow control, no carrier detect
  fury_port.c_cflag |= CS8;     // 8 bits
  fury_port.c_cflag &= ~PARENB; // No parity
  fury_port.c_cflag &= ~CSTOPB; // one stop bit
  fury_port.c_cflag &= ~CRTSCTS;// No hardware flow control
  fury_port.c_cflag |= CLOCAL;  // No carrier detect
  fury_port.c_cflag |= CREAD;   // Read enabled

  // non-canonical mode, disable echo/special interpretations
  fury_port.c_lflag &= ~ICANON;
  fury_port.c_lflag &= ~ECHO;
  fury_port.c_lflag &= ~ECHOE;
  fury_port.c_lflag &= ~ECHONL;
  fury_port.c_lflag &= ~ISIG;

  // No software flow control
  fury_port.c_iflag &= ~IXON;
  fury_port.c_iflag &= ~IXOFF;
  fury_port.c_iflag &= ~IXANY;

  // Disable special interpretations of certain input bytes
  fury_port.c_iflag &= ~IGNBRK;
  fury_port.c_iflag &= ~BRKINT;
  fury_port.c_iflag &= ~PARMRK;
  fury_port.c_iflag &= ~ISTRIP;
  fury_port.c_iflag &= ~INLCR;
  fury_port.c_iflag &= ~IGNCR;
  fury_port.c_iflag &= ~ICRNL;

  // Disable special interpretations of certain output bytes
  fury_port.c_oflag &= ~OPOST;
  fury_port.c_oflag &= ~OPNLCR;

  // Set timeouts to non-blocking mode, return immediately from read()
  //fury_port.c_cc[VTIME] = 0;
  //fury_port.c_cc[VMIN] = 0;

  // Set timeout to 1 second. read() will block for 1 second or until any data is available.
  fury_port.c_cc[VTIME] = 1;
  fury_port.c_cc[VMIN] = 0;

  // Set timeout to 1 second after first byte received, block until 10 chars received
  // Note that this will block indefinitely if no chars are ever received in read()
  //fury_port.c_cc[VTIME] = 1;
  //fury_port.c_cc[VMIN] = 10;

  // Block until 10 chars received with read(), no timeout
  //fury_port.c_cc[VTIME] = 0;
  //fury_port.c_cc[VMIN] = 10;

  // Set baudrate to 115200 on input and output
  cfsetispeed(&fury_port, B115200);
  cfsetospeed(&fury_port, B115200);

  // Save settings
  if (tcsetattr(serial_port, TCSANOW, &fury_port) != 0) {
    RUNTIME_MSG("Error: " << COLOR_RED(strerror(errno)));
  }

}

/**
 * @brief      Queries the Fury GPSDO to ensure it is operating and configured properly
 *
 * @return      bool, status is true if good, false otherwise
 */
bool clocks::get_status() {
  uint32_t bytes_read = 0;
  uint32_t total_bytes = 0;
  // Open the serial port file for read-write
  int port = open(serial_port, O_RDWR);
  if (port < 0) {
    RUNTIME_MSG("Error: " << COLOR_RED(strerror(errno)) << " opening serial port: "
        << COLOR_RED(serial_port));
    return false;
  }
  unsigned char status_query[] = {'S','Y','S','T',':','S','T','A','T','?','\n'};
  write(port, id_query, sizeof(id_query));
  unsigned char reply[1024];
  while(bytes_read > 0) {
    bytes_read = read(port, &reply, sizeof(reply));
    total_bytes += bytes_read;
  }
  DEBUG_MSG(reply);
  if (total_bytes > 0) {
    return true;
  }
  return false;
}

/**
 * @brief       Queries the Fury GPSDO for a gps lock
 *
 * @return      bool, true if locked, false otherwise
 */
bool clocks::is_locked() {
      uint32_t bytes_read = 0;
  uint32_t total_bytes = 0;
  // Open the serial port file for read-write
  int port = open(serial_port, O_RDWR);
  if (port < 0) {
    RUNTIME_MSG("Error: " << COLOR_RED(strerror(errno)) << " opening serial port: "
        << COLOR_RED(serial_port));
    return false;
  }
  unsigned char status_query[] = {'S','Y','N','C',':','L','O','C','K','?','\n'};
  write(port, id_query, sizeof(id_query));
  unsigned char reply[1024];
  while(bytes_read > 0) {
    bytes_read = read(port, &reply, sizeof(reply));
    total_bytes += bytes_read;
  }
  DEBUG_MSG(reply);
  if (total_bytes > 0) {
    return true;
  }
  return false;
}
