// Copyright SuperDARN Canada 2018
#ifndef SRC_USRP_DRIVERS_UTILS_ZMQ_BOREALIS_HELPERS_HPP_
#define SRC_USRP_DRIVERS_UTILS_ZMQ_BOREALIS_HELPERS_HPP_

#include <string>
#include <vector>
#include <zmq.hpp>

#define ERR_CHK_ZMQ(x)         \
  try {                        \
    x;                         \
  } catch (zmq::error_t & e) { \
  }  // TODO(keith): handle error

#define RECV_REPLY(x, y) recv_string(x, y)
#define RECV_REQUEST(x, y) recv_string(x, y)
#define RECV_PULSE(x, y) recv_string(x, y)

#define SEND_REPLY(x, y, z) send_string(x, y, z)
#define SEND_REQUEST(x, y, z) send_string(x, y, z)
#define SEND_PULSE(x, y, z) send_string(x, y, z)

std::vector<zmq::socket_t> create_sockets(zmq::context_t &context,
                                          std::vector<std::string> identities,
                                          std::string router_address);

void send_string(zmq::socket_t &socket, std::string recv_iden,
                 std::string &data_msg);
std::string recv_string(zmq::socket_t &socket, std::string sender_iden);
zmq::message_t recv_message(zmq::socket_t &socket, std::string sender_iden);
void send_message(zmq::socket_t &socket, std::string recv_iden,
                  zmq::message_t &msg);
void router(zmq::context_t &context, std::string router_address);

#endif  // SRC_USRP_DRIVERS_UTILS_ZMQ_BOREALIS_HELPERS_HPP_
