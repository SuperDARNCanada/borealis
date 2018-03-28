#include "zmq_borealis_helpers.hpp"
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <iostream>
std::vector<zmq::socket_t> create_sockets(zmq::context_t &context,
                                          std::vector<std::string> identities,
                                          std::string router_address) {

  //zmq::context_t context(1); // 1 is context num. Only need one per program as per examples
  std::vector<zmq::socket_t> new_sockets;

  for (auto &iden : identities) {
  new_sockets.push_back(zmq::socket_t(context, ZMQ_DEALER));
  new_sockets.back().setsockopt(ZMQ_IDENTITY, iden.c_str(), iden.length());
  new_sockets.back().connect(router_address);
  }

  return new_sockets;

}

std::string recv_data(zmq::socket_t &socket, std::string sender_iden) {
  zmq::multipart_t receiver;
  ERR_CHK_ZMQ(receiver.recv(socket));
  auto sender = receiver.popstr();
  auto empty = receiver.popstr();
  auto data_msg = receiver.popstr();

  if (sender != sender_iden) {
    //todo(keith): maybe assert here instead. implies logical error
  }

  return data_msg;
}

void send_data(zmq::socket_t &socket, std::string recv_iden, std::string &data_msg) {
  zmq::multipart_t sender;
  sender.addstr(recv_iden);
  sender.addstr("");
  sender.addstr(data_msg);
  ERR_CHK_ZMQ(sender.send(socket))
}