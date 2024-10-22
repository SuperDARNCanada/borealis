#include "zmq_borealis_helpers.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <zmq.hpp>
#include <zmq_addon.hpp>
std::vector<zmq::socket_t> create_sockets(zmq::context_t &context,
                                          std::vector<std::string> identities,
                                          std::string router_address) {
  std::vector<zmq::socket_t> new_sockets;

  for (auto &iden : identities) {
    new_sockets.push_back(zmq::socket_t(context, ZMQ_DEALER));
    new_sockets.back().set(zmq::sockopt::routing_id, iden);
    new_sockets.back().connect(router_address);
  }

  return new_sockets;
}

std::string recv_string(zmq::socket_t &socket, std::string sender_iden) {
  zmq::multipart_t receiver;
  ERR_CHK_ZMQ(receiver.recv(socket));
  auto sender = receiver.popstr();
  auto empty = receiver.popstr();
  auto data_msg = receiver.popstr();

  if (sender != sender_iden) {
    // todo(keith): maybe assert here instead. implies logical error
  }

  return data_msg;
}

zmq::message_t recv_message(zmq::socket_t &socket, std::string sender_iden) {
  zmq::multipart_t receiver;
  ERR_CHK_ZMQ(recv_multipart(socket, std::back_inserter(receiver)));
  auto sender = receiver.popstr();
  assert(sender == sender_iden);

  return receiver.remove();
}

void send_message(zmq::socket_t &socket, std::string recv_iden,
                  zmq::message_t &msg) {
  zmq::multipart_t sender;
  sender.addstr(recv_iden);
  sender.addstr("");
  sender.addtyp(msg);
  sender.send(socket);
}

void send_string(zmq::socket_t &socket, std::string recv_iden,
                 std::string &data_msg) {
  zmq::multipart_t sender;
  sender.addstr(recv_iden);
  sender.addstr("");
  sender.addstr(data_msg);
  sender.send(socket);
}

void router(zmq::context_t &context, std::string router_address) {
  zmq::socket_t router(context, ZMQ_ROUTER);
  router.set(zmq::sockopt::router_mandatory, 1);
  router.bind(router_address);

  while (1) {
    zmq::multipart_t input;
    ERR_CHK_ZMQ(input.recv(router));
    auto sender = input.popstr();
    auto receiver = input.popstr();
    auto empty = input.popstr();
    auto data_msg = input.popstr();

    auto sent = false;
    while (!sent) {
      try {
        zmq::multipart_t output;
        output.addstr(receiver);
        output.addstr(sender);
        output.addstr("");
        output.addstr(data_msg);
        output.send(router);
        sent = true;
      } catch (zmq::error_t &e) {
        std::cout << "Can't send. Sleeping..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }
}
