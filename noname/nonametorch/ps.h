#pragma once

#include "common.h"
#include "utils.h"
#include <zmq.hpp>
#include <cstdlib>
#include "udp.h"

class PS {
public:
  int nw;
  PS() : nw(config_get_size()) {}
  void run() {}
};
class ReliableBcast {
  std::vector<zmq::socket_t> senders;
public:
  ReliableBcast(const std::vector<std::string> &ip_ports,
                zmq::context_t &context) {
    senders.reserve(ip_ports.size());
    for (auto ip_port : ip_ports) {
      senders.emplace_back(context, ZMQ_PUSH);
      auto &sender = senders.back();
      std::cout << "connecting to " + ip_port << "\n";
      sender.connect("tcp://" + ip_port);
    }
  }
  template <typename... Args>
  std::vector<zmq::send_result_t> send_deep_copy(Args &&... args) {
    std::vector<zmq::send_result_t> res;
    for (auto &sender : senders)
      res.push_back(sender.send(std::forward<Args>(args)...));
    return res;
  }
  template <typename... Args>
  std::vector<zmq::send_result_t> send(zmq::message_t &msg, Args &&... args) {
    std::vector<zmq::send_result_t> res;
    for (auto &sender : senders) {
      zmq::message_t m; m.copy(msg);
      res.push_back(sender.send(m, std::forward<Args>(args)...));
    }
    return res;
  }
};
class VanillaPS : public PS {
public:
  [[ noreturn ]] void run();

private:
  ThreadSafeQueue<std::string> task_q;
  ThreadSafeQueue<void *> handshake_q;
  zmq::context_t context{4};
  std::vector<std::string> w_ip_port;
  void run_sender();
  void run_recver(zmq::socket_t &);
};

void VanillaPS::run_sender() {
  ReliableBcast sender(w_ip_port, context);
  std::cout << "Senders all connected\n";
  // send back handshake
  sender.send_deep_copy(zmq::const_buffer(HAND_SHAKE_MSG.c_str(), HAND_SHAKE_MSG.size()),
              zmq::send_flags::none);
  std::cout << "Send back HANDSHAKE\n";
  std::cout << "RUNNING !!!\n";
  for (;;) {
    // work
    auto name = task_q.dequeue();
    if (name == SHUTDOWN_MSG) {
      zmq::message_t msg((void*)SHUTDOWN_MSG.c_str(), SHUTDOWN_MSG.size(), nullptr);
      sender.send(msg, zmq::send_flags::none);
      break;
    }
    auto li = lis.find(name)->second;
    {
      zmq::message_t msg_meta(li->name.size());
      memcpy(msg_meta.data(), li->name.c_str(), li->name.size());
      auto ress = sender.send(msg_meta, zmq::send_flags::sndmore);
      for (auto res : ress) {
        ASSERT(res.has_value() && res.value() == li->name.size())
            << res.has_value() << ", " << res.value() << ", "
            << li->name.size();
      }
    }

    {
      zmq::message_t msg(li->buf, li->size);
      auto ress = sender.send(msg, zmq::send_flags::none);
      for (auto res : ress)
        ASSERT(res.has_value() && res.value() == li->size)
            << res.has_value() << ", " << res.value() << ", " << li->size;
    }
    // send...
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void VanillaPS::run_recver(zmq::socket_t &receiver) {
  int term_cnt = 0;
  for (;;) {
    // recv
    zmq::message_t msg_name, msg_meta;
    receiver.recv(msg_name);
    std::string name((char *)msg_name.data(), msg_name.size());
    if (name == SHUTDOWN_MSG) {
      std::cout << "recved " + name << "\n";
      if (++term_cnt == nw) {
        task_q.enqueue(name);
        std::cout << "Receiver shutting down\n";
        break;
      }
      continue;
    }
    receiver.recv(msg_meta);
    MYLOG(1) << "recving " << name;
    auto lim = reinterpret_cast<LayerInfoMini *>(msg_meta.data());
    int rank = lim->rank;
    assert(lis.find(name) != lis.end());
    auto &li = lis.find(name)->second;
    if (li == nullptr) {
      // initialize the buffers for all workers
      get_or_register_layer(
          name, lim->size,
          torch::zeros(lim->size / sizeof(float),
                       torch::TensorOptions().dtype(torch::kFloat32)),
          lim->priroity, nw);
    }
    if (li->w_recv_cnt == 0) { // clear the server's buffer
      // memset(li->buf, 0, li->size);
      li->tensor.zero_();
    }
    // receive
    receiver.recv(
        zmq::mutable_buffer(&(li->worker_bufs[rank].front()), lim->size));
    float *ps = (float *)li->buf,
          *wr = (float *)(&li->worker_bufs[rank].front());
    // accumulate
    AT_ASSERT(li->size % sizeof(float) == 0, li->size);
    int num_params = li->size / sizeof(float);
    for (int i = 0; i < num_params; ++i)
      ps[i] += wr[i];
    // do the average
    if (++li->w_recv_cnt == nw) {
      for (int i = 0; i < num_params; ++i)
        ps[i] /= nw;
      task_q.enqueue(name);
      li->w_recv_cnt = 0;
    }
  }
}

[[ noreturn ]] void VanillaPS::run() {
  /* 
  * PS create receiver. Workers create sender, receiver
  * Workers -> PS: HANDSHAKE, IP:PORT
  * PS create sender
  * PS -> workers: HANDSHAKE
  */
  std::cout << "Running VanillaPS\n";
  zmq::socket_t receiver(context, ZMQ_PULL);
  std::cout << "binding " + config_get_PS_ip_port(1) << "\n";
  receiver.bind("tcp://" + config_get_PS_ip_port(1));
  std::cout << "Receiver bind!\n";
  //  Wait for start of batch
  w_ip_port.clear();
  for (int i = 0; i < nw; ++i) {
    std::cout << "recving from " << i << "-th\n";
    zmq::message_t message;
    receiver.recv(message);
    std::cout << "got msg!\n";
    assert(std::string((char *)message.data(), message.size()) ==
           HAND_SHAKE_MSG);
    receiver.recv(message);
    w_ip_port.push_back(std::string((char *)message.data(), message.size()));
    std::cout << "received 1 handshake from " << w_ip_port.back() << " !\n";
  }
  std::cout << "All workers pushed!\n";

  auto t = std::thread([&] { run_sender(); });
  run_recver(receiver);
  t.join();
  std::cout << "Shutdown\n";
  exit(0);
}