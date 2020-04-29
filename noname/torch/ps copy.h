#pragma once

#include "common.h"
#include "utils.h"
#include <zmq.hpp>

class PS {
public:
  int nw;
  PS() : nw(config_get_size()) {}
  void run() {}
};

class VanillaPS : public PS {
public:
  void run();

private:
  ThreadSafeQueue<std::string> task_q;
  ThreadSafeQueue<void*> handshake_q;
  zmq::context_t context{4};
  void run_sender(zmq::socket_t &);
  void run_recver();
};

void VanillaPS::run_sender(zmq::socket_t &sender) {
  for (;;) {
    // work
    auto name = task_q.dequeue();
    auto li = lis.find(name)->second;
    {
      zmq::message_t msg_meta(li->name.size());
      memcpy(msg_meta.data(), li->name.c_str(), li->name.size());
      auto res = sender.send(msg_meta, zmq::send_flags::sndmore);
      ASSERT(res.has_value() && res.value() == li->name.size())
          << res.has_value() << ", " << res.value() << ", " << li->name.size();
    }
    
    {
      zmq::message_t msg(li->buf, li->size);
      auto res = sender.send(msg, zmq::send_flags::none);
      ASSERT(res.has_value() && res.value() == li->size)
          << res.has_value() << ", " << res.value() << ", " << li->size;
    }
    // send...
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void VanillaPS::run_recver() {
  zmq::socket_t receiver(context, ZMQ_PULL);
  std::cout << "binding " + config_get_PS_ip_port(1) << "\n";
  receiver.bind("tcp://" + config_get_PS_ip_port(1));
  std::cout << "Receiver bind!\n";
  //  Wait for start of batch
  for (int i = 0; i < nw; ++i) {
    std::cout << "recving from " << i << "-th\n";
    zmq::message_t message;
    receiver.recv(message);
    std::cout << "got msg!\n";
    assert(std::string((char *)message.data(), message.size()) ==
           HAND_SHAKE_MSG);
    std::cout << "received 1 handshake!\n";
  }
  std::cout << "All workers pushed!\n";
  handshake_q.enqueue(nullptr);
  for (;;) {
    // recv
    zmq::message_t msg_name, msg_meta;
    receiver.recv(msg_name);
    receiver.recv(msg_meta);
    std::string name((char *)msg_name.data(), msg_name.size());
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

void VanillaPS::run() {
  zmq::socket_t sender(context, ZMQ_PUB);
  std::cout << "binding " + config_get_PS_ip_port() << "\n";
  sender.bind("epgm://" + config_get_PS_ip_port());
  std::cout << "Sender bind!\n";

  auto t = std::thread([&] { run_recver(); });
  auto s = handshake_q.dequeue();
  ASSERT(s == nullptr) << s;
  // send back handshake
  sender.send(zmq::const_buffer(HAND_SHAKE_MSG.c_str(), HAND_SHAKE_MSG.size()),
              zmq::send_flags::none);

  run_sender(sender);
}