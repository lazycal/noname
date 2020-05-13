#pragma once

#include "common.h"
#include "utils.h"
#include <chrono>
#include <cstring>
#include <zmq.hpp>
#include "udp.h"

const int WORKER_BASE_PORT = 9866;
class PushPullProtocol {
public:
  ThreadSafeQueue<LayerInfo *> ready_q; // store ready layers
  virtual void push(std::string name) = 0;
  virtual void run() = 0;
  virtual void shutdown() = 0;
  virtual ~PushPullProtocol() {}
};

class VanillaPushPull : public PushPullProtocol {
public:
  void push(std::string name) override {
    auto li = lis.find(name)->second;
    MYLOG(1) << "sending " << name;
    task_q.enqueue(li);
  }
  void run() override;
  void run_sender(int);
  void run_recver(zmq::socket_t &);
  void shutdown() override {
    task_q.enqueue(nullptr);
    shutdown_q.dequeue();
  }
  zmq::context_t context{4};

private:
  ThreadSafeQueue<LayerInfo *> task_q, handshake_q, shutdown_q;
};
void VanillaPushPull::run_sender(int port_num) {
  zmq::socket_t sender(context, ZMQ_PUSH);
  sender.connect("tcp://" + config_get_PS_ip_port(1));
  std::cout << "Sender Connected to " << config_get_PS_ip_port(1)
            << "! Sending HANDSHAKE\n";

  sender.send(zmq::const_buffer(HAND_SHAKE_MSG.c_str(), HAND_SHAKE_MSG.size()),
              zmq::send_flags::none);
  std::string port_s = config_get_worker_ip() + ":" + std::to_string(port_num);
  sender.send(zmq::const_buffer(port_s.c_str(), port_s.size()), zmq::send_flags::none);
  auto s = handshake_q.dequeue(); // wait for recv handshake
  ASSERT(s == nullptr) << s;
  for (;;) {
    // work
    auto it = task_q.dequeue(), li = it;
    if (li == nullptr) {
      std::cout << "sender shutting down\n";
      zmq::message_t msg((void*)SHUTDOWN_MSG.c_str(), SHUTDOWN_MSG.size(), nullptr);
      sender.send(msg, zmq::send_flags::none);
      std::cout << "shutdown MSG sent\n";
      break;
    }
    {
      zmq::message_t msg_name(li->name.size());
      memcpy(msg_name.data(), li->name.c_str(), li->name.size());
      auto res = sender.send(msg_name, zmq::send_flags::sndmore);
      ASSERT(res.has_value() && res.value() == li->name.size())
          << res.has_value() << ", " << res.value() << ", " << li->name.size();
    }

    {
      auto lim = LayerInfoMini::fromLayerInfo(*li, config_get_rank());
      zmq::message_t msg_meta(sizeof(lim));
      memcpy(msg_meta.data(), &lim, sizeof(lim));
      auto res = sender.send(msg_meta, zmq::send_flags::sndmore);
      ASSERT(res.has_value() && res.value() == sizeof(lim))
          << res.has_value() << ", " << res.value() << ", " << sizeof(lim);
    }

    zmq::message_t msg(li->buf, li->size, nullptr);
    auto res = sender.send(msg, zmq::send_flags::none);
    ASSERT(res.has_value() && res.value() == li->size)
          << res.has_value() << ", " << res.value() << ", " << li->size;

    // send...
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void VanillaPushPull::run_recver(zmq::socket_t &receiver) {
  //  Wait for start of batch
  for (;;) {
    // recv
    zmq::message_t msg_meta, msg;
    receiver.recv(msg_meta);
    std::string name((char *)msg_meta.data(), msg_meta.size());
    if (name == SHUTDOWN_MSG) {
      std::cout << "Receiver shutting down\n";
      break;
    }
    receiver.recv(msg);
    MYLOG(1) << "recving " << name;
    ASSERT(lis.find(name) != lis.end()) << name << " not found.";
    auto li = lis.find(name)->second;
    ASSERT(li->size == msg.size())
        << "msg.size()=" << msg.size() << ", li->size=" << li->size;
    memcpy(li->buf, msg.data(), li->size);
    ready_q.enqueue(li);
  }
}

void VanillaPushPull::run() {
  zmq::socket_t receiver(context, ZMQ_PULL);
  int port_num = -1;
  for (int i = 0; i < 1000; ++i) {
    try {
      port_num = WORKER_BASE_PORT+i+config_get_rank();
      receiver.bind("tcp://*:"+std::to_string(port_num));
      std::cout << "Reciever bound port at " << port_num << '\n';
      break;
    } catch (...) {
    }
  }
  ASSERT(port_num != -1);
  auto t = std::thread([=] { run_sender(port_num); });
  
  std::cout << "Receiving HANDSHAKE\n";
  zmq::message_t message;
  receiver.recv(message);
  assert(std::string((char *)message.data(), message.size()) == HAND_SHAKE_MSG);
  std::cout << "Receiver Connected!\n";
  handshake_q.enqueue(nullptr);
  std::cout << "RUNNING !!!\n";
  run_recver(receiver);
  t.join();
  std::cout << "Shutdown\n";
  shutdown_q.enqueue(nullptr);
}
