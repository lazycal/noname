#pragma once

#include "common.h"
#include "utils.h"
#include <chrono>
#include <cstring>
#include <zmq.hpp>

class PushPullProtocol {
public:
  ThreadSafeQueue<LayerInfo *> ready_q; // store ready layers
  virtual void push(std::string name) = 0;
  virtual void run() = 0;
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
  void run_sender();
  void run_recver(zmq::socket_t &);
  zmq::context_t context{4};

private:
  ThreadSafeQueue<LayerInfo *> task_q, handshake_q;
};
void VanillaPushPull::run_sender() {
  zmq::socket_t sender(context, ZMQ_PUSH);
  sender.connect("tcp://" + config_get_PS_ip_port(1));
  std::cout << "Sender Connected to " << config_get_PS_ip_port(1)
            << "! Sending HANDSHAKE\n";

  sender.send(zmq::const_buffer(HAND_SHAKE_MSG.c_str(), HAND_SHAKE_MSG.size()),
              zmq::send_flags::none);
  auto s = handshake_q.dequeue(); // wait for recv handshake
  ASSERT(s == nullptr) << s;
  for (;;) {
    // work
    auto it = task_q.dequeue(), li = it;
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
    receiver.recv(msg);
    std::string name((char *)msg_meta.data(), msg_meta.size());
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
  zmq::socket_t receiver(context, ZMQ_SUB);
  receiver.connect("pgm://" + config_get_PS_ip_port());
  receiver.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  std::cout << "Receiver bind to " << config_get_PS_ip_port() << "!\n";
  
  auto t = std::thread([&] { run_sender(); });
  std::cout << "Receiving HANDSHAKE\n";
  zmq::message_t message;
  receiver.recv(message);
  assert(std::string((char *)message.data(), message.size()) == HAND_SHAKE_MSG);
  std::cout << "Receiver Connected!\n";
  handshake_q.enqueue(nullptr);
  run_recver(receiver);
}
