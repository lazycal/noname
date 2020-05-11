#pragma once

#include "common.h"
#include "utils.h"
#include <zmq.hpp>
#include <cstdlib>
#include "ps.h"

class UDPBcast {
  std::vector<UDPSocket> udps;
  std::vector<std::thread*> thrds;
  ThreadSafeQueue<Message> recv_q;

public:
  UDPBcast(){}
  void init(const std::vector<std::string> &ip_ports) {
    udps.reserve(ip_ports.size());
    for (auto ip_port : ip_ports) {
      udps.emplace_back(recv_q);
      auto &udp = udps.back();
      std::cout << "binding to " + ip_port << "\n";
      udp.bind(ip_port);
      thrds.push_back(new std::thread([&] { udp.run(); }));
    }
  }
  
  std::vector<int> send(Message msg) {
    std::vector<int> res;
    for (auto &sender : udps) {
      res.push_back(sender.send(msg));
    }
    return res;
  }

  void confirm(std::vector<int> seqs) {
    std::this_thread::sleep_for(std::chrono::seconds(5)); // HACK
  }

  Message recv() {return recv_q.dequeue();} 
};

class LossyPS : public PS {
public:
  [[ noreturn ]] void run();

private:
  ThreadSafeQueue<std::string> task_q;
  ThreadSafeQueue<void *> handshake_q;
  zmq::context_t context{4};
  std::vector<std::string> w_ip_port;
  UDPBcast udp;
  void run_sender();
  void run_recver();
};

void LossyPS::run_sender() {
  std::cout << "Senders all connected\n";
  // send back handshake
  auto seqs = udp.send(HAND_SHAKE_MSG);
  udp.confirm(seqs);
  std::cout << "Send back HANDSHAKE\n";
  std::cout << "RUNNING !!!\n";
  for (;;) {
    // work
    auto name = task_q.dequeue();
    if (name == SHUTDOWN_MSG) {
      udp.send(SHUTDOWN_MSG);
      break;
    }
    // auto li = lis.find(name)->second;
    // {
    //   zmq::message_t msg_meta(li->name.size());
    //   memcpy(msg_meta.data(), li->name.c_str(), li->name.size());
    //   auto ress = udp.send(msg_meta, zmq::send_flags::sndmore);
    //   for (auto res : ress) {
    //     ASSERT(res.has_value() && res.value() == li->name.size())
    //         << res.has_value() << ", " << res.value() << ", "
    //         << li->name.size();
    //   }
    // }

    // {
    //   zmq::message_t msg(li->buf, li->size);
    //   auto ress = udp.send(msg, zmq::send_flags::none);
    //   for (auto res : ress)
    //     ASSERT(res.has_value() && res.value() == li->size)
    //         << res.has_value() << ", " << res.value() << ", " << li->size;
    // }

    // send...
    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void LossyPS::run_recver() {
  int term_cnt = 0;
  for (;;) {
    // recv
    // auto msg_name = udp.recv();
    // std::string name((char *)msg_name.data(), msg_name.size());
    // if (name == SHUTDOWN_MSG) {
    //   std::cout << "recved " + name << "\n";
    //   if (++term_cnt == nw) {
    //     task_q.enqueue(name);
    //     std::cout << "Receiver shutting down\n";
    //     break;
    //   }
    //   continue;
    // }
    // auto msg_meta = udp.recv();
    // MYLOG(1) << "recving " << name;
    // auto lim = reinterpret_cast<LayerInfoMini *>(msg_meta.data());
    // int rank = lim->rank;
    // assert(lis.find(name) != lis.end());
    // auto &li = lis.find(name)->second;
    // if (li == nullptr) {
    //   // initialize the buffers for all workers
    //   get_or_register_layer(
    //       name, lim->size,
    //       torch::zeros(lim->size / sizeof(float),
    //                    torch::TensorOptions().dtype(torch::kFloat32)),
    //       lim->priroity, nw);
    // }
    // if (li->w_recv_cnt == 0) { // clear the server's buffer
    //   // memset(li->buf, 0, li->size);
    //   li->tensor.zero_();
    // }
    // // receive
    // receiver.recv(
    //     zmq::mutable_buffer(&(li->worker_bufs[rank].front()), lim->size));
    // float *ps = (float *)li->buf,
    //       *wr = (float *)(&li->worker_bufs[rank].front());
    // // accumulate
    // AT_ASSERT(li->size % sizeof(float) == 0, li->size);
    // int num_params = li->size / sizeof(float);
    // for (int i = 0; i < num_params; ++i)
    //   ps[i] += wr[i];
    // // do the average
    // if (++li->w_recv_cnt == nw) {
    //   for (int i = 0; i < num_params; ++i)
    //     ps[i] /= nw;
    //   task_q.enqueue(name);
    //   li->w_recv_cnt = 0;
    // }
  }
}

[[ noreturn ]] void LossyPS::run() {
  /* 
  * PS create receiver. Workers create sender, receiver
  * Workers -> PS: HANDSHAKE, IP:PORT
  * PS create sender
  * PS -> workers: HANDSHAKE
  */
  std::vector<std::string> ip_ports;
  for (int i = 0; i < this->nw; ++i) ip_ports.push_back(config_get_PS_ip_port(i));
  udp.init(ip_ports);  
  std::cout << "Receiver bind!\n";
  //  Wait for start of batch
  for (int i = 0; i < nw; ++i) {
    std::cout << "recving from " << i << "-th\n";
    auto message = udp.recv();
    std::cout << "got msg!\n";
    assert(message.substr(0, HAND_SHAKE_MSG.size()) == HAND_SHAKE_MSG);
    auto s = message.substr(HAND_SHAKE_MSG.size(), message.size() - HAND_SHAKE_MSG.size());
    std::cout << "received 1 handshake from " << s << "!\n";
  }
  std::cout << "All workers pushed!\n";

  auto t = std::thread([&] { run_sender(); });
  run_recver();
  t.join();
  std::cout << "Shutdown\n";
  exit(0);
}