#pragma once

#include "common.h"
#include "utils.h"
#include <zmq.hpp>
#include <cstdlib>
#include "ps.h"
#include "udp.h"

class UDPBcast {
  std::vector<std::unique_ptr<UDPSocket>> udps;
  std::vector<std::thread*> thrds;
  ThreadSafeQueue<Message> recv_q;

public:
  UDPBcast(){}
  void init(const std::vector<std::string> &ip_ports) {
    udps.reserve(ip_ports.size());
    int i = 0;
    for (auto ip_port : ip_ports) {
      udps.emplace_back(std::make_unique<UDPSocket>(i++, &recv_q));
      auto &udp = udps.back();
      std::cout << "binding to " + ip_port << "\n";
      udp->bind(ip_port);
      thrds.push_back(new std::thread([&] { udp->run(); }));
    }
  }
  
  void lossy_bound_send(LayerInfo *li, int iter) {
    for (auto &sender : udps) {
      sender->lossy_bound_send(li, iter);
    }
  }

  std::vector<int> send(Message msg) {
    std::vector<int> res;
    for (auto &sender : udps) {
      res.push_back(sender->send(msg));
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
  auto seqs = udp.send(create_str_msg(HAND_SHAKE_MSG));
  udp.confirm(seqs);
  std::cout << "Send back HANDSHAKE\n";
  std::cout << "RUNNING !!!\n";
  for (;;) {
    // work
    auto name = task_q.dequeue();
    if (name == SHUTDOWN_MSG) {
      udp.send(create_str_msg(SHUTDOWN_MSG));
      std::cout << "PS sender SHUTDOWN\n";
      break;
    }

    auto li = lis.find(name)->second;
    // TODO: copy ps and iter before sending
    int tmp = li->acc_iter;
    li->iter = tmp - 1;
    MYLOG(1) << "sending " << name << " iter=" << tmp - 1;
    udp.lossy_bound_send(li, tmp - 1);
    // for (int i = 0; i < CEIL(li->size, SLICE_SIZE); ++i) {
    //   Message msg; msg.resize(sizeof(Request));
    //   auto req = reinterpret_cast<Request*>(msg.data());
    //   req->type = Request::LS;
    //   req->ls.initMeta(li->idx, li->priority, -1, i, li->acc_iter - 1);
    //   if (tmp != li->acc_iter) {
    //     MYLOG(2) << "[warning] tmp=" << tmp << " != li->acc_iter=" << li->acc_iter << " for " << li->name;
    //     break;
    //   }
    //   int st = SLICE_SIZE * i;
    //   memcpy(req->ls.data, li->ps_buf + st, std::min(SLICE_SIZE, (int)li->size - st));
    //   auto res = udp.send(msg);
    // }
    
  }
}

void LossyPS::run_recver() {
  int term_cnt = 0, iter_cnt = 0;
  for (;;) {
    // recv
    auto msg = udp.recv();
    if (msg.size() == sizeof(Request)) {
      auto req = reinterpret_cast<Request*>(msg.data());
      switch (req->type)
      {
        case Request::MSG:
          if (req->toString() == SHUTDOWN_MSG) {
            std::cout << "recved 1 shutdown\n";
            if (++term_cnt == nw) {
              task_q.enqueue(SHUTDOWN_MSG);
              std::cout << "Receiver all shutting down\n";
              return;
            }
            continue;
          }
        break;
        
        case Request::LS: {
          auto name = layer_names[req->ls.idx];
          assert(lis.find(name) != lis.end());
          auto &li = lis.find(name)->second;
          if (li == nullptr) {
            // initialize the buffers for all workers
            auto size = layer_sizes[req->ls.idx];
            get_or_register_layer(
                name, size,
                torch::zeros(size / sizeof(float),
                            torch::TensorOptions().dtype(torch::kFloat32)),
                req->ls.priority, nw);
          }
          if (req->ls.iter != li->acc_iter) continue;
          ASSERT(req->ls.iter >= iter_cnt) << "req->ls.iter=" << req->ls.iter << "iter_cnt=" << iter_cnt;
          iter_cnt = std::max(iter_cnt, req->ls.iter);
          int rank = req->ls.rank;
          MYLOG(1) << "recving " << name + " " << req->ls << " " << "iter=" << li->w_iter[rank];
          // if (li->w_iter[rank] > li->acc_iter) continue; // TEST
          if (li->w_recv_cnt == 0 && li->dirty) { // clear the server's buffer
            memset(li->ps_buf[li->acc_iter&1], 0, li->size);
            // li->tensor.zero_();
            li->dirty = false;
          }
          // copy
          int st = SLICE_SIZE * req->ls.sid, plen = std::min(SLICE_SIZE, (int)li->size - st);
          // memcpy(li->worker_bufs[rank].data() + st, req->ls.data, plen);
          auto &slc_it = li->w_slc_it[rank][req->ls.sid];
          ASSERT(slc_it <= li->acc_iter + 1) << slc_it << " acc_iter=" << li->acc_iter;
          MYLOG(1) << "test1 " << li->acc_iter << " " << name;
          if (slc_it <= li->acc_iter) {
            slc_it = li->acc_iter + 1;
            float *ps = reinterpret_cast<float*>(li->ps_buf[li->acc_iter&1] + st),
                  *wr = reinterpret_cast<float*>(req->ls.data);
                  //, *wr = (float *)(&li->worker_bufs[rank].front());
            ASSERT(li->size % sizeof(float) == 0) << li->size;
            int num_params = plen / sizeof(float);
            for (int i = 0; i < num_params; ++i)
              ps[i] += wr[i];
            // MYLOG(2) << name << " li->w_slc_recv_cnt[rank]=" << li->w_slc_recv_cnt[rank];
            MYLOG(1) << "test " << li->acc_iter << " " << name;
            if (layer_enough(++li->w_slc_recv_cnt[rank], li->size, SLICE_SIZE)
                && li->w_iter[rank] == li->acc_iter) {
              li->w_iter[rank]++; // only count once for each worker
              // do the average
              if (++li->w_recv_cnt == nw) { // TODO softer?
                int num_params = li->size / sizeof(float);
                float *ps = reinterpret_cast<float*>(li->ps_buf[li->acc_iter&1]);
                for (int i = 0; i < num_params; ++i)
                  ps[i] /= nw;
                li->w_recv_cnt = 0;
                li->acc_iter++; // TODO: race? sender seems to need this.
                for (int i = 0; i < nw; ++i) li->w_slc_recv_cnt[i] = 0;
                li->dirty = true;
                MYLOG(1) << name << " recv done. iter=" << li->acc_iter - 1;
                // end for this iteration for this layer
                // if modify this closing behavior, then dirty should be modified correspondingly
                task_q.enqueue(name);
              }
            }
          }
          // accumulate
        }
      }
    }
  }
}

[[ noreturn ]] void LossyPS::run() {
  std::cout << "Running LossyPS\n";
  std::vector<std::string> ip_ports;
  for (int i = 0; i < this->nw; ++i) ip_ports.push_back(config_get_PS_ip_port(i));
  udp.init(ip_ports);  
  std::cout << "Receiver bind!\n";
  //  Wait for start of batch
  for (int i = 0; i < nw; ++i) {
    std::cout << "recving from " << i << "-th\n";
    auto message = udp.recv();
    std::cout << "got msg!\n";
    assert(message.size() == sizeof(Request));
    auto str = reinterpret_cast<Request*>(message.data())->toString();
    assert(str.substr(0, HAND_SHAKE_MSG.size()) == HAND_SHAKE_MSG);
    auto s = str.substr(HAND_SHAKE_MSG.size(), str.size() - HAND_SHAKE_MSG.size());
    std::cout << "received 1 handshake from " << s << "!\n";
  }
  std::cout << "All workers pushed!\n";

  auto t = std::thread([&] { run_sender(); });
  run_recver();
  t.join();
  std::cout << "Shutdown\n";
  exit(0);
}