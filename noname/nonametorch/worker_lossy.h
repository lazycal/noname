#include "worker.h"

class LossyPushPull : public PushPullProtocol {
private:
  ThreadSafeQueue<LayerInfo *> task_q, handshake_q, shutdown_q;
  UDPSocket udp;
public:
  void push(std::string name) override {
    auto li = lis.find(name)->second;
    MYLOG(1) << "pushing " << name;
    task_q.enqueue(li);
  }
  void run() override {
    int port_num = -1;
    for (int i = 0; i < 1000; ++i) {
      try {
        port_num = WORKER_BASE_PORT+i+config_get_rank();
        udp.bind(INADDR_ANY, port_num);
        std::cout << "Reciever bound port at " << port_num << '\n';
        break;
      } catch (...) {
      }
    }
    ASSERT(port_num != -1);
    
    int rank = config_get_rank();
    udp.connect(config_get_PS_ip_port(rank));
    std::cout << "Sender Connected to " << config_get_PS_ip_port(rank)
              << "! Sending HANDSHAKE\n";
    std::string port_s = config_get_worker_ip() + ":" + std::to_string(port_num);
    udp.send(create_str_msg(HAND_SHAKE_MSG+port_s));

    std::cout << "Receiving HANDSHAKE\n";
    Message message = udp.recv();
    assert(message.size() == sizeof(Request));
    const Request *p = reinterpret_cast<const Request*>(message.data());
    assert(p->toString() == HAND_SHAKE_MSG);
    std::cout << "Receiver Connected!\n";
    handshake_q.enqueue(nullptr);
    std::cout << "RUNNING !!!\n";

    auto t1 = std::thread([=] { run_sender(); });
    auto t2 = std::thread([=] { run_recver(); });
    t1.join();
    t2.join();
    std::cout << "Shutdown\n";
    shutdown_q.enqueue(nullptr);
  }
  void run_sender() {
    for (;;) {
      // work
      auto it = task_q.dequeue(), li = it;
      if (li == nullptr) {
        std::cout << "sender shutting down\n";
        // TODO: below
        // auto msg = create_str_msg(SHUTDOWN_MSG);
        // udp.send(msg, true);
        // std::cout << "shutdown MSG sent\n";
        break;
      }
      int tmp = cur_iter;
      MYLOG(1) << "sending " << li->name << " with cur_iter=" << cur_iter;
      for (int i = 0; i < CEIL(li->size, SLICE_SIZE); ++i) {
        Message msg; msg.resize(sizeof(Request));
        auto req = reinterpret_cast<Request*>(msg.data());
        req->type = Request::LS;
        if (tmp != cur_iter) {
          MYLOG(2) << "[warning] tmp=" << tmp << " != cur_iter=" << cur_iter << " for " << li->name;
          break;
        }
        // TODO: barrier or lock for cur_iter?
        req->ls.initMeta(li->idx, li->priority, config_get_rank(), i, cur_iter);
        int st = SLICE_SIZE * i;
        memcpy(req->ls.data, li->buf + st, std::min(SLICE_SIZE, (int)li->size - st));
        auto res = udp.send(msg);
      }
      // send...
      // std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  void run_recver() {
  //  Wait for start of batch
    for (;;) {
      // recv
      Message msg = udp.recv();
      if (msg.size() == sizeof(Request))  {
        auto req = reinterpret_cast<Request*>(msg.data());
        std::string name;
        switch (req->type)
        {
        case Request::MSG:
          if (req->toString() == SHUTDOWN_MSG) {
            std::cout << "Receiver shutting down\n";
            return;
          } else {
            ASSERT(false) << "Unrecognized message: "+req->toString()+"\n";
          }
          break;
        
        case Request::LS: {
          auto name = layer_names[req->ls.idx];
          if (req->ls.iter != cur_iter) continue;
          MYLOG(1) << "recving " << name + " " << req->ls;
          ASSERT(lis.find(name) != lis.end()) << name << " not found.";
          auto li = lis.find(name)->second;
          int st = SLICE_SIZE * req->ls.sid;
          auto &slc_it = li->w_slc_it[0][req->ls.sid];
          if (slc_it > cur_iter) continue; // dup package
          slc_it++;
          memcpy(li->buf + st, req->ls.data, std::min(SLICE_SIZE, (int)li->size - st));
          if (layer_enough(++li->w_slc_recv_cnt[0], li->size, SLICE_SIZE)
              && li->w_iter[0] == cur_iter) {
            li->w_iter[0]++; // only count once
            ready_q.enqueue(li);
          }
          break;
        }

        default:
          ASSERT(false) << "Unrecognized type: " << req->type << "\n";
        }
      }
    }
  }
  void shutdown() override {
    task_q.enqueue(nullptr);
    shutdown_q.dequeue();
  }

};
