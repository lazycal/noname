#include "worker.h"

class LossyPushPull : public PushPullProtocol {
private:
  ThreadSafeQueue<LayerInfo *> task_q, handshake_q, shutdown_q;
  UDPSocket udp;
public:
  void push(std::string name) override {
    auto li = lis.find(name)->second;
    MYLOG(1) << "sending " << name;
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
    
    udp.connect(config_get_PS_ip_port(1));
    std::cout << "Sender Connected to " << config_get_PS_ip_port(1)
              << "! Sending HANDSHAKE\n";
    std::string port_s = config_get_worker_ip() + ":" + std::to_string(port_num);
    udp.send(HAND_SHAKE_MSG+port_s);

    std::cout << "Receiving HANDSHAKE\n";
    Message message = udp.recv();
    assert(message.size() == sizeof(Request));
    Request *p = reinterpret_cast<Request*>(message.data());
    assert(p->msg == HAND_SHAKE_MSG);
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
  }
  void run_recver() {
  }
  void shutdown() override {
    task_q.enqueue(nullptr);
    shutdown_q.dequeue();
  }

};
