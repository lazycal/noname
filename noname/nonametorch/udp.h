#pragma once

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>   
#include <unistd.h>   
#include <errno.h>   
#include <arpa/inet.h>
#include "utils.h"
#include "dbug_logging.h"
#include "common.h"

struct LayerSlice {
  int idx, priority, rank, sid, iter;
  char data[SLICE_SIZE];
  void initMeta(int idx, int priority, int rank, int sid, int iter) {
    this->idx = idx;
    this->priority = priority;
    this->rank = rank;
    this->sid = sid;
    this->iter = iter;
  }
  friend std::ostream& operator<<(std::ostream &os, const LayerSlice &ls) {
    os << "LayerSlice(idx=" << ls.idx << ", rank=" << ls.rank << ", sid=" << ls.sid
      << ", iter=" << ls.iter << ")";
    return os;
  }
};

struct Request {
  enum {
    MSG,
    LS
  } type;
  unsigned short seq;
  union {
    struct {
      int len;
      char data[100];
    } msg;
    struct LayerSlice ls;
  };
  std::string toString() const{
    assert(type == MSG);
    return std::string(msg.data, msg.len);
  }
};

struct Response {
  int /*ack_seq*/iter, idx, sid;
};
static_assert(sizeof(Request) != sizeof(Response), "sizeof(Request) != sizeof(Response)");

using Message=std::string;
Message create_str_msg(std::string s) {
  Message msg;
  msg.resize(sizeof(Request));
  auto req = reinterpret_cast<Request*>(msg.data());
  req->type = Request::MSG;
  req->msg.len = s.size();
  memcpy(req->msg.data, s.data(), s.size());
  return msg;
}

class UDPSocket {
private:
  struct info {
    LayerInfo *li;
    int iter;
    std::chrono::steady_clock::time_point ms;
  };
  ThreadSafeQueue<Message> *recv_q, *send_q;
  ThreadSafeQueue<info> layer_q, ack_q;
  bool del{false};
  int fd, rtt{100}, peer_rank; // TODO: measure rtt
  std::thread *_t;
public:
  bool connected{false};
  UDPSocket(int peer_rank, ThreadSafeQueue<Message> *_recv_q=nullptr): 
    recv_q(_recv_q), peer_rank(peer_rank) {
    if (recv_q == nullptr) {
      recv_q = new ThreadSafeQueue<Message>;
      del = true;
    }
    send_q = new ThreadSafeQueue<Message>;
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    // int recv_size = 2 * 1024 * 1024;
    // setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char *)&recv_size,sizeof(recv_size));  
    _t = new std::thread([=] { this->run(); });
  }
  ~UDPSocket() {
    // _t->join();
    if (del) delete recv_q; 
    delete send_q;
    delete _t;
  }

  void run_layer_timer() {
    for (;;) {
      auto inf = ack_q.dequeue();
      auto t = std::chrono::steady_clock::now();
      if (t < inf.ms) std::this_thread::sleep_for(inf.ms - t);
      layer_q.enqueue(inf);
    }
  }

  void run_layer_sender() {
    for (;;) {
      auto inf = layer_q.dequeue();
      int n_ack = 0;
      LayerInfo *li = inf.li;
      int iter = inf.iter;
      bool expired = false, is_worker = config_get_role() == "worker";
      auto begin = std::chrono::steady_clock::now();
      for (int i = 0; i < CEIL(li->size, SLICE_SIZE); ++i) {
        // ASSERT(li->ack[peer_rank][i] >= iter) << li->ack[peer_rank][i] << " " << iter;
        if (li->ack[peer_rank][i] > iter) {
          n_ack++;
          continue; // acked
        }
        Message msg; msg.resize(sizeof(Request));
        auto req = reinterpret_cast<Request*>(msg.data());
        req->type = Request::LS;
        req->ls.initMeta(li->idx, li->priority, config_get_rank2(), i, iter);
        if (iter != li->iter) {
          MYLOG(1) << "[warning] iter=" << iter << " != li->iter=" << li->iter << " for " << li->name;
          expired = true;
          break;
        }
        int st = SLICE_SIZE * i;
        uint8_t *p_buf = is_worker ? li->buf : li->ps_buf[iter&1];
        memcpy(req->ls.data, p_buf + st, std::min(SLICE_SIZE, (int)li->size - st));
        auto res = this->send(msg);
      }
      if (!expired && !layer_enough(n_ack, li->size, SLICE_SIZE)) {
        inf.ms = begin + std::chrono::milliseconds(rtt);
        ack_q.enqueue(inf);
      } else {
        MYLOG(1) << "Sent " << li->name << " with iter=" << iter << 
        " expired=" << expired << " n_ack=" << n_ack;
      }
    }
  }

  void lossy_bound_send(LayerInfo *li, int iter) {
    layer_q.enqueue(info{li, iter});
  }

  // IMPORTANT: when you send, you must have connected beforehand
  int send(const Message &m) {
    // TODO: congestion control and sequence number
    send_q->enqueue(m);
    return 0;
  }

  Message recv() {
    return recv_q->dequeue();
  }

  void bind(uint32_t addr, int port) {
    struct sockaddr_in servaddr; 
    bzero(&servaddr, sizeof(servaddr));      
    servaddr.sin_addr.s_addr = htonl(addr); 
    servaddr.sin_port = htons(port); 
    servaddr.sin_family = AF_INET;
   
    // bind server address to socket descriptor 
    if (::bind(fd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
      perror("bind failed");
      assert(false);
    }

    char str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(servaddr.sin_addr), str, INET_ADDRSTRLEN);
    std::cout << "binding " << str << ":" << port << "\n";
  }

  void bind(std::string s) {
    int idx = s.find(":"), addr_int;
    auto addr = s.substr(0, idx), port = s.substr(idx + 1, s.size() - idx - 1);
    inet_pton(AF_INET, addr.c_str(), &addr_int);
    bind(ntohl(addr_int), std::stoi(port));
  }

  void connect(uint32_t addr, int port) {
    if (connected) return;
    connected = true;
    struct sockaddr_in servaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_addr.s_addr = htonl(addr);
    servaddr.sin_port = htons(port);
    servaddr.sin_family = AF_INET;
    if (::connect(fd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
      perror("connect failed");
      assert(false);
    }

    char str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(servaddr.sin_addr), str, INET_ADDRSTRLEN);
    std::cout << "connecting " << str << ":" << port << "\n";
  }

  void connect(std::string s) {
    int idx = s.find(":"), addr_int;
    auto addr = s.substr(0, idx), port = s.substr(idx + 1, s.size() - idx - 1);
    inet_pton(AF_INET, addr.c_str(), &addr_int);
    connect(ntohl(addr_int), std::stoi(port));
  }

  void refill(int &bytes_avai, std::chrono::steady_clock::time_point &begin) {
    auto end = std::chrono::steady_clock::now();
    int el = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    bytes_avai = std::min(MAX_BYTES_AVAI * 1ll, 1ll * SR * el);
    begin = end;
  }

  void run_sender() {
    int bytes_avai = MAX_BYTES_AVAI;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (unsigned int i = 0; ; i++) {
      Message m = send_q->dequeue();
      assert(this->connected);
      if (bytes_avai < 0) while (1) {
        refill(bytes_avai, begin);
        if (bytes_avai < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        else break;
        MYLOG(1) << "refilling to " << bytes_avai;
      }
      // if bytes_avai > 0 then considered available for sending
      int n;
      if ((n = ::send(fd, m.data(), m.size(), 0)) == -1) {
        perror("send error");
        assert(false);
      }
      ASSERT(n == sizeof(Request) || n == sizeof(Response)) << n;
      bytes_avai -= n;
      // if ((i & 0xf) == 0) refill(bytes_avai, begin);
    }
  }

  void send_ack(Message &m) {
    const auto rq = reinterpret_cast<Request*>(m.data());
    if (rq->type != Request::LS) return;
    Message am; am.resize(sizeof(Response));
    auto rp = reinterpret_cast<Response*>(am.data());
    rp->idx = rq->ls.idx; rp->sid = rq->ls.sid; rp->iter = rq->ls.iter + 1;
    send(am);
  }

  void recv_ack(Message &m) {
    auto rp = reinterpret_cast<Response*>(m.data());
    auto li = lis.find(layer_names.find(rp->idx)->second)->second;
    if (li->ack[peer_rank][rp->sid] < rp->iter)
      li->ack[peer_rank][rp->sid] = rp->iter;
    // TODO: fence
  }

  void do_ack(Message &m) {
    if (m.size() == sizeof(Request)) send_ack(m);
    else recv_ack(m);
  }

  void run_recver() {
    struct sockaddr_in clent_addr;
    int buffer_len = SLICE_SIZE*2, n;
    socklen_t len = sizeof(clent_addr);;
    auto buffer = new char[buffer_len];
    n = recvfrom(fd, buffer, buffer_len, 0, (struct sockaddr*)&clent_addr, &len);
    if (n == -1) { perror(""); assert(false);  }
    assert(n == sizeof(Request) || n == sizeof(Response));
    connect(ntohl(clent_addr.sin_addr.s_addr), ntohs(clent_addr.sin_port));
    MYLOG(2) << "connected after 1st recv";
    Message m(buffer, buffer+n); // TODO force copy
    do_ack(m);
    if (m.size() == sizeof(Request)) recv_q->enqueue(m);
    for (;;) {
      n = ::recv(fd, buffer, buffer_len, 0);
      if (n == -1) { perror("recv error"); assert(false); }
      ASSERT(n == sizeof(Request) || n == sizeof(Response)) << n;
      Message m(buffer, buffer+n);
      do_ack(m);
      if (m.size() == sizeof(Request)) recv_q->enqueue(m); // TODO force copy
    }
  }

  void run() {
    auto t1 = std::thread([=]{this->run_sender();});
    auto t2 = std::thread([=]{this->run_recver();});
    auto t3 = std::thread([=]{this->run_layer_sender();});
    auto t4 = std::thread([=]{this->run_layer_timer();});
    t1.join();
    t2.join();
    t3.join();
    t4.join();
  }
};
