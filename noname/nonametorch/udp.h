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
#include <cstdio>

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
  // unsigned short seq;
  long long ts;
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
  long long ts;
};
static_assert(sizeof(Request) != sizeof(Response), "sizeof(Request) != sizeof(Response)");

class Message1 {
  int len{0};
  std::shared_ptr<char>p;
public:
  Message1(char *s, char *t): p(std::shared_ptr<char>(new char[len], std::default_delete<char[]>())) {
    memcpy(p.get(), s, t-s);
  }
  Message1() {}
  Message1& operator=(const Message1 &o) { 
    len = o.len;
    // p = o.p;
    p = std::shared_ptr<char>(new char[len], std::default_delete<char[]>());
    memcpy(p.get(), o.p.get(), len);
    return *this;
  }
  void resize(int nlen) {
    assert(len == 0);
    len = nlen;
    p = std::shared_ptr<char>(new char[len], std::default_delete<char[]>());
  }
  Message1(const Message1 &o) {
    len = o.len;
    // p = o.p;
    p = std::shared_ptr<char>(new char[len], std::default_delete<char[]>());
    memcpy(p.get(), o.p.get(), len);
  }
  char* data() const {
    return p.get();
  }
  size_t size() const {
    return len;
  }
  operator std::string() const { return std::string(p.get(), len); }
};
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

// #ifndef DMLC_RTT
// #define DMLC_RTT 10
// #endif
// #ifndef DMLC_BUC_SZ
// #define DMLC_BUC_SZ 100 // /100
// #endif
const int MAX_BYTES_AVAI = 1000000;
class UDPSocket {
private:
  const double ALPHA = 0.9, BETA = 2;
  struct info {
    LayerInfo *li;
    int iter;
    std::chrono::steady_clock::time_point ms;
    Message m;
  };
  ThreadSafeQueue<Message> *recv_q, *send_q;
  ThreadSafeQueue<info> layer_q, ack_q;
  bool del{false}, autortt;
  int fd, peer_rank, wind_sz, SEND_RATE; // TODO: measure rtt
  int bytes_avai{MAX_BYTES_AVAI};
  double rtt, acc_sr;
  std::thread *_t;
  long long begin;
public:
  bool connected{false};
  UDPSocket(int peer_rank, ThreadSafeQueue<Message> *_recv_q=nullptr): 
    recv_q(_recv_q), peer_rank(peer_rank) {

    begin = get_ms(); //std::chrono::steady_clock::now();
    auto rttp = getenv("DMLC_RTT");
    if (rttp != nullptr) {
      autortt = false;
      rtt = atoi(rttp);
    } else {
      autortt = true;
      rtt = 10;
    }
    auto SEND_RATEp = getenv("SEND_RATE");
    if (SEND_RATEp != nullptr) SEND_RATE = atoi(SEND_RATEp);
    else SEND_RATE = 10000;
    auto wind_szp = getenv("DMLC_BUC_SZ");
    if (wind_szp != nullptr) wind_sz = atoi(wind_szp);
    else wind_sz = 10;
    std::cout << "setting rtt to " << rtt << " autortt=" << autortt << " SEND_RATE to " << SEND_RATE << " BUC_SZ=" << wind_sz << "\n";
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
      if (li == nullptr) { // it's an ack // TODO: mvoe to recv thread
        ASSERT(false) << "li should not be null";
        continue;
      }
      int iter = inf.iter;
      bool expired = false, is_worker = config_get_role() == "worker";
      auto st = std::chrono::steady_clock::now();
      MYLOG(2) << "udp sending " << li->name << " iter=" << iter << " peer=" << peer_rank << " send_q.size=" << send_q->q.size();
      for (int i = 0, tt_slc = CEIL(li->size, SLICE_SIZE); i < tt_slc; ++i) {
        // ASSERT(li->ack[peer_rank][i] >= iter) << li->ack[peer_rank][i] << " " << iter;
        if (li->ack[peer_rank][i] > iter) {
          n_ack++;
          continue; // acked
        }
        // if (i  / 2 * 1. / tt_slc > THRES) break;
        Message msg; msg.resize(sizeof(Request));
        auto req = reinterpret_cast<Request*>(msg.data());
        long long begin_ms = get_ms();
        req->ts = begin_ms;
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
        // auto res = this->send(msg);
        this->_send_msg(msg);
        // ::send(fd, msg.data(), msg.size(), 0);
      }
      if (!expired && !layer_enough(n_ack, li->size, SLICE_SIZE)) {
        inf.ms = st + std::chrono::milliseconds((int)(rtt*BETA));
        ack_q.enqueue(inf);
      } else {
        MYLOG(2) << "Sent " << li->name << " with iter=" << iter << 
        " expired=" << expired << " n_ack=" << n_ack;
      }
    }
  }

  void lossy_bound_send(LayerInfo *li, int iter) {
    layer_q.enqueue(info{li, iter});
  }

  // IMPORTANT: when you send, you must have connected beforehand
  int send(const Message &m) {
    send_q->enqueue(m, wind_sz);
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

  void refill(int &bytes_avai, long long &begin) {
    auto end = get_ms(), el = end - begin;
    bytes_avai = std::min(MAX_BYTES_AVAI * 1ll, bytes_avai + 1ll * SEND_RATE * el);
    begin = end;
    // if ((rand() & 0xff) == 0) MYLOG(3) << "refilling to " << bytes_avai << " with el=" << el/1000. << "s";
  }

  void _send_msg(Message &m) {
      if (bytes_avai < 0) while (1) {
        refill(bytes_avai, begin);
        if (bytes_avai < 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        else break;
      }
      // if ((rand() & 0xff) == 0) MYLOG(3) << "bytes_avai=" << bytes_avai;
      // if bytes_avai > 0 then considered available for sending
      // if (m.size() == sizeof(Request)) {
      //   reinterpret_cast<Request*>(m.data())->ts = get_ms();
      // }
      int n;
      if ((n = ::send(fd, m.data(), m.size(), 0)) == -1) {
        perror("send error");
        assert(false);
      }
      ASSERT(n == sizeof(Request) || n == sizeof(Response)) << n;
      bytes_avai -= n + 28;
      // if ((i & 0xf) == 0) refill(bytes_avai, begin);
  }

  void run_sender() {
    for (unsigned int i = 0; ; i++) {
      Message m = send_q->dequeue();
      assert(this->connected);
      _send_msg(m);
    }
  }

  void send_ack(Message &m) {
    const auto rq = reinterpret_cast<Request*>(m.data());
    if (rq->type != Request::LS) return;
    Message am; am.resize(sizeof(Response));
    auto rp = reinterpret_cast<Response*>(am.data());
    rp->idx = rq->ls.idx; rp->sid = rq->ls.sid; rp->iter = rq->ls.iter + 1;
    rp->ts = rq->ts;
    // MYLOG(2) << "sending ack " << + rp->idx << " iter=" << rp->iter - 1 << " sid="  << rp->sid;
    // send(am);
    ::send(fd, am.data(), am.size(), 0);
  }

  long long get_ms() {
    using namespace std::chrono;
    return duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
  }
  void recv_ack(Message &m) {
    auto rp = reinterpret_cast<Response*>(m.data());
    auto li = lis.find(layer_names.find(rp->idx)->second)->second;
    if (li->ack[peer_rank][rp->sid] < rp->iter) {
      li->ack[peer_rank][rp->sid] = rp->iter;
      MYLOG(2) << "ack " + li->name << " iter=" << rp->iter - 1 << " sid="  << rp->sid;
    }
    if (autortt) {
      long long n_rtt = get_ms() - rp->ts;
      if ((rand() & 0xfff) == 0)
        MYLOG(3) << "old_accumulated_rtt=" << rtt << " n_rtt=" << n_rtt << " new_accumulated_rtt=" <<
        rtt * ALPHA + (1 - ALPHA) * n_rtt;
      rtt = rtt * ALPHA + (1 - ALPHA) * n_rtt;
    }
    // TODO: fence
  }

  void do_ack(Message &m) {
    if (m.size() == sizeof(Request)) send_ack(m);
    else {
      // info inf{nullptr};
      // inf.m = m;
      // layer_q.enqueue(inf);
      recv_ack(m);
    }
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
      if (m.size() == sizeof(Request)) {
        recv_q->enqueue(m); // TODO force copy
        if (rand() % 100 < 1) MYLOG(2) << "recv qsize=" << recv_q->q.size();
      }
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
