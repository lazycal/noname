#pragma once

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>   
#include <unistd.h>   
#include <errno.h>   
#include <arpa/inet.h>
#include "utils.h"
#include "dbug_logging.h"

const int MAX_BYTES_AVAI = 10000;
const int SLICE_SIZE = 512;
static_assert(SLICE_SIZE % 4 == 0, "SLICE_SIZE % 4 !=0");
const int SR = 10000; // Bytes per milisecond

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
  int ack_seq, idx, sid;
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
  ThreadSafeQueue<Message> *recv_q, *send_q;
  bool del{false};
  int fd;
  std::thread *_t;
public:
  bool connected{false};
  UDPSocket(ThreadSafeQueue<Message> *_recv_q=nullptr): recv_q(_recv_q) {
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
    bytes_avai = std::min(MAX_BYTES_AVAI, SR * el);
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
      ASSERT(n == sizeof(Request)) << n;
      bytes_avai -= n;
      // if ((i & 0xf) == 0) refill(bytes_avai, begin);
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
    recv_q->enqueue(Message(buffer, buffer+n)); // TODO force copy
    for (;;) {
      n = ::recv(fd, buffer, buffer_len, 0);
      if (n == -1) { perror("recv error"); assert(false); }
      ASSERT(n == sizeof(Request) || n == sizeof(Response)) << n;
      recv_q->enqueue(Message(buffer, buffer+n)); // TODO force copy
    }
  }

  void run() {
    auto t1 = std::thread([=]{this->run_sender();});
    auto t2 = std::thread([=]{this->run_recver();});
    t1.join();
    t2.join();
  }
};
