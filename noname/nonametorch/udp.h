#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>   
#include <unistd.h>   
#include <errno.h>   
#include <arpa/inet.h>

using Message=std::string;
const int DATA_SIZE = 512;
struct Request {
  unsigned int type, seq;
  union {
    char msg[100];
    struct LayerSlice ls;
  };
};

struct LayerSlice {
  int idx, priority, rank, st;
  float data[DATA_SIZE/4];
};

struct Response {
  int ack_seq, idx, st;
};
static_assert(sizeof(Request) != sizeof(Response));

class UDPSocket {
private:
  ThreadSafeQueue<Message> *recv_q, *send_q;
  bool del{false};
  int fd;
public:
  bool connected{false};
  UDPSocket(ThreadSafeQueue<Message> *recv_q=nullptr): recv_q(recv_q) {
    if (recv_q == nullptr) {
      recv_q = new ThreadSafeQueue<Message>;
      send_q = new ThreadSafeQueue<Message>;
      del = true;
    }
    fd = socket(AF_INET, SOCK_DGRAM, 0);
  }
  ~UDPSocket() { 
    if (del) delete recv_q; 
    delete send_q;
  }

  // IMPORTANT: when you send, you must have connected beforehand
  int send(Message &m) {
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
    ::bind(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));

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
    struct sockaddr_in servaddr;
    bzero(&servaddr, sizeof(servaddr));
    servaddr.sin_addr.s_addr = htonl(addr);
    servaddr.sin_port = htons(port);
    servaddr.sin_family = AF_INET;
    ::connect(fd, (struct sockaddr*)&servaddr, sizeof(servaddr));

    char str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(servaddr.sin_addr), str, INET_ADDRSTRLEN);
    std::cout << "connecting " << str << ":" << port << "\n";
  }

  void connect(std::string s) {
    if (connected) return;
    connected = true;

    int idx = s.find(":"), addr_int;
    auto addr = s.substr(0, idx), port = s.substr(idx + 1, s.size() - idx - 1);
    inet_pton(AF_INET, addr.c_str(), &addr_int);
    connect(ntohl(addr_int), std::stoi(port));
  }

  void run_sender() {
    for (;;) {
      Message m = send_q->dequeue();
      assert(this->connected);
      ::send(fd, m.data(), m.size(), 0); 
    }
  }

  void run_recver() {
    struct sockaddr_in clent_addr;
    int buffer_len = DATA_SIZE*2, n;
    socklen_t len = sizeof(clent_addr);;
    auto buffer = new char[buffer_len];
    n = recvfrom(fd, buffer, buffer_len, 0, (struct sockaddr*)&clent_addr, &len);
    recv_q->enqueue(Message(buffer, buffer+n)); // TODO force copy
    connect(clent_addr.sin_addr.s_addr, clent_addr.sin_port);
    for (;;) {
      n = ::recv(fd, buffer, buffer_len, 0);
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
