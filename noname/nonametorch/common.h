#pragma once

#include <TH/TH.h>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <torch/extension.h>
#include "dbug_logging.h"

#ifdef HAVE_CUDA
#include "helper_cuda.h"
#include <cuda_runtime.h>

void _malloc_cuda(void **ptr, size_t size) { cudaMallocHost(ptr, size); }
void _free_cuda(void *ptr) { cudaFreeHost(ptr); }
#endif
void _malloc(void **ptr, size_t size) { *ptr = malloc(size); }
void _free(void *ptr) { free(ptr); }

const std::string HAND_SHAKE_MSG = "noname__start!";
const std::string SHUTDOWN_MSG = "#SHUTDOWN#";

const int SLICE_SIZE = 512;
static_assert(SLICE_SIZE % 4 == 0, "SLICE_SIZE % 4 !=0");
// #ifndef SEND_RATE
// #define SEND_RATE 10000 // Bytes per milisecond
// #endif
// #ifndef LOSS_RATE
// #define LOSS_RATE 90 // /100
// #endif
int THRES = 90;
int get_PP_METHOD() {
  auto p = getenv("PP_METHOD");
  int ppm = 1; // 1 for udp; 0 for tcp;
  if (p != nullptr) ppm = std::atoi(p);
  return ppm;
}

inline int CEIL(int x, int y) { return (x + y - 1) / y; }
bool layer_enough(int slc_cnt, size_t size, int SLICE_SIZE) {
  int n_slc = CEIL(size, SLICE_SIZE);
  return slc_cnt * 100 >= n_slc * THRES && slc_cnt >= 1;
}

int cur_iter = 0; // TODO: extern
struct LayerInfo {
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::string name;
  size_t size;
  torch::Tensor tensor;
  int priority, idx, iter;
  int cnt; // used in PS
  uint8_t *buf;
  // for ps below
  int w_recv_cnt{0}, acc_iter{0}; // TODO: move the fields for PS to a new class.
  bool dirty{true};
  std::vector<std::vector<uint8_t>> worker_bufs; // TODO: flattened 2d array
  std::vector<std::vector<int>> w_slc_it;
  std::vector<int> w_slc_recv_cnt, w_iter;
  std::vector<std::vector<int>> ack;
  uint8_t *ps_buf[2];

  LayerInfo(const std::string &name, size_t size, torch::Tensor tensor,
            int priority, int idx, int nw)
      : name(name), size(size), tensor(tensor), priority(priority), idx(idx),
        iter(cur_iter), worker_bufs(nw, std::vector<uint8_t>(size, 0)),
        w_slc_it(nw, std::vector<int>(CEIL(size, SLICE_SIZE), 0)), w_slc_recv_cnt(nw, 0),
        w_iter(nw, 0), ack(nw, std::vector<int>(CEIL(size, SLICE_SIZE), 0))
         {
    ASSERT(size < 1ll << 31) << "size=" << size << " too long.";
    if (tensor.device().is_cuda()) {
      void *tmp;
      _malloc_cuda(&tmp, size);
      buf = reinterpret_cast<uint8_t*>(tmp);
    } else
      buf = reinterpret_cast<uint8_t*>(tensor.data_ptr<float>());
    ps_buf[0] = buf;
    void *tmp;
    _malloc(&tmp, size);
    ps_buf[1] = reinterpret_cast<uint8_t*>(tmp);
  }
  ~LayerInfo() {
    if (tensor.device().is_cuda())
      _free_cuda(reinterpret_cast<void*>(buf));
    _free(reinterpret_cast<void*>(ps_buf[1]));
  }
};
std::ostream &operator<<(std::ostream &os, const LayerInfo &li) {
  os << "LayerInfo(id=" << li.idx << ", name=" << li.name << ", size=" << li.size
     << ", priority=" << li.priority << ")";
  return os;
}
struct LayerInfoMini {
  size_t size;
  int priroity;
  int rank;
  static LayerInfoMini fromLayerInfo(const LayerInfo &li, int rank) {
    return {li.size, li.priority, rank};
  }
};

std::map<std::string, LayerInfo *>
    _lis; // TODO: move _lis inside a class to better hide it. // TODO: extern
static std::map<std::string, int> _layer_id; // HACK
static std::map<int, std::string> _layer_names;
static std::map<int, size_t> _layer_sizes;
const std::map<std::string, LayerInfo *> &lis = _lis;
std::map<int, std::string> &layer_names = _layer_names;
std::map<int, size_t> &layer_sizes = _layer_sizes;

void declare(std::string name, int idx, size_t size) { 
  _lis[name] = nullptr; 
  _layer_id[name] = idx;
  _layer_names[idx] = name;
  _layer_sizes[idx] = size;
}
LayerInfo *get_or_register_layer(const std::string &name, size_t size,
                                 torch::Tensor tensor, int priority, int nw = 1) {
  ASSERT(_lis.find(name) != _lis.end()) << name;
  auto &res = _lis[name];
  if (res == nullptr) {
    res = new LayerInfo(name, size, tensor, priority, _layer_id[name], nw); // TODO: barrier?
    std::cout << "registering " << *res << "\n";
  }
  return res;
}

int config_get_size() {
  auto p = getenv("DMLC_NUM_WORKER");
  return (p == nullptr) ? 1 : atoi(p);
}

std::string config_get_PS_ip_port(int delta = 0, bool is_ps = false) {
  auto p1 = getenv("DMLC_PS_ROOT_URI");
  assert(p1);
  auto p2 = getenv("DMLC_PS_ROOT_PORT");
  assert(p2);
  std::string port_s = p2;
  if (delta) {
    int port = atoi(p2) + delta;
    port_s = std::to_string(port);
  }
  return (is_ps ? "0.0.0.0" : p1) + std::string(":") + port_s;
}

std::string config_get_role() {
  auto p = getenv("DMLC_ROLE");
  assert(p);
  return p;
}

int config_get_rank() {
  auto p = getenv("DMLC_WORKER_ID");
  assert(p);
  return atoi(p);
}

int config_get_rank2() {
  if (config_get_role() == "server") return -1;
  else return config_get_rank();
}

std::string config_get_worker_ip() {
  auto p = getenv("WORKER_IP");
  assert(p);
  return p;
}

void config_init() {
  auto p = getenv("THRES");
  if (p) {
    THRES = std::atoi(p);
  }
  std::cout << "setting THRES to " << THRES << "\n";
}