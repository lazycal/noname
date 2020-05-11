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

void _malloc(void **ptr, size_t size) { cudaMallocHost(ptr, size); }
void _free(void *ptr) { cudaFreeHost(ptr); }
#else
void _malloc(void **ptr, size_t size) { *ptr = malloc(size); }
void _free(void *ptr) { free(ptr); }
#endif

const std::string HAND_SHAKE_MSG = "noname__start!";
const std::string SHUTDOWN_MSG = "#SHUTDOWN#";

int cur_iter = 0; // TODO: extern
struct LayerInfo {
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::string name;
  size_t size;
  torch::Tensor tensor;
  int priority, iter;
  int cnt; // used in PS
  void *buf;
  int w_recv_cnt; // TODO: move the fields for PS to a new class.
  std::vector<std::vector<uint8_t>> worker_bufs; // TODO: flattened 2d array

  LayerInfo(const std::string &name, size_t size, torch::Tensor tensor,
            int priority, int nw = 0)
      : name(name), size(size), tensor(tensor), priority(priority),
        iter(cur_iter), w_recv_cnt(0), worker_bufs(nw, std::vector<uint8_t>(size, 0)) {
    if (tensor.device().is_cuda())
      _malloc(&buf, size);
    else
      buf = tensor.data_ptr<float>();
  }
  ~LayerInfo() {
    if (tensor.device().is_cuda())
      _free(&buf);
  }
};
std::ostream &operator<<(std::ostream &os, const LayerInfo &li) {
  os << "LayerInfo(name=" << li.name << ", size=" << li.size
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
const std::map<std::string, LayerInfo *> &lis = _lis;

void declare(std::string name) { _lis[name] = nullptr; }
LayerInfo *get_or_register_layer(const std::string &name, size_t size,
                                 torch::Tensor tensor, int priority, int nw = 0) {
  ASSERT(_lis.find(name) != _lis.end()) << name;
  auto &res = _lis[name];
  if (res == nullptr) {
    res = new LayerInfo(name, size, tensor, priority, nw); // TODO: barrier?
    std::cout << "registering " << *res << "\n";
  }
  return res;
}

int config_get_size() {
  auto p = getenv("DMLC_NUM_WORKER");
  return (p == nullptr) ? 1 : atoi(p);
}

std::string config_get_PS_ip_port(int delta = 0) {
  auto p1 = getenv("DMLC_PS_ROOT_URI");
  assert(p1);
  auto p2 = getenv("DMLC_PS_ROOT_PORT");
  assert(p2);
  std::string port_s = p2;
  if (delta) {
    int port = atoi(p2) + delta;
    port_s = std::to_string(port);
  }
  return p1 + std::string(":") + port_s;
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

std::string config_get_worker_ip() {
  auto p = getenv("WORKER_IP");
  assert(p);
  return p;
}