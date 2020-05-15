#include "common.h"
#include "ps.h"
#include "ps_lossy.h"
#include "worker_lossy.h"
#include "utils.h"
#include "worker.h"
#include <TH/TH.h>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <torch/extension.h>

// #ifndef PP_METHOD
// #define PP_METHOD 1 // for udp; 0 for tcp
// #endif
// #if PP_METHOD == 0
// using PushPullMethod=VanillaPushPull;
// using PSMethod=VanillaPS;
// #elif PP_METHOD == 1
// using PushPullMethod=LossyPushPull;
// using PSMethod=LossyPS;
// #endif
class CommunicationManager {
public:
  const std::map<std::string, LayerInfo *> &lis;
  // ThreadSafeQueue<LayerInfo *> q;
  std::unique_ptr<PushPullProtocol> p3;
  CommunicationManager(const std::map<std::string, LayerInfo *> &lis)
      : lis(lis) {}
  void set_p3() {
    if (get_PP_METHOD() == 0) p3 = std::make_unique<VanillaPushPull>();
    else p3 = std::make_unique<LossyPushPull>();
  }
  void push_pull(std::string name) {
    auto it = lis.find(name);
    assert(it != lis.end());
    // q.enqueue(it->second);
    p3->push(it->second->name);
  }
  void run();
  void shutdown() {
    p3->ready_q.enqueue(nullptr);
    p3->shutdown();
  }
} cm(lis);

void CommunicationManager::run() {
  std::cout << "CommunicationManager running!\n";
  // receiver
  auto p3_t = std::thread([&] { p3->run(); });

  for (;;) {
    // update layerinfo
    auto li = p3->ready_q.dequeue();
    if (li == nullptr) break;
#ifdef HAVE_CUDA
    auto &tensor = li->tensor;
    if (tensor.device().is_cuda()) {
      checkCudaErrors(cudaMemcpy(tensor.data_ptr<float>(), li->buf, li->size,
                                 cudaMemcpyHostToDevice));
    }
#endif
    {
      std::lock_guard<std::mutex> lk(li->d_mutex); // only need to lock iter
      li->iter++;
      MYLOG(1) << "Layer " << li->name << " advances to iter " << li->iter
               << ", current iter=" << cur_iter;
    }
    li->d_condition.notify_one();
  }
  p3_t.join();
}

// struct MemPool {
//   std::map<std::string, void*> m;
//   void *add(const std::string &name, size_t size) {
//     void *a;
//     _malloc(&a, size);
//     m[name] = a;
//     return a;
//   }
//   void *get_or_add(const std::string &name, size_t size) {
//     auto& res = m[name];
//     if (res == nullptr) {
//       res = new char[size];
//       std::cout << "add(" << name << ", " << size << ")\n";
//     }
//     return res;
//   }
// } memPool;

inline size_t sizeof_tensor(torch::Tensor tensor) {
  return tensor.element_size() * tensor.numel();
}

std::string push_pull_async_inplace(torch::Tensor tensor, int average,
                                    const std::string &name, int iter,
                                    int priority) {
  AT_ASSERT(tensor.dtype() == torch::kFloat32,
            "Unsupported Dtype: ", tensor.dtype().name());
  // auto before = tensor.mean();
  // auto is_param = (name.find("Parameter") != std::string::npos);
  auto sz = sizeof_tensor(tensor);
  // get_or_register_layer(name, sz);
  // if (!is_param) {
    get_or_register_layer(name, sz, tensor,
                          priority); // TODO: register at the beginning
#ifdef HAVE_CUDA
    if (tensor.device().is_cuda()) {
      assert(lis.find(name) != lis.end());
      assert(lis.find(name)->second->tensor.is_same(tensor));
      checkCudaErrors(cudaMemcpy(lis.find(name)->second->buf,
                                 tensor.data_ptr<float>(), sz,
                                 cudaMemcpyDeviceToHost));
    }
#endif
    // push...
    // memset(ht, 0, sz);
    MYLOG(1) << "begin push_pull: " << name;
    cm.push_pull(name);
    MYLOG(1) << "end push_pull: " << name;

    // std::cout << "mean: before=" << before << ", after=" << tensor.mean() <<
    // "\n";
  // }
  // std::cout << output.sizes() << " " << average << " " << name << " " <<
  // version
  //           << std::endl;
  return name;
}

void synchronize(const std::string &name) {
  auto li = lis.find(name)->second;
  MYLOG(2) << "["
           << "cur_iter=" << cur_iter << "] synchronize: " << *li;
  std::unique_lock<std::mutex> lk(li->d_mutex);
  li->d_condition.wait(lk, [&] {
    return li->iter > cur_iter;
  });
}

void init() {}

void declare_done() {
  auto role = config_get_role();
  if (role == "server") {
    std::cout << "launching server...\n";
    PS* p;
    if (get_PP_METHOD() == 0) p = new VanillaPS();
    else p = new LossyPS();
    p->run();
  } else {
    std::cout << "launching worker...\n";
    AT_ASSERT(role == "worker", role);
    // ASSERT(role=="worker") << role;
    cm.set_p3();
    new std::thread([&] { cm.run(); });
  }
}

void advance_iter() {
  cur_iter++;
  MYLOG(1) << "advance_iter: cur_iter=" << cur_iter;
  for (auto it : lis) {
    assert(it.second->iter == cur_iter);
  }
}

void noname_shutdown() {
  std::cout << "shutting down...\n";
  cm.shutdown();
  std::cout << "shutdown done...\n";
}

PYBIND11_MODULE(c_lib, m) {
  m.def("push_pull_async_inplace", &push_pull_async_inplace, "");
  m.def("synchronize", &synchronize, "");
  m.def("size", &config_get_size, "");
  // m.def("local_size", &local_size, "");
  // m.def("rank", &rank, "");
  m.def("init", &init, "");
  m.def("step", &advance_iter, "");
  m.def("declare", &declare, "");
  m.def("declare_done", &declare_done, "");
  m.def("shutdown", &noname_shutdown, "");
}
