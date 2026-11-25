// A CUDA 12.9 capture bridge for one conditional node with two bodies.
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace {
namespace alloc = c10::cuda::CUDACachingAllocator;
using CaptureId = unsigned long long;

void check_allocator() {
  TORCH_CHECK(alloc::name() == "native", "IF/ELSE capture requires the native PyTorch CUDA allocator");
  TORCH_CHECK(!alloc::CUDAAllocatorConfig::graph_capture_record_stream_reuse(),
              "Conditional capture requires graph_capture_record_stream_reuse=False");
}

struct CaptureInfo {
  cudaGraph_t graph = nullptr;
  CaptureId id = 0;
  const cudaGraphNode_t* dependencies = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  size_t count = 0;
};

CaptureInfo current_capture() {
  cudaStreamCaptureStatus status;
  CaptureInfo info;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v3(
      c10::cuda::getCurrentCUDAStream(), &status, &info.id, &info.graph,
      &info.dependencies, &info.edges, &info.count));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive, "A parent CUDA capture is required");
  return info;
}

CaptureId capture_id(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  CaptureId id = 0;
  if (cudaStreamGetCaptureInfo(stream, &status, &id) == cudaSuccess &&
      status == cudaStreamCaptureStatusActive) {
    return id;
  }
  return 0;
}

class CaptureIds {
 public:
  explicit CaptureIds(CaptureId root) : ids_{root} {}

  bool contains(CaptureId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
  }

  void push(CaptureId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    ids_.push_back(id);
  }

  void pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    ids_.pop_back();
  }

 private:
  mutable std::mutex mutex_;
  std::vector<CaptureId> ids_;
};

struct ChildStream {
  cudaStream_t value = nullptr;
  ChildStream() { C10_CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~ChildStream() { C10_CUDA_CHECK_WARN(cudaStreamDestroy(value)); }
  ChildStream(const ChildStream&) = delete;
  ChildStream& operator=(const ChildStream&) = delete;
};

class Capture {
 public:
  explicit Capture(c10::MempoolId_t pool)
      : pool_(pool), device_(c10::cuda::getCurrentCUDAStream().device_index()),
        root_(current_capture().id), active_(std::make_shared<CaptureIds>(root_)) {
    TORCH_CHECK(pool.first || pool.second, "Pass the pool of the active Torch graph");
    check_allocator();
    replace_filter([ids = active_](cudaStream_t stream) {
      return ids->contains(capture_id(stream));
    });
  }

  void close() {
    if (!closed_) {
      TORCH_CHECK(current_capture().id == root_, "Close the bridge on the root capture stream");
      replace_filter([root = root_](cudaStream_t stream) { return capture_id(stream) == root; });
      closed_ = true;
    }
  }

  cudaGraphConditionalHandle new_handle() const {
    const auto info = current_capture();
    TORCH_CHECK(!closed_ && active_->contains(info.id), "A bridge capture is required");
    cudaGraphConditionalHandle handle;
    C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, info.graph, 0, 0));
    return handle;
  }

  void set_handle(cudaGraphConditionalHandle handle, const at::Tensor& predicate) const {
    at::cuda::CUDAGraph::set_conditional_handle(handle, predicate);
  }

  void capture_if_else(cudaGraphConditionalHandle handle, const pybind11::function& if_body,
                       const pybind11::function& else_body) const {
    const auto info = current_capture();
    TORCH_CHECK(!closed_ && active_->contains(info.id), "A bridge capture is required");
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 2;
    cudaGraphNode_t node;
    C10_CUDA_CHECK(cudaGraphAddNode_v2(&node, info.graph, info.dependencies, info.edges, info.count, &params));
    C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies_v2(
        c10::cuda::getCurrentCUDAStream(), &node, nullptr, 1, cudaStreamSetCaptureDependencies));
    capture_body(params.conditional.phGraph_out[0], if_body);
    capture_body(params.conditional.phGraph_out[1], else_body);
  }

 private:
  void replace_filter(std::function<bool(cudaStream_t)> filter) {
    auto* host = at::getHostAllocator(at::kCUDA);
    alloc::endAllocateToPool(device_, pool_);
    host->end_allocate_to_pool(pool_);
    host->begin_allocate_to_pool(pool_, [filter](c10::Stream stream) {
      return filter(c10::cuda::CUDAStream(c10::cuda::CUDAStream::UNCHECKED, stream));
    });
    alloc::beginAllocateToPool(device_, pool_, std::move(filter));
    // The root Torch graph still owns its original pool references.
    host->release_pool(pool_);
    alloc::releasePool(device_, pool_);
  }

  void capture_body(cudaGraph_t graph, const pybind11::function& operation) const {
    ChildStream child;
    C10_CUDA_CHECK(cudaStreamBeginCaptureToGraph(
        child.value, graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    alloc::markCaptureBegin(device_);
    const auto id = capture_id(child.value);
    active_->push(id);
    const auto end_capture = [&] {
      const auto result = cudaStreamEndCapture(child.value, nullptr);
      active_->pop();
      alloc::markCaptureEnd(device_);
      return result;
    };
    try {
      TORCH_CHECK(id != 0, "Child CUDA capture did not start");
      c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(child.value, device_));
      operation();
    } catch (...) {
      C10_CUDA_CHECK_WARN(end_capture());
      throw;
    }
    C10_CUDA_CHECK(end_capture());
  }

  c10::MempoolId_t pool_;
  c10::DeviceIndex device_;
  CaptureId root_;
  std::shared_ptr<CaptureIds> active_;
  bool closed_ = false;
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("check_allocator", &check_allocator);
  pybind11::class_<Capture>(module, "Capture")
      .def(pybind11::init<c10::MempoolId_t>())
      .def("new_handle", &Capture::new_handle)
      .def("set_handle", &Capture::set_handle)
      .def("capture_if_else", &Capture::capture_if_else)
      .def("close", &Capture::close);
}
