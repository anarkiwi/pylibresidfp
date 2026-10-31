#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "residfp/residfp.h"

namespace py = pybind11;
using samples_t = py::array_t<int16_t>;
using writes_t =
    py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr size_t kScratch = 1 << 16;
constexpr size_t kStateHeader = sizeof(int64_t);

class Chip {
public:
  Chip(int model, double clock_hz, double sampling_hz, int method, bool filter)
      : per_cycle_(sampling_hz / clock_hz) {
    if (!sid_.setChipModel(model ? reSIDfp::CSG8580 : reSIDfp::MOS6581))
      throw std::invalid_argument("unsupported chip model");
    sid_.enableFilter(filter);
    const auto m = method ? reSIDfp::RESAMPLE : reSIDfp::DECIMATE;
    if (!sid_.setSamplingParameters(clock_hz, m, sampling_hz))
      throw std::invalid_argument("unsupported sampling parameters");
    max_step_ = std::max<int64_t>(
        1, static_cast<int64_t>(kScratch / (2.0 * per_cycle_ + 2.0)));
  }

  int64_t cycle() const { return cycle_; }

  void write(int reg, int value) {
    Claim claim(busy_);
    sid_.write(reg & 0x1f, static_cast<unsigned char>(value));
  }

  int read(int reg) {
    Claim claim(busy_);
    return sid_.read(reg & 0x1f);
  }

  int peek(int reg) {
    Claim claim(busy_);
    return sid_.peek(reg & 0x1f);
  }

  void reset() {
    Claim claim(busy_);
    sid_.reset();
  }

  samples_t clock(int64_t cycles) {
    Claim claim(busy_);
    if (cycles < 0)
      throw std::invalid_argument("cycles must be non-negative");
    return render({}, cycle_ + cycles);
  }

  samples_t run(const writes_t &writes, int64_t until) {
    Claim claim(busy_);
    return render(checked(writes, until), until);
  }

  int serve(const writes_t &writes, int64_t cycle, int reg) {
    Claim claim(busy_);
    render(checked(writes, cycle), cycle);
    return sid_.read(reg & 0x1f);
  }

  py::bytes save_state() {
    Claim claim(busy_);
    std::string buf(kStateHeader + sid_.stateSize(), '\0');
    std::memcpy(buf.data(), &cycle_, kStateHeader);
    int n = 0;
    {
      py::gil_scoped_release release;
      n = sid_.saveState(buf.data() + kStateHeader,
                         static_cast<int>(buf.size() - kStateHeader));
    }
    buf.resize(kStateHeader + n);
    return py::bytes(buf);
  }

  void restore_state(const std::string &state) {
    Claim claim(busy_);
    if (state.size() != kStateHeader + sid_.stateSize())
      throw std::invalid_argument("state size mismatch");
    std::string body(state, kStateHeader);
    {
      py::gil_scoped_release release;
      sid_.restoreState(body.data(), static_cast<int>(body.size()));
    }
    std::memcpy(&cycle_, state.data(), kStateHeader);
  }

private:
  struct Claim {
    explicit Claim(std::atomic<bool> &flag) : flag_(flag) {
      if (flag_.exchange(true))
        throw std::runtime_error("the SID is in use by another thread");
    }
    ~Claim() { flag_.store(false); }
    Claim(const Claim &) = delete;
    Claim &operator=(const Claim &) = delete;
    std::atomic<bool> &flag_;
  };

  std::vector<int64_t> checked(const writes_t &writes, int64_t until) const {
    if (writes.ndim() != 2 || writes.shape(1) != 3)
      throw std::invalid_argument("writes must have shape (N, 3)");
    if (until < cycle_)
      throw std::invalid_argument("until is before the current cycle");
    std::vector<int64_t> w(writes.data(), writes.data() + writes.size());
    int64_t prev = cycle_;
    for (size_t i = 0; i < w.size(); i += 3) {
      if (w[i] < prev)
        throw std::invalid_argument(
            "write cycles must not precede the current cycle or each other");
      if (w[i] > until)
        throw std::invalid_argument("write cycle is after until");
      if (w[i + 1] < 0 || w[i + 1] > 0x1f)
        throw std::invalid_argument("register out of range");
      if (w[i + 2] < 0 || w[i + 2] > 0xff)
        throw std::invalid_argument("value out of range");
      prev = w[i];
    }
    return w;
  }

  samples_t render(const std::vector<int64_t> &w, int64_t until) {
    const size_t count = w.size() / 3;
    std::vector<int16_t> out;
    out.reserve(static_cast<size_t>((until - cycle_) * per_cycle_) + count + 1);
    {
      py::gil_scoped_release release;
      for (size_t i = 0; i < count; ++i) {
        advance(w[3 * i], out);
        sid_.write(static_cast<int>(w[3 * i + 1]),
                   static_cast<unsigned char>(w[3 * i + 2]));
      }
      advance(until, out);
    }
    samples_t arr(static_cast<py::ssize_t>(out.size()));
    std::copy(out.begin(), out.end(), arr.mutable_data());
    return arr;
  }

  void advance(int64_t target, std::vector<int16_t> &out) {
    while (cycle_ < target) {
      const auto step =
          static_cast<unsigned>(std::min<int64_t>(target - cycle_, max_step_));
      const int n = sid_.clock(step, scratch_.data());
      out.insert(out.end(), scratch_.begin(), scratch_.begin() + n);
      cycle_ += step;
    }
  }

  reSIDfp::residfp sid_;
  double per_cycle_;
  int64_t max_step_ = 1;
  std::vector<int16_t> scratch_ = std::vector<int16_t>(kScratch);
  int64_t cycle_ = 0;
  std::atomic<bool> busy_{false};
};

} // namespace

PYBIND11_MODULE(_core, m) {
  py::class_<Chip>(m, "Chip")
      .def(py::init<int, double, double, int, bool>())
      .def_property_readonly("cycle", &Chip::cycle)
      .def("write", &Chip::write)
      .def("read", &Chip::read)
      .def("peek", &Chip::peek)
      .def("reset", &Chip::reset)
      .def("clock", &Chip::clock)
      .def("run", &Chip::run)
      .def("serve", &Chip::serve)
      .def("save_state", &Chip::save_state)
      .def("restore_state", &Chip::restore_state);
}
