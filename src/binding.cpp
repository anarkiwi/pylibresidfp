#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "EnvelopeGenerator.h"
#include "SID.h"
#include "WaveformGenerator.h"
#include "residfp/residfp.h"

namespace py = pybind11;

namespace reSIDfp {

class Probe {
public:
  static const EnvelopeGenerator &env(residfp &r, int v) {
    return *r.sid.voice[v].envelope();
  }
  static const WaveformGenerator &wave(residfp &r, int v) {
    return *r.sid.voice[v].wave();
  }
  static void set_lfsr(EnvelopeGenerator &e, uint16_t lfsr) {
    e.lfsr = lfsr;
    e.resetLfsr = false;
  }
  static uint16_t lfsr(const EnvelopeGenerator &e) { return e.lfsr; }
  static uint16_t rate(const EnvelopeGenerator &e) { return e.rate; }
  static bool idle(const EnvelopeGenerator &e) {
    return !(e.state_pipeline | e.envelope_pipeline | e.exponential_pipeline |
             e.resetLfsr | e.new_exponential_counter_period);
  }
  static uint8_t level(const EnvelopeGenerator &e) {
    return e.envelope_counter;
  }
  static uint8_t state(const EnvelopeGenerator &e) {
    return static_cast<uint8_t>(e.state);
  }
  static uint8_t level(residfp &r, int v) { return level(env(r, v)); }
  static uint8_t state(residfp &r, int v) { return state(env(r, v)); }
  static py::dict voices(residfp &r) {
    py::dict out;
    const auto field = [&](const char *name, auto get) {
      py::array_t<int64_t> a(3);
      for (int v = 0; v < 3; ++v)
        a.mutable_at(v) = static_cast<int64_t>(get(v));
      out[name] = a;
    };
    field("envelope", [&](int v) { return env(r, v).envelope_counter; });
    field("state", [&](int v) { return static_cast<int>(env(r, v).state); });
    field("next_state",
          [&](int v) { return static_cast<int>(env(r, v).next_state); });
    field("gate", [&](int v) { return env(r, v).gate; });
    field("lfsr", [&](int v) { return env(r, v).lfsr; });
    field("rate", [&](int v) { return env(r, v).rate; });
    field("exponential_counter",
          [&](int v) { return env(r, v).exponential_counter; });
    field("exponential_counter_period",
          [&](int v) { return env(r, v).exponential_counter_period; });
    field("counter_enabled", [&](int v) { return env(r, v).counter_enabled; });
    field("state_pipeline", [&](int v) { return env(r, v).state_pipeline; });
    field("envelope_pipeline",
          [&](int v) { return env(r, v).envelope_pipeline; });
    field("exponential_pipeline",
          [&](int v) { return env(r, v).exponential_pipeline; });
    field("reset_lfsr", [&](int v) { return env(r, v).resetLfsr; });
    field("attack", [&](int v) { return env(r, v).attack; });
    field("decay", [&](int v) { return env(r, v).decay; });
    field("sustain", [&](int v) { return env(r, v).sustain; });
    field("release", [&](int v) { return env(r, v).release; });
    field("accumulator", [&](int v) { return wave(r, v).accumulator; });
    field("freq", [&](int v) { return wave(r, v).freq; });
    field("pw", [&](int v) { return wave(r, v).pw; });
    field("waveform", [&](int v) { return wave(r, v).waveform; });
    field("test", [&](int v) { return wave(r, v).test; });
    field("sync", [&](int v) { return wave(r, v).sync; });
    field("ring", [&](int v) { return wave(r, v).ring_msb_mask != 0; });
    return out;
  }
};

} // namespace reSIDfp

namespace {

constexpr int64_t kPeriod = 32767;

struct LfsrTable {
  std::array<uint16_t, kPeriod> state{};
  std::array<uint16_t, kPeriod + 1> pos{};
  LfsrTable() {
    reSIDfp::EnvelopeGenerator e;
    for (int64_t i = 0; i < kPeriod; ++i) {
      state[i] = reSIDfp::Probe::lfsr(e);
      pos[state[i]] = static_cast<uint16_t>(i);
      e.clock();
    }
  }
};

const LfsrTable &table() {
  static const LfsrTable t;
  return t;
}

int64_t wrap(int64_t x) { return ((x % kPeriod) + kPeriod) % kPeriod; }

struct Write {
  int64_t cycle;
  int reg;
  uint8_t value;
};

struct Event {
  int64_t base, slope;
  uint8_t level, state;
};

struct Piece {
  int64_t k0, k1;
  std::vector<Event> ev;
};

// Phases [k0, k1) share every envelope field but the LFSR. Unshifted: all are
// at cycle t and the LFSR of phase k is that of k0 advanced by k - k0.
// Shifted: all hold the same LFSR but phase k is at cycle t - (k - k0).
struct Group {
  int64_t k0, k1, t;
  bool shifted;
  size_t wi;
  reSIDfp::EnvelopeGenerator e;
  std::vector<Event> ev;
};

class PhaseSim {
public:
  PhaseSim(const reSIDfp::EnvelopeGenerator &e0, std::vector<Write> writes,
           int64_t start, int64_t until)
      : e0_(e0), wr_(std::move(writes)), start_(start), until_(until) {}

  std::vector<Piece> run() {
    const auto &tab = table();
    Group g{0, tab.pos[reSIDfp::Probe::rate(e0_)] + 1, start_, false, 0, e0_,
            {}};
    reSIDfp::Probe::set_lfsr(g.e, tab.state[0]);
    g.ev.push_back(
        {start_, 0, reSIDfp::Probe::level(g.e), reSIDfp::Probe::state(g.e)});
    stack_.push_back(std::move(g));
    while (!stack_.empty()) {
      Group cur = std::move(stack_.back());
      stack_.pop_back();
      drive(std::move(cur));
    }
    std::sort(done_.begin(), done_.end(),
              [](const Piece &a, const Piece &b) { return a.k0 < b.k0; });
    return merge();
  }

private:
  using EG = reSIDfp::EnvelopeGenerator;

  int64_t index(const EG &e) const {
    return table().pos[reSIDfp::Probe::lfsr(e)];
  }

  static void place(EG &e, int64_t idx) {
    reSIDfp::Probe::set_lfsr(e, table().state[wrap(idx)]);
  }

  Group child(const Group &g, int64_t a, int64_t b) const {
    Group c = g;
    c.k0 = g.k0 + a;
    c.k1 = g.k0 + b;
    if (g.shifted)
      c.t -= a;
    for (auto &e : c.ev)
      e.base += e.slope * a;
    return c;
  }

  void note(Group &g) const {
    const uint8_t l = reSIDfp::Probe::level(g.e);
    const uint8_t s = reSIDfp::Probe::state(g.e);
    if (l != g.ev.back().level || s != g.ev.back().state)
      g.ev.push_back({g.t, g.shifted ? -1 : 0, l, s});
  }

  void apply(Group &g) const {
    for (; g.wi < wr_.size() && wr_[g.wi].cycle == g.t; ++g.wi) {
      switch (wr_[g.wi].reg) {
      case 4:
        g.e.writeCONTROL_REG(wr_[g.wi].value);
        break;
      case 5:
        g.e.writeATTACK_DECAY(wr_[g.wi].value);
        break;
      default:
        g.e.writeSUSTAIN_RELEASE(wr_[g.wi].value);
        break;
      }
    }
  }

  void finish(Group &&g) { done_.push_back({g.k0, g.k1, std::move(g.ev)}); }

  void drive(Group g) {
    for (;;) {
      const int64_t w = g.wi < wr_.size() ? wr_[g.wi].cycle : until_;
      if (!g.shifted) {
        if (g.t == w) {
          if (w == until_)
            return finish(std::move(g));
          apply(g);
        } else if (reSIDfp::Probe::idle(g.e)) {
          return jump_unshifted(g, w);
        } else if (step_unshifted(g)) {
          return;
        }
      } else if (reSIDfp::Probe::idle(g.e) &&
                 reSIDfp::Probe::lfsr(g.e) != reSIDfp::Probe::rate(g.e)) {
        return jump_shifted(g, w);
      } else if (g.t == w) {
        Group first = child(g, 0, 1);
        first.shifted = false;
        g = child(g, 1, g.k1 - g.k0);
        if (w == until_) {
          finish(std::move(first));
        } else {
          apply(first);
          stack_.push_back(std::move(first));
        }
        if (g.k0 == g.k1)
          return;
      } else {
        g.e.clock();
        ++g.t;
        note(g);
      }
    }
  }

  bool step_unshifted(Group &g) {
    const int64_t n = g.k1 - g.k0;
    if (n == 1) {
      g.e.clock();
      ++g.t;
      note(g);
      return false;
    }
    const EG before = g.e;
    g.e.clock();
    ++g.t;
    note(g);
    const int64_t i0 = index(before);
    const int64_t hit = wrap(table().pos[reSIDfp::Probe::rate(g.e)] - i0);
    if (hit >= n)
      return false;
    const auto spawn = [&](int64_t a, int64_t b) {
      Group c = child(g, a, b);
      c.e = before;
      place(c.e, i0 + a);
      c.e.clock();
      stack_.push_back(std::move(c));
    };
    if (hit > 0)
      spawn(0, hit);
    spawn(hit, hit + 1);
    stack_.back().shifted = true;
    if (hit + 1 < n)
      spawn(hit + 1, n);
    return true;
  }

  void jump_unshifted(Group &g, int64_t w) {
    const auto &tab = table();
    const int64_t n = g.k1 - g.k0, h = w - g.t, i0 = index(g.e);
    const int64_t rate = tab.pos[reSIDfp::Probe::rate(g.e)];
    const int64_t u = wrap(rate - i0);
    const int64_t cuts[3] = {0, std::min(n, u + 1), n};
    for (int s = 0; s < 2; ++s) {
      const int64_t a = cuts[s], b = cuts[s + 1];
      if (a >= b)
        continue;
      const int64_t span = (s ? u + kPeriod : u) - a;
      const int64_t c = std::min(b, std::max(a, a + span - h + 1));
      if (c > a) {
        Group late = child(g, a, c);
        place(late.e, i0 + a + h);
        late.t = w;
        stack_.push_back(std::move(late));
      }
      if (c < b) {
        Group early = child(g, c, b);
        place(early.e, rate);
        early.t = g.t + span - (c - a);
        early.shifted = true;
        stack_.push_back(std::move(early));
      }
    }
  }

  void jump_shifted(Group &g, int64_t w) {
    const auto &tab = table();
    const int64_t n = g.k1 - g.k0, i0 = index(g.e);
    const int64_t m = wrap(tab.pos[reSIDfp::Probe::rate(g.e)] - i0);
    const int64_t lead = w - g.t;
    const int64_t na = std::min(n, std::max<int64_t>(0, m - lead + 1));
    if (na > 0) {
      Group due = child(g, 0, na);
      place(due.e, i0 + lead);
      due.t = w;
      due.shifted = false;
      stack_.push_back(std::move(due));
    }
    if (na < n) {
      Group rest = child(g, na, n);
      place(rest.e, i0 + m);
      rest.t += m;
      stack_.push_back(std::move(rest));
    }
  }

  static bool join(Piece &a, const Piece &b) {
    if (a.ev.size() != b.ev.size())
      return false;
    const int64_t la = a.k1 - a.k0, lb = b.k1 - b.k0;
    std::vector<int64_t> slope(a.ev.size());
    for (size_t i = 0; i < a.ev.size(); ++i) {
      const Event &x = a.ev[i], &y = b.ev[i];
      if (x.level != y.level || x.state != y.state ||
          (la > 1 && lb > 1 && x.slope != y.slope))
        return false;
      const int64_t cand[2] = {lb > 1 || la == 1 ? y.slope : x.slope, x.slope};
      const int tries = la == 1 && lb == 1 ? 2 : 1;
      int c = 0;
      while (c < tries && x.base + cand[c] * la != y.base)
        ++c;
      if (c == tries)
        return false;
      slope[i] = cand[c];
    }
    for (size_t i = 0; i < slope.size(); ++i)
      a.ev[i].slope = slope[i];
    a.k1 = b.k1;
    return true;
  }

  std::vector<Piece> merge() {
    std::vector<Piece> out;
    for (auto &p : done_) {
      if (out.empty() || !join(out.back(), p))
        out.push_back(std::move(p));
    }
    return out;
  }

  EG e0_;
  std::vector<Write> wr_;
  int64_t start_, until_;
  std::vector<Group> stack_;
  std::vector<Piece> done_;
};

} // namespace

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

  py::dict voices() {
    Claim claim(busy_);
    return reSIDfp::Probe::voices(sid_);
  }

  py::tuple trace(const writes_t &writes, int64_t until) {
    Claim claim(busy_);
    const auto w = checked(writes, until);
    std::vector<int64_t> cycles;
    std::vector<uint8_t> voice, level, state;
    {
      py::gil_scoped_release release;
      uint8_t last[3][2];
      const auto note = [&](bool always) {
        for (int v = 0; v < 3; ++v) {
          const uint8_t l = reSIDfp::Probe::level(sid_, v);
          const uint8_t s = reSIDfp::Probe::state(sid_, v);
          if (always || l != last[v][0] || s != last[v][1]) {
            cycles.push_back(cycle_);
            voice.push_back(static_cast<uint8_t>(v));
            level.push_back(l);
            state.push_back(s);
            last[v][0] = l;
            last[v][1] = s;
          }
        }
      };
      const auto step_to = [&](int64_t target) {
        while (cycle_ < target) {
          sid_.clockDigital(1);
          ++cycle_;
          note(false);
        }
      };
      note(true);
      for (size_t i = 0; i < w.size(); i += 3) {
        step_to(w[i]);
        sid_.write(static_cast<int>(w[i + 1]),
                   static_cast<unsigned char>(w[i + 2]));
      }
      step_to(until);
    }
    const auto array = [](const auto &v) {
      using T = typename std::decay_t<decltype(v)>::value_type;
      py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
      std::copy(v.begin(), v.end(), a.mutable_data());
      return a;
    };
    return py::make_tuple(array(cycles), array(voice), array(level),
                          array(state));
  }

  py::tuple envelope_trace(int voice, const writes_t &writes, int64_t until,
                           int lfsr) {
    Claim claim(busy_);
    if (voice < 0 || voice > 2)
      throw std::invalid_argument("voice out of range");
    if (lfsr > 0x7fff || lfsr == 0)
      throw std::invalid_argument("lfsr must be a non-zero 15-bit state");
    const auto w = checked(writes, until);
    std::vector<int64_t> cycles;
    std::vector<uint8_t> level, state;
    {
      py::gil_scoped_release release;
      reSIDfp::EnvelopeGenerator env = reSIDfp::Probe::env(sid_, voice);
      if (lfsr >= 0)
        reSIDfp::Probe::set_lfsr(env, static_cast<uint16_t>(lfsr));
      const auto note = [&](int64_t at) {
        const uint8_t l = reSIDfp::Probe::level(env);
        const uint8_t s = reSIDfp::Probe::state(env);
        if (cycles.empty() || l != level.back() || s != state.back()) {
          cycles.push_back(at);
          level.push_back(l);
          state.push_back(s);
        }
      };
      int64_t now = cycle_;
      note(now);
      size_t i = 0;
      while (now < until) {
        for (; i < w.size() && w[i] == now; i += 3) {
          const auto value = static_cast<uint8_t>(w[i + 2]);
          switch (w[i + 1] - 7 * voice) {
          case 4:
            env.writeCONTROL_REG(value);
            break;
          case 5:
            env.writeATTACK_DECAY(value);
            break;
          case 6:
            env.writeSUSTAIN_RELEASE(value);
            break;
          default:
            break;
          }
        }
        env.clock();
        note(++now);
      }
    }
    const auto array = [](const auto &v) {
      using T = typename std::decay_t<decltype(v)>::value_type;
      py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
      std::copy(v.begin(), v.end(), a.mutable_data());
      return a;
    };
    return py::make_tuple(array(cycles), array(level), array(state));
  }

  py::tuple envelope_phases(int voice, const writes_t &writes, int64_t until) {
    Claim claim(busy_);
    if (voice < 0 || voice > 2)
      throw std::invalid_argument("voice out of range");
    std::vector<Write> mine;
    const auto w = checked(writes, until);
    for (size_t i = 0; i < w.size(); i += 3) {
      const int reg = static_cast<int>(w[i + 1]) - 7 * voice;
      if (w[i] < until && reg >= 4 && reg <= 6)
        mine.push_back({w[i], reg, static_cast<uint8_t>(w[i + 2])});
    }
    std::vector<Piece> pieces;
    {
      py::gil_scoped_release release;
      pieces = PhaseSim(reSIDfp::Probe::env(sid_, voice), std::move(mine),
                        cycle_, until)
                   .run();
    }
    const auto &tab = table();
    const int64_t count = pieces.back().k1;
    size_t total = 0;
    for (const auto &p : pieces)
      total += p.ev.size();
    py::array_t<int64_t> pa(
        {static_cast<py::ssize_t>(pieces.size()), static_cast<py::ssize_t>(4)});
    py::array_t<int64_t> ea(
        {static_cast<py::ssize_t>(total), static_cast<py::ssize_t>(4)});
    py::array_t<uint16_t> states(static_cast<py::ssize_t>(count));
    std::copy(tab.state.begin(), tab.state.begin() + count,
              states.mutable_data());
    int64_t *prow = pa.mutable_data(), *erow = ea.mutable_data();
    int64_t first = 0;
    for (const auto &p : pieces) {
      const int64_t row[4] = {p.k0, p.k1, first,
                              static_cast<int64_t>(p.ev.size())};
      std::copy(row, row + 4, prow);
      prow += 4;
      first += static_cast<int64_t>(p.ev.size());
      for (const auto &e : p.ev) {
        const int64_t r[4] = {e.base, e.slope, e.level, e.state};
        std::copy(r, r + 4, erow);
        erow += 4;
      }
    }
    return py::make_tuple(pa, ea, count, states);
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
      .def("voices", &Chip::voices)
      .def("trace", &Chip::trace)
      .def("envelope_trace", &Chip::envelope_trace)
      .def("envelope_phases", &Chip::envelope_phases)
      .def("save_state", &Chip::save_state)
      .def("restore_state", &Chip::restore_state);
}
