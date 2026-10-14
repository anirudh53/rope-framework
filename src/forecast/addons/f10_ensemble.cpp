#include "f10_ensemble.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace rope::forecast {

static constexpr std::uint32_t F10_MAGIC   = 0x52504631u;  // "RPF1"
static constexpr std::uint32_t F10_VERSION = 1u;

namespace {

struct Reader {
    std::ifstream f;
    std::filesystem::path path;
    template <typename T> T get() {
        T v{};
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!f) throw std::runtime_error("F10Ensemble::load: unexpected EOF in " + path.string());
        return v;
    }
    std::vector<float> floats(std::size_t n) {
        std::vector<float> v(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
        if (!f) throw std::runtime_error("F10Ensemble::load: unexpected EOF in " + path.string());
        return v;
    }
};

}  // namespace

F10Ensemble F10Ensemble::load(const std::filesystem::path& path) {
    Reader r{std::ifstream(path, std::ios::binary), path};
    if (!r.f) throw std::runtime_error("F10Ensemble::load: cannot open " + path.string());
    if (r.get<std::uint32_t>() != F10_MAGIC)
        throw std::runtime_error("F10Ensemble::load: bad magic in " + path.string() +
                                 " (not a .f10bin file)");
    const auto version = r.get<std::uint32_t>();
    if (version != F10_VERSION)
        throw std::runtime_error("F10Ensemble::load: unsupported version " +
                                 std::to_string(version) + " in " + path.string());

    F10Ensemble e;
    e.scale_mean_   = r.get<double>();
    e.scale_std_    = r.get<double>();
    e.max_lookback_ = static_cast<int>(r.get<std::uint32_t>());
    e.n_features_   = static_cast<int>(r.get<std::uint32_t>());
    e.avg_days_     = static_cast<int>(r.get<std::uint32_t>());
    const auto n_members = r.get<std::uint32_t>();
    if (e.n_features_ != 2 || e.max_lookback_ <= 0 || e.avg_days_ <= 0 || n_members == 0 ||
        !(e.scale_std_ > 0.0))
        throw std::runtime_error("F10Ensemble::load: invalid header in " + path.string());

    e.members_.resize(n_members);
    for (auto& m : e.members_) {
        m.lookback       = static_cast<int>(r.get<std::uint32_t>());
        m.sequence_input = r.get<std::uint32_t>() == 0;
        const auto n_layers = r.get<std::uint32_t>();
        if (m.lookback <= 1 || m.lookback > e.max_lookback_ || n_layers == 0)
            throw std::runtime_error("F10Ensemble::load: invalid member in " + path.string());
        m.layers.resize(n_layers);
        for (auto& L : m.layers) {
            const auto type = r.get<std::uint32_t>();
            L.act              = static_cast<Act>(r.get<std::uint32_t>());
            L.rec_act          = static_cast<Act>(r.get<std::uint32_t>());
            L.return_sequences = r.get<std::uint32_t>() != 0;
            L.in_dim           = static_cast<int>(r.get<std::uint32_t>());
            L.units            = static_cast<int>(r.get<std::uint32_t>());
            if (type > 1 || static_cast<int>(L.act) > 3 || static_cast<int>(L.rec_act) > 3 ||
                L.in_dim <= 0 || L.units <= 0)
                throw std::runtime_error("F10Ensemble::load: invalid layer in " + path.string());
            L.lstm = type == 0;
            const std::size_t u = static_cast<std::size_t>(L.units), in = static_cast<std::size_t>(L.in_dim);
            if (L.lstm) {
                L.kernel    = r.floats(in * 4 * u);
                L.recurrent = r.floats(u * 4 * u);
                L.bias      = r.floats(4 * u);
            } else {
                L.kernel = r.floats(in * u);
                L.bias   = r.floats(u);
            }
        }
    }
    return e;
}

double F10Ensemble::apply(Act a, double v) {
    switch (a) {
        case Act::Linear:  return v;
        case Act::Tanh:    return std::tanh(v);
        case Act::Sigmoid: return 1.0 / (1.0 + std::exp(-v));
        case Act::Relu:    return v > 0.0 ? v : 0.0;
    }
    return v;
}

// One member, one step: x is the normalized window (2*lookback values). Returns the next value.
double F10Ensemble::predict(const Member& m, const std::vector<double>& x) const {
    // Sequence of feature vectors: LSTM input is (T, 1); an MLP sees one flat vector.
    std::vector<std::vector<double>> seq;
    if (m.sequence_input) {
        seq.reserve(x.size());
        for (double v : x) seq.push_back({v});
    } else {
        seq.push_back(x);
    }

    for (const Layer& L : m.layers) {
        const int u = L.units;
        if (L.lstm) {
            std::vector<double> h(u, 0.0), c(u, 0.0), z(4 * u);
            std::vector<std::vector<double>> out;
            for (const auto& xt : seq) {
                if (static_cast<int>(xt.size()) != L.in_dim)
                    throw std::runtime_error("F10Ensemble: LSTM input size mismatch");
                for (int j = 0; j < 4 * u; ++j) {
                    double s = L.bias[j];
                    for (int i = 0; i < L.in_dim; ++i) s += xt[i] * L.kernel[i * 4 * u + j];
                    for (int i = 0; i < u; ++i)        s += h[i]  * L.recurrent[i * 4 * u + j];
                    z[j] = s;
                }
                // Keras gate order: input, forget, cell, output
                for (int k = 0; k < u; ++k) {
                    const double ig = apply(L.rec_act, z[k]);
                    const double fg = apply(L.rec_act, z[u + k]);
                    const double cg = apply(L.act,     z[2 * u + k]);
                    const double og = apply(L.rec_act, z[3 * u + k]);
                    c[k] = fg * c[k] + ig * cg;
                    h[k] = og * apply(L.act, c[k]);
                }
                if (L.return_sequences) out.push_back(h);
            }
            if (L.return_sequences) seq = std::move(out);
            else                    seq = {h};
        } else {
            for (auto& xt : seq) {
                if (static_cast<int>(xt.size()) != L.in_dim)
                    throw std::runtime_error("F10Ensemble: Dense input size mismatch");
                std::vector<double> y(u);
                for (int j = 0; j < u; ++j) {
                    double s = L.bias[j];
                    for (int i = 0; i < L.in_dim; ++i) s += xt[i] * L.kernel[i * u + j];
                    y[j] = apply(L.act, s);
                }
                xt = std::move(y);
            }
        }
    }
    if (seq.size() != 1 || seq[0].empty())
        throw std::runtime_error("F10Ensemble: member output is not a single vector");
    return seq[0][0];
}

// Autoregressive daily rollout, as the reference dynamic_pred: append [prediction, extrapolated
// backward mean] and slide the window by one day (two values).
std::vector<double> F10Ensemble::rollout(const Member& m, std::vector<double> x, int n_days) const {
    const int T = static_cast<int>(x.size());
    std::vector<double> out(n_days);
    for (int d = 0; d < n_days; ++d) {
        // mean(diff(x[::-2])) + x[-1]: x[::-2] = x[T-1], x[T-3], ..., x[1] (the backward means)
        double sum = 0.0;
        int    cnt = 0;
        for (int i = T - 1; i - 2 >= 0; i -= 2) { sum += x[i - 2] - x[i]; ++cnt; }
        const double back_ave = (cnt > 0 ? sum / cnt : 0.0) + x[T - 1];
        const double pred = predict(m, x);
        out[d] = pred;
        for (int i = 0; i + 2 < T; ++i) x[i] = x[i + 2];
        x[T - 2] = pred;
        x[T - 1] = back_ave;
    }
    return out;
}

F10Forecast F10Ensemble::forecast(const std::vector<double>& history, int n_days) const {
    if (n_days <= 0) throw std::runtime_error("F10Ensemble::forecast: n_days must be > 0");
    if (static_cast<int>(history.size()) < history_days())
        throw std::runtime_error("F10Ensemble::forecast: need " + std::to_string(history_days()) +
                                 " days of F10.7 history, got " + std::to_string(history.size()));

    // Window: last max_lookback days of [f10, trailing avg_days mean], interleaved, normalized.
    const int H = static_cast<int>(history.size());
    std::vector<double> flat(static_cast<std::size_t>(2 * max_lookback_));
    for (int k = 0; k < max_lookback_; ++k) {
        const int d = H - max_lookback_ + k;
        double s = 0.0;
        for (int j = d - avg_days_ + 1; j <= d; ++j) s += history[j];
        flat[2 * k]     = (history[d] - scale_mean_) / scale_std_;
        flat[2 * k + 1] = (s / avg_days_ - scale_mean_) / scale_std_;
    }

    std::vector<std::vector<double>> preds(members_.size());
#ifdef ROPE_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(members_.size()); ++i) {
        const Member& m = members_[i];
        std::vector<double> x(flat.end() - 2 * m.lookback, flat.end());
        preds[i] = rollout(m, std::move(x), n_days);
    }

    F10Forecast fc;
    fc.mean.assign(n_days, 0.0);
    fc.std.assign(n_days, 0.0);
    const double n = static_cast<double>(members_.size());
    for (int d = 0; d < n_days; ++d) {
        double s = 0.0;
        for (const auto& p : preds) s += p[d] * scale_std_ + scale_mean_;
        const double mu = s / n;
        double v = 0.0;
        for (const auto& p : preds) { const double e = p[d] * scale_std_ + scale_mean_ - mu; v += e * e; }
        fc.mean[d] = mu;
        fc.std[d]  = std::sqrt(v / n);
    }
    return fc;
}

} // namespace rope::forecast
