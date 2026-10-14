#pragma once
// F10.7 forecaster ensemble (.f10bin) — the [f10_forecaster] add-on.
//
// Daily, autoregressive F10.7 forecast from an ensemble of small LSTM / MLP networks (the
// "dynamic lookback" ensembles of Daniell & Mehta 2023, 180 members: lookbacks 7..22 days x
// 3 architectures x 10 seeds). Each member sees the last `lookback` days of
// [f10, backward 12-day mean] (normalized with one scalar mean/std) and predicts the next day;
// the prediction and an extrapolated backward mean are appended and the window slides. The
// ensemble mean is the forecast and the ensemble spread (population std) its uncertainty.
//
// File layout: see ROPE-TIEGCM tools/export_framework_addons.py (written by it). Weights are
// float32 on disk; the forward pass runs in double, as the reference implementation does.

#include <filesystem>
#include <string>
#include <vector>

namespace rope::forecast {

struct F10Forecast {
    std::vector<double> mean;   // (n_days) sfu, day0 .. day0+n_days-1
    std::vector<double> std;    // (n_days) sfu, ensemble population std
};

class F10Ensemble {
public:
    static F10Ensemble load(const std::filesystem::path& path);

    // history: daily F10.7 (sfu), oldest first, ending with the day before day0.
    // Needs at least history_days() values.
    F10Forecast forecast(const std::vector<double>& history, int n_days) const;

    int history_days() const { return max_lookback_ + avg_days_ - 1; }
    int n_members()    const { return static_cast<int>(members_.size()); }

private:
    enum class Act : int { Linear = 0, Tanh = 1, Sigmoid = 2, Relu = 3 };
    struct Layer {
        bool lstm = false;
        Act  act = Act::Linear, rec_act = Act::Sigmoid;
        bool return_sequences = false;
        int  in_dim = 0, units = 0;
        std::vector<float> kernel, recurrent, bias;
    };
    struct Member {
        int  lookback = 0;
        bool sequence_input = true;   // LSTM: (2*lookback, 1); MLP: flat 2*lookback
        std::vector<Layer> layers;
    };

    double scale_mean_ = 0.0, scale_std_ = 1.0;
    int    max_lookback_ = 0, n_features_ = 0, avg_days_ = 0;
    std::vector<Member> members_;

    double predict(const Member& m, const std::vector<double>& x) const;
    std::vector<double> rollout(const Member& m, std::vector<double> x, int n_days) const;
    static double apply(Act a, double v);
};

} // namespace rope::forecast
