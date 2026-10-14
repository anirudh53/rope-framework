#pragma once
// Data-assimilation initial-condition table (.dabin) — the [data_assimilation] add-on.
//
// One record per hour T of an assimilation run: the seq_len hourly latent states that seed the
// rollout (T-S+1 .. T, oldest first) and their joint covariance, already expressed in the
// model's own latent space (written by the model's export tooling, e.g. ROPE-TIEGCM's
// tools/export_framework_addons.py, which fits latents to this model's decoder).
//
// Header (little-endian):
//   uint32  magic      = 0x52504441  ("RPDA")
//   uint32  version    = 1
//   uint32  n_records
//   uint32  seq_len    (S)
//   uint32  latent_dim (K)
//   uint32  source_len, char source[source_len]   -- free-text provenance, not NUL-terminated
//
// Records (n_records, sorted by tp):
//   int64   tp                    -- hour T, UTC seconds since epoch
//   int32   n_obs_span            -- observations that informed these hours (0 = model only)
//   float32 z[S*K]                -- latent states, hours T-S+1 .. T
//   float32 sigma_joint[(S*K)^2]  -- joint covariance, row-major, same ordering as z

#include "rope/core/datetime.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rope::io {

struct DAState {
    TimePoint          tp = 0;
    int                n_obs_span = 0;
    std::vector<float> z;            // (S*K)
    std::vector<float> sigma_joint;  // (S*K)^2
};

class DAStateTable {
public:
    // Throws on missing file, bad magic/version, or truncated data.
    static DAStateTable load(const std::filesystem::path& path);

    int seq_len()    const { return S_; }
    int latent_dim() const { return K_; }
    std::size_t size() const { return tps_.size(); }
    TimePoint time_min() const { return tps_.front(); }
    TimePoint time_max() const { return tps_.back(); }
    const std::string& source() const { return source_; }

    // State for hour T. Throws (naming the table's coverage) when T is not in the table.
    DAState at(TimePoint T) const;

private:
    int S_ = 0, K_ = 0;
    std::string            source_;
    std::vector<TimePoint> tps_;
    std::vector<int>       n_obs_;
    std::vector<float>     z_;      // n * S*K
    std::vector<float>     sigma_;  // n * (S*K)^2
};

} // namespace rope::io
