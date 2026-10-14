#include "rope/io/da_states.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace rope::io {

static constexpr std::uint32_t DA_MAGIC      = 0x52504441u;  // "RPDA"
static constexpr std::uint32_t DA_VERSION    = 1u;
static constexpr std::uint32_t DA_MAX_SOURCE = 1u << 16;

DAStateTable DAStateTable::load(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("DAStateTable::load: cannot open " + path.string());

    std::uint32_t magic = 0, version, n, S, K, src_len;
    f.read(reinterpret_cast<char*>(&magic),   4);
    if (!f || magic != DA_MAGIC)
        throw std::runtime_error("DAStateTable::load: bad magic in " + path.string() +
                                 " (not a .dabin file)");
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&n),       4);
    f.read(reinterpret_cast<char*>(&S),       4);
    f.read(reinterpret_cast<char*>(&K),       4);
    f.read(reinterpret_cast<char*>(&src_len), 4);
    if (!f)
        throw std::runtime_error("DAStateTable::load: failed to read header from " + path.string());
    if (version != DA_VERSION)
        throw std::runtime_error("DAStateTable::load: unsupported version " +
                                 std::to_string(version) + " in " + path.string());
    if (n == 0 || S == 0 || K == 0)
        throw std::runtime_error("DAStateTable::load: empty table or zero dimension in " +
                                 path.string());
    if (src_len > DA_MAX_SOURCE)
        throw std::runtime_error("DAStateTable::load: implausible source length in " + path.string());

    DAStateTable t;
    t.S_ = static_cast<int>(S);
    t.K_ = static_cast<int>(K);
    t.source_.assign(src_len, '\0');
    f.read(t.source_.data(), src_len);

    const std::size_t nz = static_cast<std::size_t>(S) * K;
    t.tps_.resize(n);
    t.n_obs_.resize(n);
    t.z_.resize(n * nz);
    t.sigma_.resize(n * nz * nz);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::int64_t  tp;
        std::int32_t  nobs;
        f.read(reinterpret_cast<char*>(&tp),   8);
        f.read(reinterpret_cast<char*>(&nobs), 4);
        f.read(reinterpret_cast<char*>(t.z_.data() + i * nz), static_cast<std::streamsize>(nz * 4));
        f.read(reinterpret_cast<char*>(t.sigma_.data() + i * nz * nz),
               static_cast<std::streamsize>(nz * nz * 4));
        if (!f)
            throw std::runtime_error("DAStateTable::load: unexpected EOF at record " +
                                     std::to_string(i) + " in " + path.string());
        t.tps_[i]   = static_cast<TimePoint>(tp);
        t.n_obs_[i] = nobs;
    }
    if (!std::is_sorted(t.tps_.begin(), t.tps_.end()))
        throw std::runtime_error("DAStateTable::load: records not sorted by time in " + path.string());
    return t;
}

DAState DAStateTable::at(TimePoint T) const {
    auto it = std::lower_bound(tps_.begin(), tps_.end(), T);
    if (it == tps_.end() || *it != T)
        throw std::runtime_error(
            "data assimilation: no assimilated state for " + format_iso(T) +
            "; the DA table covers " + format_iso(time_min()) + " .. " + format_iso(time_max()) +
            " (hourly). Choose a start inside that range or disable [data_assimilation].");
    const std::size_t i  = static_cast<std::size_t>(it - tps_.begin());
    const std::size_t nz = static_cast<std::size_t>(S_) * K_;
    DAState s;
    s.tp         = T;
    s.n_obs_span = n_obs_[i];
    s.z.assign(z_.begin() + i * nz, z_.begin() + (i + 1) * nz);
    s.sigma_joint.assign(sigma_.begin() + i * nz * nz, sigma_.begin() + (i + 1) * nz * nz);
    return s;
}

} // namespace rope::io
