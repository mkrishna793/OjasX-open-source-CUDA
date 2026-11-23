// Hardened C++20 zero-leakage gradient/loss gate and Merkle ledger primitive
// for OjasX-Aegis. Compiled with -O3 -std=c++20 -shared -fPIC -fvisibility=hidden.

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(_WIN32)
#define AEGIS_EXPORT extern "C" __declspec(dllexport)
#else
#define AEGIS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr const char* kAegisGateVersion = "ojasx-aegis-gate-1.0.0";
constexpr const char kMaskDomain[] = "OJASX_AEGIS_MASK_V1:";
constexpr size_t kMaskDomainLen = sizeof(kMaskDomain) - 1;

enum AegisStatus : int32_t {
  AEGIS_OK = 0,
  AEGIS_ERR_NULL_PTR = -1,
  AEGIS_ERR_INVALID_COUNT = -2,
  AEGIS_ERR_NON_FINITE = -3,
  AEGIS_ERR_INVALID_WEIGHT = -4,
  AEGIS_ERR_FLAG_WEIGHT_CONFLICT = -5,
  AEGIS_ERR_MASK_TAMPERED = -6,
  AEGIS_ERR_ZERO_LEAKAGE_FAILED = -7,
};

// Self-contained FIPS 180-4 SHA-256 (zero external OpenSSL linkage dependency).
struct Sha256 {
  uint32_t state[8];
  uint64_t bit_len;
  uint8_t buffer[64];
  size_t buf_len;

  static constexpr uint32_t K[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
      0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
      0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
      0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
      0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
      0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
      0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
  };

  void init() noexcept {
    state[0] = 0x6a09e667u;
    state[1] = 0xbb67ae85u;
    state[2] = 0x3c6ef372u;
    state[3] = 0xa54ff53au;
    state[4] = 0x510e527fu;
    state[5] = 0x9b05688cu;
    state[6] = 0x1f83d9abu;
    state[7] = 0x5be0cd19u;
    bit_len = 0;
    buf_len = 0;
  }

  void compress(const uint8_t block[64]) noexcept {
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) {
      const size_t j = i * 4;
      w[i] = (static_cast<uint32_t>(block[j]) << 24) |
             (static_cast<uint32_t>(block[j + 1]) << 16) |
             (static_cast<uint32_t>(block[j + 2]) << 8) |
             static_cast<uint32_t>(block[j + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
      const uint32_t s0 =
          std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 =
          std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (size_t i = 0; i < 64; ++i) {
      const uint32_t S1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = h + S1 + ch + K[i] + w[i];
      const uint32_t S0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = S0 + maj;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  void update(const uint8_t* data, size_t len) noexcept {
    if (len == 0) return;
    bit_len += static_cast<uint64_t>(len) * 8ULL;
    size_t offset = 0;

    if (buf_len > 0) {
      const size_t space = 64 - buf_len;
      const size_t take = (len < space) ? len : space;
      std::memcpy(buffer + buf_len, data, take);
      buf_len += take;
      offset += take;
      if (buf_len == 64) {
        compress(buffer);
        buf_len = 0;
      }
    }

    while (offset + 64 <= len) {
      compress(data + offset);
      offset += 64;
    }

    if (offset < len) {
      buf_len = len - offset;
      std::memcpy(buffer, data + offset, buf_len);
    }
  }

  void finalize(uint8_t out_digest[32]) noexcept {
    buffer[buf_len++] = 0x80;
    if (buf_len > 56) {
      std::memset(buffer + buf_len, 0, 64 - buf_len);
      compress(buffer);
      buf_len = 0;
    }
    std::memset(buffer + buf_len, 0, 56 - buf_len);
    for (int i = 0; i < 8; ++i) {
      buffer[56 + i] = static_cast<uint8_t>((bit_len >> ((7 - i) * 8)) & 0xFFu);
    }
    compress(buffer);

    for (size_t i = 0; i < 8; ++i) {
      out_digest[i * 4 + 0] = static_cast<uint8_t>((state[i] >> 24) & 0xFFu);
      out_digest[i * 4 + 1] = static_cast<uint8_t>((state[i] >> 16) & 0xFFu);
      out_digest[i * 4 + 2] = static_cast<uint8_t>((state[i] >> 8) & 0xFFu);
      out_digest[i * 4 + 3] = static_cast<uint8_t>(state[i] & 0xFFu);
    }
  }
};

inline void encode_be64(uint64_t v, uint8_t out[8]) noexcept {
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<uint8_t>((v >> ((7 - i) * 8)) & 0xFFu);
  }
}

inline bool constant_time_equal_32(const uint8_t* a, const uint8_t* b) noexcept {
  uint8_t diff = 0;
  for (size_t i = 0; i < 32; ++i) {
    diff |= static_cast<uint8_t>(a[i] ^ b[i]);
  }
  return diff == 0;
}

inline void zero_f32_buffer(float* buf, size_t total) noexcept {
  if (buf == nullptr) return;
  for (size_t i = 0; i < total; ++i) {
    buf[i] = 0.0f;
  }
}

int32_t verify_mask_and_hash_internal(
    const float* weights,
    const uint8_t* flags,
    size_t n,
    const uint8_t* expected_sha256,
    uint8_t* out_sha256) noexcept {
  if (weights == nullptr) {
    return AEGIS_ERR_NULL_PTR;
  }
  if (n == 0) {
    return AEGIS_ERR_INVALID_COUNT;
  }

  Sha256 hasher;
  hasher.init();
  hasher.update(reinterpret_cast<const uint8_t*>(kMaskDomain), kMaskDomainLen);
  uint8_t len_be[8];
  encode_be64(static_cast<uint64_t>(n), len_be);
  hasher.update(len_be, 8);

  for (size_t i = 0; i < n; ++i) {
    const float w = weights[i];
    if (!std::isfinite(w)) {
      return AEGIS_ERR_NON_FINITE;
    }
    if (w != 0.0f && w != 1.0f) {
      return AEGIS_ERR_INVALID_WEIGHT;
    }
    if (flags != nullptr && flags[i] != 0 && w != 0.0f) {
      return AEGIS_ERR_FLAG_WEIGHT_CONFLICT;
    }
    const uint8_t canonical_bit = (w == 1.0f) ? 0x01u : 0x00u;
    hasher.update(&canonical_bit, 1);
  }

  uint8_t digest[32];
  hasher.finalize(digest);
  if (out_sha256 != nullptr) {
    std::memcpy(out_sha256, digest, 32);
  }
  if (expected_sha256 != nullptr && !constant_time_equal_32(digest, expected_sha256)) {
    return AEGIS_ERR_MASK_TAMPERED;
  }
  return AEGIS_OK;
}

}  // namespace

AEGIS_EXPORT const char* aegis_gate_version(void) {
  return kAegisGateVersion;
}

AEGIS_EXPORT int32_t aegis_verify_mask_integrity(
    const float* weights,
    const uint8_t* flags,
    size_t n,
    const uint8_t* expected_sha256,
    uint8_t* out_sha256) {
  return verify_mask_and_hash_internal(
      weights, flags, n, expected_sha256, out_sha256);
}

AEGIS_EXPORT int32_t aegis_apply_zero_mask_f32(
    const float* values_in,
    float* values_out,
    const float* weights,
    const uint8_t* flags,
    size_t n,
    size_t stride,
    const uint8_t* expected_mask_sha256,
    uint8_t* out_mask_sha256) {
  if (values_in == nullptr || values_out == nullptr || weights == nullptr) {
    return AEGIS_ERR_NULL_PTR;
  }
  if (n == 0 || stride == 0 || n > (std::numeric_limits<size_t>::max() / stride)) {
    return AEGIS_ERR_INVALID_COUNT;
  }

  const size_t total = n * stride;
  const int32_t mask_status = verify_mask_and_hash_internal(
      weights, flags, n, expected_mask_sha256, out_mask_sha256);
  if (mask_status != AEGIS_OK) {
    zero_f32_buffer(values_out, total);
    return mask_status;
  }

  for (size_t i = 0; i < n; ++i) {
    const bool allow = (weights[i] == 1.0f) && (flags == nullptr || flags[i] == 0);
    const size_t base = i * stride;
    if (!allow) {
      // Zero-leakage enforcement: write exact +0.0f without multiplying by values_in.
      for (size_t k = 0; k < stride; ++k) {
        values_out[base + k] = 0.0f;
      }
    } else {
      for (size_t k = 0; k < stride; ++k) {
        const float v = values_in[base + k];
        if (!std::isfinite(v)) {
          zero_f32_buffer(values_out, total);
          return AEGIS_ERR_NON_FINITE;
        }
        values_out[base + k] = v;
      }
    }
  }

  // Mandatory zero-leakage postcondition audit over output buffer.
  for (size_t i = 0; i < n; ++i) {
    const bool allow = (weights[i] == 1.0f) && (flags == nullptr || flags[i] == 0);
    const size_t base = i * stride;
    if (!allow) {
      for (size_t k = 0; k < stride; ++k) {
        uint32_t bits = 0;
        std::memcpy(&bits, &values_out[base + k], sizeof(bits));
        if (bits != 0u) {
          zero_f32_buffer(values_out, total);
          return AEGIS_ERR_ZERO_LEAKAGE_FAILED;
        }
      }
    }
  }

  return AEGIS_OK;
}

AEGIS_EXPORT int32_t aegis_compute_merkle_step(
    const uint8_t* prev_chain_hash,
    const uint8_t* merkle_root,
    const uint8_t* policy_hash,
    const uint8_t* mask_hash,
    const uint8_t* grad_sketch,
    uint64_t step,
    uint8_t* out_chain_hash) {
  if (prev_chain_hash == nullptr || merkle_root == nullptr ||
      policy_hash == nullptr || mask_hash == nullptr ||
      grad_sketch == nullptr || out_chain_hash == nullptr) {
    return AEGIS_ERR_NULL_PTR;
  }

  uint8_t preimage[168];
  std::memcpy(preimage + 0, prev_chain_hash, 32);
  std::memcpy(preimage + 32, merkle_root, 32);
  std::memcpy(preimage + 64, policy_hash, 32);
  std::memcpy(preimage + 96, mask_hash, 32);
  std::memcpy(preimage + 128, grad_sketch, 32);
  encode_be64(step, preimage + 160);

  uint8_t digest[32];
  Sha256 hasher;
  hasher.init();
  hasher.update(preimage, sizeof(preimage));
  hasher.finalize(digest);
  std::memcpy(out_chain_hash, digest, 32);
  return AEGIS_OK;
}
