// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace simpatico {

// Lookup modes the fused Dict encoder understands (dict_meta[2]).
inline constexpr std::int64_t kDictLookupBitmap = 0;
inline constexpr std::int64_t kDictLookupSearch = 1;

// A key range up to this wide is indexed through a presence bitmap (2 MB at the limit) rather
// than by binary search over the keys.
inline constexpr std::uint64_t kDictMaxBitmapRange = std::uint64_t{1} << 24;

// Whether codes for `num_keys` keys are at least two bits narrower than offsets over `span`
// (max - min); short of that, a frame-of-reference bitpack stores the values about as narrowly.
[[nodiscard]] inline bool dict_codes_pay(std::int64_t num_keys, std::uint64_t span)
{
  return std::bit_width(static_cast<std::uint64_t>(num_keys - 1)) + 2 <= std::bit_width(span);
}

// The sorted distinct values of a column and the lookup structures the fused Dict encoder reads
// to map a value to its index among them. Field layout matches the kernel's input buffers.
struct dict_keys {
  rmm::device_buffer meta;    // int64[4]: {key minimum, key count, lookup mode, bitmap words}
  rmm::device_buffer bitmap;  // uint32[max(1, words)]
  rmm::device_buffer rank;    // int32[max(1, words)]: keys below each bitmap word
  rmm::device_buffer keys;    // elem[num_keys], ascending as signed integers of elem_size bytes
  std::int32_t num_keys{0};
};

// Build the dictionary of `n` elements of `elem_size` bytes (1, 2, 4 or 8) at `data`, compared as
// signed integers of that width. Returns nullopt when the column has more than `max_keys`
// distinct values. Stream-ordered; the result is valid on `stream`.
[[nodiscard]] std::optional<dict_keys> build_dict_keys(void const* data,
                                                       std::int64_t n,
                                                       std::size_t elem_size,
                                                       std::int64_t max_keys,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr);

}  // namespace simpatico
