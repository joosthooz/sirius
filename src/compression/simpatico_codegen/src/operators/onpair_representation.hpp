// SPDX-License-Identifier: Apache-2.0
// OnPair-GPU string codec — decode-only port of Vortex's onpair decode kernels
// (vortex-cuda/kernels/src/onpair_shmem_4tpt_split8read.cu +
// vortex-cuda/cub/kernels/onpair.cu) into simpatico's leaf-codec shape,
// alongside fsst_compressor.cu. There is no GPU OnPair *compressor* anywhere
// upstream to port (Vortex's own onpair_compress is CPU-only) — compress()
// here is an intentional stub. Compressed data is produced externally (by
// Vortex's CPU compressor) and loaded via load_from_dump().
//
// This is a standalone, self-contained representation — deliberately NOT
// wired into operator_registry.cpp / representation_factory.cpp's DSL
// dispatch tables (no plan file / .hpln round-trip support), since nothing
// can compress onpair data through Sirius's plan interpreter yet. Only the
// OpId::OnPair enum tag was added to the shared registry header for identity.
//
// v4 ("ONP4") format: chars-only, decoupled from row structure — same split
// FSST already has under str_split (str_split.chars -> fsst is a plain leaf
// codec that never sees row offsets at all; str_split.offsets -> bitpack is a
// wholly separate, already-solved cascade). Earlier versions (ONP2/ONP3)
// bolted a bespoke row-offsets encoding (raw u32-per-row, then a hand-rolled
// bitpack) onto this representation; that's gone. decompress() now returns
// only the flat decoded chars buffer (a plain UINT8 column, uncompressed_bytes
// elements — exactly fsst_compressed_representation::decompress()'s contract
// for the chars leaf under str_split, see fsst_compressor.cu). Row offsets are
// the caller's job: load them as ground-truth plain values via
// load_row_offsets_from_dump() and round-trip them through Sirius's real,
// already-working str_split/bitpack machinery
// (simpatico::compress_with_plan / simpatico::decompress with the plan
// "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed" — the
// identical cascade FSST's own plans already use for str_split.offsets), then
// zip the two decoded pieces into a STRING column the same way the plan
// interpreter combines str_split's offsets/chars children.
//
// Precomputed chars-side metadata (chunk_offsets, one u64 per 128-token decode
// batch) is unchanged from v2/v3 and stays: it is this codec's own placement
// metadata, the direct analogue of FSST's chunk_uncomp frame-header field, not
// row-offset data. It is computed at dump/compress time (dump_onpair.rs) so
// the GPU never has to rediscover it via a scan at decode time.

#pragma once

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>

#include <rmm/device_buffer.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {

// On-disk dump format written by vortex-src/vortex-cuda/examples/dump_onpair.rs
// ("ONP4"):
//   char[4]  magic "ONP4"
//   u64 n_rows
//   u64 dict_size
//   u64 n_tokens
//   u64 code_width         (always 2 on disk: dump tool widens u8 codes to u16)
//   u64 uncompressed_bytes
//   u64 num_batches        (n_tokens / 128, rounded up)
//   u8  dict_padded[dict_size * 16]      (MAX_TOKEN_SIZE = 16)
//   u8  dict_s8[dict_size * 8]
//   u8  lens[dict_size]
//   u16 codes[n_tokens]                  (little-endian)
//   u64 chunk_offsets[num_batches + 1]   (little-endian; precomputed output byte offset preceding
//                                          each 128-token decode batch — feeds the decode kernel
//                                          directly, no GPU-side scan needed)
//   --- out-of-band ground truth, NOT part of the compressed payload ---
//   u64 row_offsets_len                  (== n_rows + 1)
//   i32 row_offsets[row_offsets_len]     (plain, little-endian; for the caller to compress via
//                                          Sirius's own str_split/bitpack path, not this codec)
struct onpair_compressed_representation : standalone_compressed_representation {
  std::size_t dict_size          = 0;
  std::size_t n_tokens           = 0;
  std::size_t num_batches        = 0;
  std::size_t uncompressed_bytes = 0;

  rmm::device_buffer dict_padded_buf;    // dict_size * 16 bytes
  rmm::device_buffer dict_s8_buf;        // dict_size * 8 bytes
  rmm::device_buffer lens_buf;           // dict_size bytes
  rmm::device_buffer codes_buf;          // n_tokens * 2 bytes (u16)
  rmm::device_buffer chunk_offsets_buf;  // (num_batches + 1) * 8 bytes (u64) — precomputed

  onpair_compressed_representation(cudf::size_type n_rows,
                                   std::size_t dict_size_,
                                   std::size_t n_tokens_,
                                   std::size_t num_batches_,
                                   std::size_t uncompressed_bytes_,
                                   rmm::device_buffer dict_padded,
                                   rmm::device_buffer dict_s8,
                                   rmm::device_buffer lens,
                                   rmm::device_buffer codes,
                                   rmm::device_buffer chunk_offsets)
    // original_type is UINT8: this leaf decompresses to a flat chars byte
    // buffer, not a STRING column — same contract as fsst under str_split.
    // num_rows here tracks the original row count for bookkeeping only (it is
    // not used to shape the decompressed output, which is byte-indexed).
    : standalone_compressed_representation(cudf::data_type{cudf::type_id::UINT8}, n_rows),
      dict_size(dict_size_),
      n_tokens(n_tokens_),
      num_batches(num_batches_),
      uncompressed_bytes(uncompressed_bytes_),
      dict_padded_buf(std::move(dict_padded)),
      dict_s8_buf(std::move(dict_s8)),
      lens_buf(std::move(lens)),
      codes_buf(std::move(codes)),
      chunk_offsets_buf(std::move(chunk_offsets))
  {
  }

  // Decodes the OnPair token stream to a flat UINT8 column of
  // `uncompressed_bytes` elements — the chars payload only, no row offsets.
  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const override;

  // Benchmarking helper (not part of the standalone_compressed_representation
  // interface): times the isolated decode kernel alone across `iters` repeated
  // launches, matching how Vortex's raw kernel throughput was measured (nsys,
  // offsets/materialization overhead excluded). With chunk_offsets precomputed
  // at dump time, this differs from decompress()'s full-pipeline timing only
  // by the (already-cheap) output-buffer allocation.
  double time_decode_kernel_only_ms(rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr,
                                    int iters) const;

  OpId kind() const override { return OpId::OnPair; }

  std::size_t compressed_payload_bytes() const
  {
    return dict_padded_buf.size() + dict_s8_buf.size() + lens_buf.size() + codes_buf.size() +
          chunk_offsets_buf.size();
  }

  // Loads a dump_onpair.rs "ONP4" binary file's chars-only payload straight to
  // device buffers (row_offsets are skipped here — see load_row_offsets_from_dump).
  static std::unique_ptr<onpair_compressed_representation> load_from_dump(
    std::string const& path, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  // Reads the same dump file's out-of-band ground-truth row offsets as a
  // plain host-side vector (n_rows + 1 int32 values), for the caller to feed
  // into Sirius's real str_split/bitpack compress/decompress path. This is
  // deliberately host-side and separate from load_from_dump(): it is test/
  // bench scaffolding standing in for "the original string column's offsets,"
  // not part of what onpair_compressed_representation itself represents.
  static std::vector<std::int32_t> load_row_offsets_from_dump(std::string const& path);
};

struct onpair_compressor : compressor {
  std::unique_ptr<compressed_representation> compress(cudf::column_view,
                                                      rmm::cuda_stream_view,
                                                      rmm::device_async_resource_ref) override
  {
    throw std::runtime_error(
      "onpair: GPU compression is not implemented (no GPU OnPair encoder exists upstream in "
      "Vortex either) — this is a decode-only port. Use "
      "onpair_compressed_representation::load_from_dump() with data produced by Vortex's CPU "
      "onpair_compress.");
  }
};

}  // namespace simpatico
