#ifdef EVAL_MULTILAYER
#include <eval/sparse.h>

using namespace raphael::nnue;
using std::popcount;



#ifdef USE_SIMD
void SparseIterator::add_nonzeros(VecU8 l0_out0, VecU8 l0_out1) {
    constexpr i32 regw32 = ALIGNMENT / sizeof(i32);

#ifdef USE_AVX512
    static_assert(USE_SIMD == 512);
    static_assert(regw32 % 8 == 0);
    const auto mask = _mm512_kunpackw(nonzero_mask(l0_out1), nonzero_mask(l0_out0));
    const auto idxs = _mm512_maskz_compress_epi16(mask, offset_);
    _mm512_storeu_si512(&indices_[count_], idxs);
    offset_ = add_i16(offset_, full_i16(32));
    count_ += popcount(mask);
#elif defined(USE_NEON)
    static_assert(USE_SIMD == 128);
    static_assert(regw32 == 4);

    const u32 full_mask = (nonzero_mask(l0_out1) << regw32) | nonzero_mask(l0_out0);
    const u8 mask = static_cast<u8>(full_mask & 0xFF);
    const auto idxs = add_i16(offset_, load_i16(reinterpret_cast<const i16*>(&nonzero_idx[mask])));

    // Store only the active indices; this keeps writes within indices_ even
    // when the final sparse batch has fewer than eight nonzero tiles.
    alignas(16) i16 packed[8];
    store_i16(packed, idxs);
    const i32 nnz = popcount(mask);
    for (i32 i = 0; i < nnz; ++i) indices_[count_ + i] = static_cast<u16>(packed[i]);

    offset_ = add_i16(offset_, full_i16(8));
    count_ += nnz;
#else
    static_assert(regw32 % 8 == 0);
    u32 full_mask = (nonzero_mask(l0_out1) << regw32) | nonzero_mask(l0_out0);

    for (i32 i = 0; i < regw32 / 4; i++) {
        const u8 mask = full_mask & 0xFF;
        full_mask >>= 8;
        const auto idxs = _mm_add_epi16(
            offset_, _mm_load_si128(reinterpret_cast<const __m128i*>(&nonzero_idx[mask]))
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&indices_[count_]), idxs);
        offset_ = _mm_add_epi16(offset_, _mm_set1_epi16(8));
        count_ += popcount(mask);
    }
#endif

    assert(count_ <= L1_SIZE / 4);
}

i32 SparseIterator::count() const { return count_; }

i32 SparseIterator::index(i32 nnz_id) const {
    assert(nnz_id < count_);
    return indices_[nnz_id];
}
#endif
#endif
