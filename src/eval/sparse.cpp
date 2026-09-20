#ifdef EVAL_NNUE

#include <eval/sparse.h>

#ifdef USE_SIMD

namespace raphael::nnue::sparse {

namespace {

alignas(ALIGNMENT) static constexpr u16 nonzero_idx[256][8] = [] {
    u16 table[256][8]{};

    for (u32 mask = 0; mask < 256; ++mask) {
        u32 count = 0;

        for (u32 i = 0; i < 8; ++i) {
            if (mask & (1u << i))
                table[mask][count++] = static_cast<u16>(i);
        }
    }

    return table;
}();

}  // namespace


SparseInput SparseInput::load(const void* src) {
    return {load_i16(src)};
}


void SparseInput::store_into(void* dst) const {
    store_i16(dst, offset_);
}


}  // namespace raphael::nnue::sparse

#endif
#endif
