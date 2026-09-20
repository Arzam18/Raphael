#pragma once

#ifdef EVAL_NNUE

#include <eval/simd.h>

#ifdef USE_SIMD

namespace raphael::nnue::sparse {

struct SparseInput {
    VecI16 offset_;

    [[nodiscard]] static SparseInput load(
        const void* src
    );

    void store_into(void* dst) const;
};

}  // namespace raphael::nnue::sparse

#endif
#endif
