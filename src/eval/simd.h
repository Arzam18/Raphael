#pragma once

#ifdef EVAL_NNUE

#include <chess/types.h>

#if defined(__aarch64__) || defined(__ARM_NEON)

#include <arm_neon.h>

#define USE_NEON
#define USE_SIMD 128
#define ALIGNMENT 16
#define SIMD_REGS 8

struct NeonVec {
    int8x16_t raw;
};

using VecU8 = NeonVec;
using VecI8 = NeonVec;
using VecI16 = NeonVec;
using VecI32 = NeonVec;

[[nodiscard]] inline VecU8 load_u8(const u8* src) {
    return {vld1q_s8(reinterpret_cast<const int8_t*>(src))};
}

[[nodiscard]] inline VecI8 load_i8(const i8* src) {
    return {vld1q_s8(reinterpret_cast<const int8_t*>(src))};
}

[[nodiscard]] inline VecI16 load_i16(const i16* src) {
    return {
        vreinterpretq_s8_s16(
            vld1q_s16(src)
        )
    };
}

[[nodiscard]] inline VecI32 load_i32(const i32* src) {
    return {
        vreinterpretq_s8_s32(
            vld1q_s32(src)
        )
    };
}

inline void store_u8(u8* dst, VecU8 src) {
    vst1q_u8(
        reinterpret_cast<uint8_t*>(dst),
        vreinterpretq_u8_s8(src.raw)
    );
}

inline void store_i16(i16* dst, VecI16 src) {
    vst1q_s16(
        dst,
        vreinterpretq_s16_s8(src.raw)
    );
}

inline void store_i32(i32* dst, VecI32 src) {
    vst1q_s32(
        dst,
        vreinterpretq_s32_s8(src.raw)
    );
}

[[nodiscard]] inline VecI16 zero_i16() {
    return {
        vdupq_n_s8(0)
    };
}

[[nodiscard]] inline VecI32 zero_i32() {
    return {
        vdupq_n_s8(0)
    };
}

[[nodiscard]] inline VecI16 full_i16(i16 val) {
    return {
        vreinterpretq_s8_s16(
            vdupq_n_s16(val)
        )
    };
}

[[nodiscard]] inline VecI32 full_i32(i32 val) {
    return {
        vreinterpretq_s8_s32(
            vdupq_n_s32(val)
        )
    };
}

[[nodiscard]] inline VecI16 add_i16(VecI16 a, VecI16 b) {
    return {
        vreinterpretq_s8_s16(
            vaddq_s16(
                vreinterpretq_s16_s8(a.raw),
                vreinterpretq_s16_s8(b.raw)
            )
        )
    };
}

[[nodiscard]] inline VecI32 add_i32(VecI32 a, VecI32 b) {
    return {
        vreinterpretq_s8_s32(
            vaddq_s32(
                vreinterpretq_s32_s8(a.raw),
                vreinterpretq_s32_s8(b.raw)
            )
        )
    };
}

[[nodiscard]] inline VecI16 sub_i16(VecI16 a, VecI16 b) {
    return {
        vreinterpretq_s8_s16(
            vsubq_s16(
                vreinterpretq_s16_s8(a.raw),
                vreinterpretq_s16_s8(b.raw)
            )
        )
    };
}

[[nodiscard]] inline VecI32 mullo_i32(VecI32 a, VecI32 b) {
    return {
        vreinterpretq_s8_s32(
            vmulq_s32(
                vreinterpretq_s32_s8(a.raw),
                vreinterpretq_s32_s8(b.raw)
            )
        )
    };
}

[[nodiscard]] inline VecI16 mullo_i16(VecI16 a, VecI16 b) {
    return {
        vreinterpretq_s8_s16(
            vmulq_s16(
                vreinterpretq_s16_s8(a.raw),
                vreinterpretq_s16_s8(b.raw)
            )
        )
    };
}

[[nodiscard]] inline VecI16 mulhi_i16(VecI16 a, VecI16 b) {
    const auto aa = vreinterpretq_s16_s8(a.raw);
    const auto bb = vreinterpretq_s16_s8(b.raw);

    const auto lo = vmull_s16(
        vget_low_s16(aa),
        vget_low_s16(bb)
    );

    const auto hi = vmull_s16(
        vget_high_s16(aa),
        vget_high_s16(bb)
    );

    return {
        vreinterpretq_s8_s16(
            vcombine_s16(
                vshrn_n_s32(lo, 16),
                vshrn_n_s32(hi, 16)
            )
        )
    };
}

[[nodiscard]] inline VecI32 madd_i16(VecI16 a, VecI16 b) {
    const auto aa = vreinterpretq_s16_s8(a.raw);
    const auto bb = vreinterpretq_s16_s8(b.raw);

    const auto products = vmulq_s16(aa, bb);

    return {
        vreinterpretq_s8_s32(
            vpaddlq_s16(products)
        )
    };
}

[[nodiscard]] inline VecI16 clamp_i16(
    VecI16 reg,
    VecI16 mins,
    VecI16 maxs
) {
    const auto r = vreinterpretq_s16_s8(reg.raw);
    const auto lo = vreinterpretq_s16_s8(mins.raw);
    const auto hi = vreinterpretq_s16_s8(maxs.raw);

    return {
        vreinterpretq_s8_s16(
            vminq_s16(
                vmaxq_s16(r, lo),
                hi
            )
        )
    };
}

[[nodiscard]] inline VecI32 clamp_i32(
    VecI32 reg,
    VecI32 mins,
    VecI32 maxs
) {
    const auto r = vreinterpretq_s32_s8(reg.raw);
    const auto lo = vreinterpretq_s32_s8(mins.raw);
    const auto hi = vreinterpretq_s32_s8(maxs.raw);

    return {
        vreinterpretq_s8_s32(
            vminq_s32(
                vmaxq_s32(r, lo),
                hi
            )
        )
    };
}

[[nodiscard]] inline VecI16 lshift_i16(
    VecI16 reg,
    i32 shift
) {
    return {
        vreinterpretq_s8_s16(
            vshlq_s16(
                vreinterpretq_s16_s8(reg.raw),
                vdupq_n_s16(static_cast<i16>(shift))
            )
        )
    };
}

[[nodiscard]] inline VecI32 rshift_i32(
    VecI32 reg,
    i32 shift
) {
    return {
        vreinterpretq_s8_s32(
            vshlq_s32(
                vreinterpretq_s32_s8(reg.raw),
                vdupq_n_s32(-shift)
            )
        )
    };
}

[[nodiscard]] inline VecI16 low_i8_i16(VecI8 reg) {
    return {
        vreinterpretq_s8_s16(
            vmovl_s8(
                vget_low_s8(reg.raw)
            )
        )
    };
}

[[nodiscard]] inline VecI16 high_i8_i16(VecI8 reg) {
    return {
        vreinterpretq_s8_s16(
            vmovl_s8(
                vget_high_s8(reg.raw)
            )
        )
    };
}

[[nodiscard]] inline VecU8 pack_u8_i16(
    VecI16 a,
    VecI16 b
) {
    return {
        vreinterpretq_s8_u8(
            vcombine_u8(
                vqmovun_s16(
                    vreinterpretq_s16_s8(a.raw)
                ),
                vqmovun_s16(
                    vreinterpretq_s16_s8(b.raw)
                )
            )
        )
    };
}

[[nodiscard]] inline VecU8 tile_u8(const u8* vals) {
    return {
        vreinterpretq_s8_u32(
            vdupq_n_u32(
                *reinterpret_cast<const u32*>(vals)
            )
        )
    };
}

[[nodiscard]] inline u32 nonzero_mask(VecU8 reg) {
    const auto x = vreinterpretq_u32_s8(reg.raw);

    const auto nz = vcgtq_u32(
        x,
        vdupq_n_u32(0)
    );

    alignas(16) u32 lanes[4];

    vst1q_u32(lanes, nz);

    u32 mask = 0;

    for (u32 i = 0; i < 4; ++i)
        mask |= (lanes[i] != 0) << i;

    return mask;
}

[[nodiscard]] inline VecI32 dpbusd_i32(
    VecI32 a,
    VecU8 b,
    VecI8 c
) {
    const auto bu = vreinterpretq_u8_s8(b.raw);
    const auto cs = c.raw;

    const auto blo = vmovl_u8(
        vget_low_u8(bu)
    );

    const auto bhi = vmovl_u8(
        vget_high_u8(bu)
    );

    const auto clo = vmovl_s8(
        vget_low_s8(cs)
    );

    const auto chi = vmovl_s8(
        vget_high_s8(cs)
    );

    const auto plo = vmulq_s16(
        vreinterpretq_s16_u16(blo),
        clo
    );

    const auto phi = vmulq_s16(
        vreinterpretq_s16_u16(bhi),
        chi
    );

    const auto sums = vaddq_s32(
        vpaddlq_s16(plo),
        vpaddlq_s16(phi)
    );

    return add_i32(
        a,
        {
            vreinterpretq_s8_s32(sums)
        }
    );
}

[[nodiscard]] inline VecI32 fmadd_i32(
    VecI32 a,
    VecI32 b,
    VecI32 c
) {
    return add_i32(
        mullo_i32(a, b),
        c
    );
}

[[nodiscard]] inline i32 hadd_i32(VecI32 reg) {
    return vaddvq_s32(
        vreinterpretq_s32_s8(reg.raw)
    );
}

#elif defined(__AVX512VBMI2__)

#include <immintrin.h>

#define USE_AVX512
#define USE_SIMD 512
#define ALIGNMENT 64
#define SIMD_REGS 32

using VecU8 = __m512i;
using VecI8 = __m512i;
using VecI16 = __m512i;
using VecI32 = __m512i;

[[nodiscard]] inline VecU8 load_u8(const u8* src) {
    return _mm512_load_si512(src);
}

[[nodiscard]] inline VecI8 load_i8(const i8* src) {
    return _mm512_load_si512(src);
}

[[nodiscard]] inline VecI16 load_i16(const i16* src) {
    return _mm512_load_si512(src);
}

[[nodiscard]] inline VecI32 load_i32(const i32* src) {
    return _mm512_load_si512(src);
}

inline void store_u8(u8* dst, VecU8 src) {
    _mm512_store_si512(dst, src);
}

inline void store_i16(i16* dst, VecI16 src) {
    _mm512_store_si512(dst, src);
}

inline void store_i32(i32* dst, VecI32 src) {
    _mm512_store_si512(dst, src);
}

[[nodiscard]] inline VecI16 zero_i16() {
    return _mm512_setzero_si512();
}

[[nodiscard]] inline VecI32 zero_i32() {
    return _mm512_setzero_si512();
}

[[nodiscard]] inline VecI16 full_i16(i16 val) {
    return _mm512_set1_epi16(val);
}

[[nodiscard]] inline VecI32 full_i32(i32 val) {
    return _mm512_set1_epi32(val);
}

[[nodiscard]] inline VecI16 add_i16(VecI16 a, VecI16 b) {
    return _mm512_add_epi16(a, b);
}

[[nodiscard]] inline VecI32 add_i32(VecI32 a, VecI32 b) {
    return _mm512_add_epi32(a, b);
}

[[nodiscard]] inline VecI16 sub_i16(VecI16 a, VecI16 b) {
    return _mm512_sub_epi16(a, b);
}

[[nodiscard]] inline VecI32 mullo_i32(VecI32 a, VecI32 b) {
    return _mm512_mullo_epi32(a, b);
}

[[nodiscard]] inline VecI16 mullo_i16(VecI16 a, VecI16 b) {
    return _mm512_mullo_epi16(a, b);
}

[[nodiscard]] inline VecI16 mulhi_i16(VecI16 a, VecI16 b) {
    return _mm512_mulhi_epi16(a, b);
}

[[nodiscard]] inline VecI32 madd_i16(VecI16 a, VecI16 b) {
    return _mm512_madd_epi16(a, b);
}

[[nodiscard]] inline VecI16 clamp_i16(
    VecI16 reg,
    VecI16 mins,
    VecI16 maxs
) {
    return _mm512_min_epi16(
        _mm512_max_epi16(reg, mins),
        maxs
    );
}

[[nodiscard]] inline VecI32 clamp_i32(
    VecI32 reg,
    VecI32 mins,
    VecI32 maxs
) {
    return _mm512_min_epi32(
        _mm512_max_epi32(reg, mins),
        maxs
    );
}

[[nodiscard]] inline VecI16 lshift_i16(
    VecI16 reg,
    i32 shift
) {
    return _mm512_slli_epi16(reg, shift);
}

[[nodiscard]] inline VecI32 rshift_i32(
    VecI32 reg,
    i32 shift
) {
    return _mm512_srai_epi32(reg, shift);
}

[[nodiscard]] inline VecI16 low_i8_i16(VecI8 reg) {
    return _mm512_cvtepi8_epi16(
        _mm512_castsi512_si256(reg)
    );
}

[[nodiscard]] inline VecI16 high_i8_i16(VecI8 reg) {
    return _mm512_cvtepi8_epi16(
        _mm512_extracti64x4_epi64(reg, 1)
    );
}

[[nodiscard]] inline VecU8 pack_u8_i16(
    VecI16 a,
    VecI16 b
) {
    return _mm512_packus_epi16(a, b);
}

[[nodiscard]] inline VecU8 tile_u8(const u8* vals) {
    return _mm512_set1_epi32(
        *reinterpret_cast<const u32*>(vals)
    );
}

[[nodiscard]] inline u32 nonzero_mask(VecU8 reg) {
    return _mm512_cmpgt_epi32_mask(
        reg,
        zero_i32()
    );
}

[[nodiscard]] inline VecI32 dpbusd_i32(
    VecI32 a,
    VecU8 b,
    VecI8 c
) {
    return _mm512_dpbusd_epi32(a, b, c);
}

[[nodiscard]] inline VecI32 fmadd_i32(
    VecI32 a,
    VecI32 b,
    VecI32 c
) {
    return add_i32(
        mullo_i32(a, b),
        c
    );
}

[[nodiscard]] inline i32 hadd_i32(VecI32 reg) {
    __m256i lo256 =
        _mm512_castsi512_si256(reg);

    __m256i hi256 =
        _mm512_extracti64x4_epi64(reg, 1);

    __m256i sum256 =
        _mm256_add_epi32(lo256, hi256);

    __m128i lo128 =
        _mm256_castsi256_si128(sum256);

    __m128i hi128 =
        _mm256_extracti128_si256(sum256, 1);

    __m128i sum128 =
        _mm_add_epi32(lo128, hi128);

    __m128i hi64 =
        _mm_unpackhi_epi64(sum128, sum128);

    __m128i sum64 =
        _mm_add_epi32(hi64, sum128);

    __m128i hi32 =
        _mm_shuffle_epi32(
            sum64,
            _MM_SHUFFLE(2, 3, 0, 1)
        );

    __m128i sum32 =
        _mm_add_epi32(sum64, hi32);

    return _mm_cvtsi128_si32(sum32);
}

#elif defined(__AVX__) || defined(__AVX2__)

#include <immintrin.h>

#define USE_AVX2
#define USE_SIMD 256
#define ALIGNMENT 32
#define SIMD_REGS 16

using VecU8 = __m256i;
using VecI8 = __m256i;
using VecI16 = __m256i;
using VecI32 = __m256i;

[[nodiscard]] inline VecU8 load_u8(const u8* src) {
    return _mm256_load_si256(
        reinterpret_cast<const VecU8*>(src)
    );
}

[[nodiscard]] inline VecI8 load_i8(const i8* src) {
    return _mm256_load_si256(
        reinterpret_cast<const VecI8*>(src)
    );
}

[[nodiscard]] inline VecI16 load_i16(const i16* src) {
    return _mm256_load_si256(
        reinterpret_cast<const VecI16*>(src)
    );
}

[[nodiscard]] inline VecI32 load_i32(const i32* src) {
    return _mm256_load_si256(
        reinterpret_cast<const VecI32*>(src)
    );
}

inline void store_u8(u8* dst, VecU8 src) {
    _mm256_store_si256(
        reinterpret_cast<VecU8*>(dst),
        src
    );
}

inline void store_i16(i16* dst, VecI16 src) {
    _mm256_store_si256(
        reinterpret_cast<VecI16*>(dst),
        src
    );
}

inline void store_i32(i32* dst, VecI32 src) {
    _mm256_store_si256(
        reinterpret_cast<VecI32*>(dst),
        src
    );
}

[[nodiscard]] inline VecI16 zero_i16() {
    return _mm256_setzero_si256();
}

[[nodiscard]] inline VecI32 zero_i32() {
    return _mm256_setzero_si256();
}

[[nodiscard]] inline VecI16 full_i16(i16 val) {
    return _mm256_set1_epi16(val);
}

[[nodiscard]] inline VecI32 full_i32(i32 val) {
    return _mm256_set1_epi32(val);
}

[[nodiscard]] inline VecI16 add_i16(VecI16 a, VecI16 b) {
    return _mm256_add_epi16(a, b);
}

[[nodiscard]] inline VecI32 add_i32(VecI32 a, VecI32 b) {
    return _mm256_add_epi32(a, b);
}

[[nodiscard]] inline VecI16 sub_i16(VecI16 a, VecI16 b) {
    return _mm256_sub_epi16(a, b);
}

[[nodiscard]] inline VecI32 mullo_i32(VecI32 a, VecI32 b) {
    return _mm256_mullo_epi32(a, b);
}

[[nodiscard]] inline VecI16 mullo_i16(VecI16 a, VecI16 b) {
    return _mm256_mullo_epi16(a, b);
}

[[nodiscard]] inline VecI16 mulhi_i16(VecI16 a, VecI16 b) {
    return _mm256_mulhi_epi16(a, b);
}

[[nodiscard]] inline VecI32 madd_i16(VecI16 a, VecI16 b) {
    return _mm256_madd_epi16(a, b);
}

[[nodiscard]] inline VecI16 clamp_i16(
    VecI16 reg,
    VecI16 mins,
    VecI16 maxs
) {
    return _mm256_min_epi16(
        _mm256_max_epi16(reg, mins),
        maxs
    );
}

[[nodiscard]] inline VecI32 clamp_i32(
    VecI32 reg,
    VecI32 mins,
    VecI32 maxs
) {
    return _mm256_min_epi32(
        _mm256_max_epi32(reg, mins),
        maxs
    );
}

[[nodiscard]] inline VecI16 lshift_i16(
    VecI16 reg,
    i32 shift
) {
    return _mm256_slli_epi16(reg, shift);
}

[[nodiscard]] inline VecI32 rshift_i32(
    VecI32 reg,
    i32 shift
) {
    return _mm256_srai_epi32(reg, shift);
}

[[nodiscard]] inline VecI16 low_i8_i16(VecI8 reg) {
    return _mm256_cvtepi8_epi16(
        _mm256_castsi256_si128(reg)
    );
}

[[nodiscard]] inline VecI16 high_i8_i16(VecI8 reg) {
    return _mm256_cvtepi8_epi16(
        _mm256_extracti128_si256(reg, 1)
    );
}

[[nodiscard]] inline VecU8 pack_u8_i16(
    VecI16 a,
    VecI16 b
) {
    return _mm256_packus_epi16(a, b);
}

[[nodiscard]] inline VecU8 tile_u8(const u8* vals) {
    return _mm256_set1_epi32(
        *reinterpret_cast<const u32*>(vals)
    );
}

[[nodiscard]] inline u32 nonzero_mask(VecU8 reg) {
    return _mm256_movemask_ps(
        _mm256_castsi256_ps(
            _mm256_cmpgt_epi32(
                reg,
                zero_i32()
            )
        )
    );
}

[[nodiscard]] inline VecI32 dpbusd_i32(
    VecI32 a,
    VecU8 b,
    VecI8 c
) {
    return add_i32(
        a,
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(b, c),
            full_i16(1)
        )
    );
}

[[nodiscard]] inline VecI32 fmadd_i32(
    VecI32 a,
    VecI32 b,
    VecI32 c
) {
    return add_i32(
        mullo_i32(a, b),
        c
    );
}

[[nodiscard]] inline i32 hadd_i32(VecI32 reg) {
    __m128i lo =
        _mm256_castsi256_si128(reg);

    __m128i hi =
        _mm256_extracti128_si256(reg, 1);

    __m128i sum =
        _mm_add_epi32(lo, hi);

    sum =
        _mm_add_epi32(
            sum,
            _mm_shuffle_epi32(
                sum,
                _MM_SHUFFLE(2, 3, 0, 1)
            )
        );

    sum =
        _mm_add_epi32(
            sum,
            _mm_shuffle_epi32(
                sum,
                _MM_SHUFFLE(1, 0, 3, 2)
            )
        );

    return _mm_cvtsi128_si32(sum);
}

#else

#define USE_SIMD 0

#endif

#endif
