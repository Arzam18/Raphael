#ifdef EVAL_NNUE
#include <algorithm>
#include <eval/geometry.h>

using std::array;
using std::pair;



#ifdef USE_SIMD
namespace raphael::nnue::geometry {
namespace internal {
struct Bits {
    static constexpr Bit WHITEPAWN = 1 << 0;
    static constexpr Bit BLACKPAWN = 1 << 1;
    static constexpr Bit KNIGHT = 1 << 2;
    static constexpr Bit BISHOP = 1 << 3;
    static constexpr Bit ROOK = 1 << 4;
    static constexpr Bit QUEEN = 1 << 5;
    static constexpr Bit KING = 1 << 6;
};

// generate lut of permutations to go from mailbox to a ray vector
static constexpr MultiArray<u8, 64, 64> PERMUTATIONS = [] {
    // https://87flowers.com/byteboard-attack-tables-1
    constexpr array<u8, 64> OFFSETS = {
        0x1F, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70,
        0x21, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x12, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0xF2, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97,
        0xE1, 0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90,
        0xDF, 0xEF, 0xDE, 0xCD, 0xBC, 0xAB, 0x9A, 0x89,
        0xEE, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9,
        0x0E, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69,
    };

    MultiArray<u8, 64, 64> perm{};

    for (u8 focus = 0; focus < 64; focus++) {
        for (u8 i = 0; i < 64; i++) {
            const u8 wide_focus = focus + (focus & 0x38);
            const u8 wide_result = wide_focus + OFFSETS[i];
            const u8 result =
                ((wide_result & 0x70) >> 1) | (wide_result & 0x07);
            const bool is_valid = (wide_result & 0x88) == 0;

            perm[focus][i] = (is_valid) ? result : 0x80;
        }
    }

    return perm;
}();

static constexpr array<Bit, 16> PIECE_TO_BIT = [] {
    array<Bit, 16> lut{};

    lut[chess::Piece::WHITEPAWN] = Bits::WHITEPAWN;
    lut[chess::Piece::BLACKPAWN] = Bits::BLACKPAWN;
    lut[chess::Piece::WHITEKNIGHT] = Bits::KNIGHT;
    lut[chess::Piece::BLACKKNIGHT] = Bits::KNIGHT;
    lut[chess::Piece::WHITEBISHOP] = Bits::BISHOP;
    lut[chess::Piece::BLACKBISHOP] = Bits::BISHOP;
    lut[chess::Piece::WHITEROOK] = Bits::ROOK;
    lut[chess::Piece::BLACKROOK] = Bits::ROOK;
    lut[chess::Piece::WHITEQUEEN] = Bits::QUEEN;
    lut[chess::Piece::BLACKQUEEN] = Bits::QUEEN;
    lut[chess::Piece::WHITEKING] = Bits::KING;
    lut[chess::Piece::BLACKKING] = Bits::KING;
    lut[chess::Piece::NONE] = 0;

    return lut;
}();

static constexpr array<BitRays, 12> OUTGOING_THREATS = [] {
    array<BitRays, 12> lut{};

    lut[chess::Piece::WHITEPAWN] = 0x02'00'00'00'00'00'02'00;
    lut[chess::Piece::BLACKPAWN] = 0x00'00'02'00'02'00'00'00;
    lut[chess::Piece::WHITEKNIGHT] = 0x01'01'01'01'01'01'01'01;
    lut[chess::Piece::BLACKKNIGHT] = 0x01'01'01'01'01'01'01'01;
    lut[chess::Piece::WHITEBISHOP] = 0xFE'00'FE'00'FE'00'FE'00;
    lut[chess::Piece::BLACKBISHOP] = 0xFE'00'FE'00'FE'00'FE'00;
    lut[chess::Piece::WHITEROOK] = 0x00'FE'00'FE'00'FE'00'FE;
    lut[chess::Piece::BLACKROOK] = 0x00'FE'00'FE'00'FE'00'FE;
    lut[chess::Piece::WHITEQUEEN] = 0xFE'FE'FE'FE'FE'FE'FE'FE'FE;
    lut[chess::Piece::BLACKQUEEN] = 0xFE'FE'FE'FE'FE'FE'FE'FE'FE;
    lut[chess::Piece::WHITEKING] = 0;
    lut[chess::Piece::BLACKKING] = 0;

    return lut;
}();

static constexpr array<Bit, 64> INCOMING_THREATS_MASK = [] {
    constexpr Bit HORS = Bits::KNIGHT;
    constexpr Bit ORTH = Bits::ROOK | Bits::QUEEN;
    constexpr Bit DIAG = Bits::BISHOP | Bits::QUEEN;
    constexpr Bit ORNR = ORTH;
    constexpr Bit WPNR = Bits::WHITEPAWN | DIAG;
    constexpr Bit BPNR = Bits::BLACKPAWN | DIAG;

    return array<Bit, 64>{
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        HORS, BPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        HORS, WPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        HORS, WPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
        HORS, ORNR, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        HORS, BPNR, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
    };
}();

static constexpr array<Bit, 64> INCOMING_SLIDER_MASK = [] {
    constexpr Bit ORTH = Bits::ROOK | Bits::QUEEN;
    constexpr Bit DIAG = Bits::BISHOP | Bits::QUEEN;
    constexpr Bit NONE = 0x80;

    return array<Bit, 64>{
        NONE, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        NONE, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
        NONE, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        NONE, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
        NONE, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        NONE, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
        NONE, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH, ORTH,
        NONE, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG, DIAG,
    };
}();
}  // namespace internal



#ifdef USE_AVX512

Vector Vector::load(const void* src) {
    return {_mm512_loadu_si512(src)};
}

void Vector::store_into(void* dst) const {
    _mm512_store_si512(dst, raw);
}

Vector Vector::flip() const {
    return {_mm512_shuffle_i64x2(raw, raw, 0b01001110)};
}

#elif defined(USE_NEON)

Vector Vector::load(const void* src) {
    const auto* ptr = static_cast<const u8*>(src);

    Vector result{};

    result.raw[0] = load_u8(ptr + 0);
    result.raw[1] = load_u8(ptr + 16);
    result.raw[2] = load_u8(ptr + 32);
    result.raw[3] = load_u8(ptr + 48);

    return result;
}

void Vector::store_into(void* dst) const {
    auto* ptr = static_cast<u8*>(dst);

    store_u8(ptr + 0, raw[0]);
    store_u8(ptr + 16, raw[1]);
    store_u8(ptr + 32, raw[2]);
    store_u8(ptr + 48, raw[3]);
}

Vector Vector::flip() const {
    Vector result{};

    result.raw[0] = raw[2];
    result.raw[1] = raw[3];
    result.raw[2] = raw[0];
    result.raw[3] = raw[1];

    return result;
}

BitRays Vector::to_mask() const {
    BitRays result = 0;
    alignas(16) u8 bytes[16];

    for (i32 r = 0; r < 4; ++r) {
        const auto top_bits = vshrq_n_u8(
            vreinterpretq_u8_s8(raw[r].raw),
            7
        );

        vst1q_u8(bytes, top_bits);

        for (i32 i = 0; i < 16; ++i)
            result |= static_cast<BitRays>(bytes[i] & 1) << (16 * r + i);
    }

    return result;
}

#elif defined(USE_AVX2)

Vector Vector::load(const void* src) {
    return {
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 0),
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src) + 1),
    };
}

void Vector::store_into(void* dst) const {
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst) + 0, raw[0]);
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst) + 1, raw[1]);
}

Vector Vector::flip() const {
    return {raw[1], raw[0]};
}

BitRays Vector::to_mask() const {
    return (static_cast<BitRays>(_mm256_movemask_epi8(raw[1])) << 32)
           | static_cast<u32>(_mm256_movemask_epi8(raw[0]));
}

#endif


BitRays ray_fill(BitRays br) {
    br = (br + 0x7E'7E'7E'7E'7E'7E'7E'7E)
         & 0x80'80'80'80'80'80'80'80;

    return br - (br >> 7);
}

BitRays outgoing_threats(chess::Piece piece, BitRays closest) {
    return internal::OUTGOING_THREATS[piece] & closest;
}


#ifdef USE_AVX512

BitRays incoming_attackers(const Vector& bits, BitRays closest) {
    const auto mask = Vector::load(internal::INCOMING_THREATS_MASK.data());

    return _mm512_test_epi8_mask(bits.raw, mask.raw) & closest;
}

BitRays incoming_sliders(const Vector& bits, BitRays closest) {
    const auto mask = Vector::load(internal::INCOMING_SLIDER_MASK.data());

    return _mm512_test_epi8_mask(bits.raw, mask.raw)
           & closest
           & 0xFE'FE'FE'FE'FE'FE'FE'FE;
}

BitRays closest_occupied(const Vector& bits) {
    const BitRays occupied = _mm512_test_epi8_mask(bits.raw, bits.raw);
    const BitRays o = occupied | 0x81'81'81'81'81'81'81'81;

    return (o ^ (o - 0x03'03'03'03'03'03'03'03)) & occupied;
}

Permutation permutation_for(chess::Square focus) {
    const auto indices = Vector::load(
        internal::PERMUTATIONS[focus].data()
    );

    const auto valid = _mm512_testn_epi8_mask(
        indices.raw,
        _mm512_set1_epi8(0x80)
    );

    return {indices, valid};
}

pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    const Vector& masked_mailbox
) {
    const auto lut = _mm512_broadcast_i32x4(
        _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(
                internal::PIECE_TO_BIT.data()
            )
        )
    );

    const Vector permuted{
        _mm512_permutexvar_epi8(
            perm.indices.raw,
            masked_mailbox.raw
        )
    };

    const Vector bits{
        _mm512_maskz_shuffle_epi8(
            perm.valid,
            lut,
            permuted.raw
        )
    };

    return {permuted, bits};
}

std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    std::span<const chess::Piece, 64> mailbox
) {
    return permute_mailbox(
        perm,
        Vector::load(mailbox.data())
    );
}

std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    std::span<const chess::Piece, 64> mailbox,
    chess::Square ignore
) {
    const Vector masked_mailbox{
        _mm512_mask_blend_epi8(
            static_cast<u64>(
                chess::BitBoard::from_square(ignore)
            ),
            Vector::load(mailbox.data()).raw,
            _mm512_set1_epi8(chess::Piece::NONE)
        )
    };

    return permute_mailbox(perm, masked_mailbox);
}

#elif defined(USE_NEON)

BitRays incoming_attackers(
    const Vector& bits,
    BitRays closest
) {
    const auto mask = Vector::load(
        internal::INCOMING_THREATS_MASK.data()
    );

    Vector v{};

    v.raw[0].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[0].raw),
            vreinterpretq_u8_s8(mask.raw[0].raw)
        ),
        vdupq_n_u8(0)
    ));

    v.raw[1].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[1].raw),
            vreinterpretq_u8_s8(mask.raw[1].raw)
        ),
        vdupq_n_u8(0)
    ));

    v.raw[2].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[2].raw),
            vreinterpretq_u8_s8(mask.raw[2].raw)
        ),
        vdupq_n_u8(0)
    ));

    v.raw[3].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[3].raw),
            vreinterpretq_u8_s8(mask.raw[3].raw)
        ),
        vdupq_n_u8(0)
    ));

    return v.to_mask() & closest;
}

BitRays incoming_sliders(
    const Vector& bits,
    BitRays closest
) {
    const auto mask = Vector::load(
        internal::INCOMING_SLIDER_MASK.data()
    );

    Vector v{};

    v.raw[0].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[0].raw),
            vreinterpretq_u8_s8(mask.raw[0].raw)
        ),
        vdupq_n_u8(0)
    ));

    v.raw[1].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[1].raw),
            vreinterpretq_u8_s8(mask.raw[1].raw)
        ),
        vdupq_n_u8(0)
    ));

    v.raw[2].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[2].raw),
            vdupq_n_u8(0)
        ),
        vdupq_n_u8(0)
    ));

    v.raw[3].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vandq_u8(
            vreinterpretq_u8_s8(bits.raw[3].raw),
            vreinterpretq_u8_s8(mask.raw[3].raw)
        ),
        vdupq_n_u8(0)
    ));

    return v.to_mask()
           & closest
           & 0xFE'FE'FE'FE'FE'FE'FE'FE;
}

BitRays closest_occupied(const Vector& bits) {
    Vector v{};

    v.raw[0].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vreinterpretq_u8_s8(bits.raw[0].raw),
        vdupq_n_u8(0)
    ));

    v.raw[1].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vreinterpretq_u8_s8(bits.raw[1].raw),
        vdupq_n_u8(0)
    ));

    v.raw[2].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vreinterpretq_u8_s8(bits.raw[2].raw),
        vdupq_n_u8(0)
    ));

    v.raw[3].raw = vreinterpretq_s8_u8(vcgtq_u8(
        vreinterpretq_u8_s8(bits.raw[3].raw),
        vdupq_n_u8(0)
    ));

    const BitRays occupied = v.to_mask();
    const BitRays o =
        occupied | 0x81'81'81'81'81'81'81'81;

    return (o ^ (o - 0x03'03'03'03'03'03'03'03))
           & occupied;
}

Permutation permutation_for(chess::Square focus) {
    const auto indices = Vector::load(
        internal::PERMUTATIONS[focus].data()
    );

    const auto bad = vdupq_n_u8(0x80);

    Vector invalid{};

    invalid.raw[0].raw = vreinterpretq_s8_u8(vceqq_u8(
        vreinterpretq_u8_s8(indices.raw[0].raw),
        bad
    ));

    invalid.raw[1].raw = vreinterpretq_s8_u8(vceqq_u8(
        vreinterpretq_u8_s8(indices.raw[1].raw),
        bad
    ));

    invalid.raw[2].raw = vreinterpretq_s8_u8(vceqq_u8(
        vreinterpretq_u8_s8(indices.raw[2].raw),
        bad
    ));

    invalid.raw[3].raw = vreinterpretq_s8_u8(vceqq_u8(
        vreinterpretq_u8_s8(indices.raw[3].raw),
        bad
    ));

    return {indices, invalid};
}

pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    const Vector& masked_mailbox
) {
    const auto lut =
        vld1q_u8(internal::PIECE_TO_BIT.data());

    const uint8x16x4_t table = {{
        vreinterpretq_u8_s8(masked_mailbox.raw[0].raw),
        vreinterpretq_u8_s8(masked_mailbox.raw[1].raw),
        vreinterpretq_u8_s8(masked_mailbox.raw[2].raw),
        vreinterpretq_u8_s8(masked_mailbox.raw[3].raw),
    }};

    Vector permuted{};

    permuted.raw[0].raw = vreinterpretq_s8_u8(
        vqtbl4q_u8(
            table,
            vreinterpretq_u8_s8(
                perm.indices.raw[0].raw
            )
        )
    );

    permuted.raw[1].raw = vreinterpretq_s8_u8(
        vqtbl4q_u8(
            table,
            vreinterpretq_u8_s8(
                perm.indices.raw[1].raw
            )
        )
    );

    permuted.raw[2].raw = vreinterpretq_s8_u8(
        vqtbl4q_u8(
            table,
            vreinterpretq_u8_s8(
                perm.indices.raw[2].raw
            )
        )
    );

    permuted.raw[3].raw = vreinterpretq_s8_u8(
        vqtbl4q_u8(
            table,
            vreinterpretq_u8_s8(
                perm.indices.raw[3].raw
            )
        )
    );

    Vector bits{};

    bits.raw[0].raw = vreinterpretq_s8_u8(
        vbicq_u8(
            vreinterpretq_u8_s8(
                vqtbl1q_u8(
                    lut,
                    vreinterpretq_u8_s8(
                        permuted.raw[0].raw
                    )
                )
            ),
            vreinterpretq_u8_s8(
                perm.invalid.raw[0].raw
            )
        )
    );

    bits.raw[1].raw = vreinterpretq_s8_u8(
        vbicq_u8(
            vreinterpretq_u8_s8(
                vqtbl1q_u8(
                    lut,
                    vreinterpretq_u8_s8(
                        permuted.raw[1].raw
                    )
                )
            ),
            vreinterpretq_u8_s8(
                perm.invalid.raw[1].raw
            )
        )
    );

    bits.raw[2].raw = vreinterpretq_s8_u8(
        vbicq_u8(
            vreinterpretq_u8_s8(
                vqtbl1q_u8(
                    lut,
                    vreinterpretq_u8_s8(
                        permuted.raw[2].raw
                    )
                )
            ),
            vreinterpretq_u8_s8(
                perm.invalid.raw[2].raw
            )
        )
    );

    bits.raw[3].raw = vreinterpretq_s8_u8(
        vbicq_u8(
            vreinterpretq_u8_s8(
                vqtbl1q_u8(
                    lut,
                    vreinterpretq_u8_s8(
                        permuted.raw[3].raw
                    )
                )
            ),
            vreinterpretq_u8_s8(
                perm.invalid.raw[3].raw
            )
        )
    );

    return {permuted, bits};
}

std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    std::span<const chess::Piece, 64> mailbox
) {
    return permute_mailbox(
        perm,
        Vector::load(mailbox.data())
    );
}

std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    std::span<const chess::Piece, 64> mailbox,
    chess::Square ignore
) {
    array<chess::Piece, 64> copy{};

    std::copy(
        mailbox.begin(),
        mailbox.end(),
        copy.begin()
    );

    copy[static_cast<usize>(ignore)] =
        chess::Piece::NONE;

    return permute_mailbox(
        perm,
        Vector::load(copy.data())
    );
}

#elif defined(USE_AVX2)

BitRays incoming_attackers(
    const Vector& bits,
    BitRays closest
) {
    const auto mask = Vector::load(
        internal::INCOMING_THREATS_MASK.data()
    );

    const Vector v{
        {
            _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    bits.raw[0],
                    mask.raw[0]
                ),
                _mm256_setzero_si256()
            ),

            _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    bits.raw[1],
                    mask.raw[1]
                ),
                _mm256_setzero_si256()
            ),
        }
    };

    return ~v.to_mask() & closest;
}

BitRays incoming_sliders(
    const Vector& bits,
    BitRays closest
) {
    const auto mask = Vector::load(
        internal::INCOMING_SLIDER_MASK.data()
    );

    const Vector v{
        {
            _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    bits.raw[0],
                    mask.raw[0]
                ),
                _mm256_setzero_si256()
            ),

            _mm256_cmpeq_epi8(
                _mm256_and_si256(
                    bits.raw[1],
                    mask.raw[1]
                ),
                _mm256_setzero_si256()
            ),
        }
    };

    return ~v.to_mask()
           & closest
           & 0xFE'FE'FE'FE'FE'FE'FE'FE;
}

BitRays closest_occupied(const Vector& bits) {
    const Vector unoccupied{
        _mm256_cmpeq_epi8(
            bits.raw[0],
            _mm256_setzero_si256()
        ),

        _mm256_cmpeq_epi8(
            bits.raw[1],
            _mm256_setzero_si256()
        ),
    };

    const BitRays occupied = ~unoccupied.to_mask();

    const BitRays o =
        occupied | 0x81'81'81'81'81'81'81'81;

    return (o ^ (o - 0x03'03'03'03'03'03'03'03))
           & occupied;
}

Permutation permutation_for(chess::Square focus) {
    const auto indices = Vector::load(
        internal::PERMUTATIONS[focus].data()
    );

    const Vector invalid{
        {
            _mm256_cmpeq_epi8(
                indices.raw[0],
                _mm256_set1_epi8(0x80)
            ),

            _mm256_cmpeq_epi8(
                indices.raw[1],
                _mm256_set1_epi8(0x80)
            ),
        }
    };

    return {indices, invalid};
}

pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    const Vector& masked_mailbox
) {
    const auto lut = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(
                internal::PIECE_TO_BIT.data()
            )
        )
    );

    const auto half_swizzler =
        [](__m256i bytes0, __m256i bytes1, __m256i idxs) {
            const auto mask0 =
                _mm256_slli_epi64(idxs, 2);

            const auto mask1 =
                _mm256_slli_epi64(idxs, 3);

            const auto lolo0 =
                _mm256_shuffle_epi8(
                    _mm256_permute2x128_si256(
                        bytes0, bytes0, 0x00
                    ),
                    idxs
                );

            const auto hihi0 =
                _mm256_shuffle_epi8(
                    _mm256_permute2x128_si256(
                        bytes0, bytes0, 0x11
                    ),
                    idxs
                );

            const auto x =
                _mm256_blendv_epi8(
                    lolo0,
                    hihi0,
                    mask1
                );

            const auto lolo1 =
                _mm256_shuffle_epi8(
                    _mm256_permute2x128_si256(
                        bytes1, bytes1, 0x00
                    ),
                    idxs
                );

            const auto hihi1 =
                _mm256_shuffle_epi8(
                    _mm256_permute2x128_si256(
                        bytes1, bytes1, 0x11
                    ),
                    idxs
                );

            const auto y =
                _mm256_blendv_epi8(
                    lolo1,
                    hihi1,
                    mask1
                );

            return _mm256_blendv_epi8(
                x,
                y,
                mask0
            );
        };

    const Vector permuted{
        {
            half_swizzler(
                masked_mailbox.raw[0],
                masked_mailbox.raw[1],
                perm.indices.raw[0]
            ),

            half_swizzler(
                masked_mailbox.raw[0],
                masked_mailbox.raw[1],
                perm.indices.raw[1]
            ),
        }
    };

    const Vector bits{
        {
            _mm256_andnot_si256(
                perm.invalid.raw[0],
                _mm256_shuffle_epi8(
                    lut,
                    permuted.raw[0]
                )
            ),

            _mm256_andnot_si256(
                perm.invalid.raw[1],
                _mm256_shuffle_epi8(
                    lut,
                    permuted.raw[1]
                )
            ),
        }
    };

    return {permuted, bits};
}

std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    std::span<const chess::Piece, 64> mailbox
) {
    return permute_mailbox(
        perm,
        Vector::load(mailbox.data())
    );
}

std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm,
    std::span<const chess::Piece, 64> mailbox,
    chess::Square ignore
) {
    constexpr array<u8, 64> IOTA = {
        0,  1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63
    };

    const auto iota = Vector::load(&IOTA);

    const auto ignore_vec =
        _mm256_set1_epi8(static_cast<i8>(ignore));

    const auto non_vec =
        _mm256_set1_epi8(
            static_cast<i8>(chess::Piece::NONE)
        );

    const auto mb = Vector::load(mailbox.data());

    const Vector masked_mailbox{
        _mm256_blendv_epi8(
            mb.raw[0],
            non_vec,
            _mm256_cmpeq_epi8(
                iota.raw[0],
                ignore_vec
            )
        ),

        _mm256_blendv_epi8(
            mb.raw[1],
            non_vec,
            _mm256_cmpeq_epi8(
                iota.raw[1],
                ignore_vec
            )
        ),
    };

    return permute_mailbox(
        perm,
        masked_mailbox
    );
}

#endif
}  // namespace raphael::nnue::geometry
#endif
#endif
