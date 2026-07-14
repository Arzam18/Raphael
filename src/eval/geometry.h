#pragma once
#ifdef EVAL_NNUE
#include <chess/include.h>
#include <eval/simd.h>

#include <span>
#include <utility>



#ifdef USE_SIMD
namespace raphael::nnue::geometry {
using Bit = u8;       // piece type
using BitRays = u64;  // a bitboard-like structure encoding ray directions and steps


// byteboard mailbox representation encoding which pieces are on which square, using a byte per sq
struct Vector {
#ifdef USE_AVX512
    VecU8 raw;
#elif defined(USE_AVX2)
    VecU8 raw[2];
#endif

    /** Loads a Vector
     *
     * \param src data source, 8*64 bits
     * \returns the loaded Vector
     */
    [[nodiscard]] static Vector load(const void* src);

    /** Stores a Vector
     *
     * \param dst data destination, 8*64 bits
     */
    void store_into(void* dst) const;

    /** Rotates the rays by a 180 degrees
     *
     * \returns the rotated vector
     */
    [[nodiscard]] Vector flip() const;

#if defined(USE_AVX2)
    /** Returns a mask of locations where _mm256_cmpeq_epi8 returns true for each byte
     *
     * \returns the mask of set bytes
     */
    [[nodiscard]] BitRays to_mask() const;
#endif
};


// square indices for each ray bit + mask of in/out-of-bound squares
struct Permutation {
#ifdef USE_AVX512
    Vector indices;
    BitRays valid;
#elif defined(USE_AVX2)
    Vector indices;
    Vector invalid;
#endif
};


/** Fills the whole ray if any bit along that direction (except the knight ray) is set
 *
 * \param the bitray to fill
 * \returns filled bitray
 */
[[nodiscard]] BitRays ray_fill(BitRays br);

/** Returns closest pieces the given piece can attack along each direction
 *
 * \param piece attacking piece
 * \param closest closest occupied squares along each direction
 * \returns closest victims
 */
[[nodiscard]] BitRays outgoing_threats(chess::Piece piece, BitRays closest);

/** Returns closest pieces that are attacking us for each direction
 *
 * \param bits mailbox byteboard
 * \param closest closest occupied squares along each direction
 * \returns closest attackers
 */
[[nodiscard]] BitRays incoming_attackers(const Vector& bits, BitRays closest);

/** Return closest sliders that are attacking us for each direction
 *
 * \param bits mailbox byteboard
 * \param closest closest occupied squares along each direction
 * \returns closest attacking sliders
 */
[[nodiscard]] BitRays incoming_sliders(const Vector& bits, BitRays closest);

/** Compute nearest occupied squares along each ray direction
 *
 * \param bits mailbox byteboard
 * \returns closest occupied squares along each direction
 */
[[nodiscard]] BitRays closest_occupied(const Vector& bits);

/** Returns the permutation for rays from a focus square
 *
 * \param focus square to generate rays from
 * \returns the corresponding permutation
 */
[[nodiscard]] Permutation permutation_for(chess::Square focus);

/** Permutes a mailbox into ray space
 *
 * \param perm the permutation map for the focus square
 * \param masked_mailbox mailbox byteboard
 * \returns the piece types and piece bits permuted into ray space
 */
[[nodiscard]] std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm, const Vector& masked_mailbox
);

/** Permutes a mailbox into ray space
 *
 * \param perm the permutation map for the focus square
 * \param mailbox board mailbox
 * \returns the piece types and piece bits permuted into ray space
 */
[[nodiscard]] std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm, std::span<const chess::Piece, 64> mailbox
);

/** Permutes a mailbox into ray space but treats a square as empty
 *
 * \param perm the permutation map for the focus square
 * \param mailbox board mailbox
 * \param ignore square to ignore
 * \returns the piece types and piece bits permuted into ray space
 */
[[nodiscard]] std::pair<Vector, Vector> permute_mailbox(
    const Permutation& perm, std::span<const chess::Piece, 64> mailbox, chess::Square ignore
);
}  // namespace raphael::nnue::geometry
#endif
#endif
