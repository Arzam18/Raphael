#pragma once
#include <chess/include.h>



namespace raphael::wdl {
struct WDL {
    i32 win;
    i32 draw;
    i32 loss;
};

/** Returns the current estimated win, draw, loss rate
 *
 * \param score unnormalized score
 * \param board current board
 * \returns the predicted wdl rates
 */
[[nodiscard]] WDL get_wdl(i32 score, const chess::Board& board);

/** Normalizes the score so 1 pawn = 0.5 win rate
 *
 * \param score unnormalized score
 * \param board current board
 * \returns the normalized score
 */
[[nodiscard]] i32 normalize_score(i32 score, const chess::Board& board);
}  // namespace raphael::wdl
