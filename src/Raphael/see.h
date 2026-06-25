#pragma once
#include <chess/include.h>



namespace raphael::see {
/** Simulates exchanges on the move destination square to evaluate if it's winning or losing
 *
 * \param move the move to evaluate
 * \param board current board
 * \param threshold minimum evaluation to count as good
 * \returns whether the move's see score is greater than or equal to the threshold
 */
bool gte(chess::Move move, const chess::Board& board, i32 threshold);
}  // namespace raphael::see
