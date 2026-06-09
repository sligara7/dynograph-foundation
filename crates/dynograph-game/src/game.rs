//! The normal-form game abstraction the analysis operates on.
//!
//! An N-player normal-form game is a list of strategy counts (one per player)
//! plus a payoff for every player in every **strategy profile** (one strategy
//! chosen per player). Profiles are interned to a dense **cell index** in
//! mixed-radix (row-major) order so payoffs live in flat `Vec`s; the caller
//! supplies cells keyed by their explicit profile and the builder validates
//! that every profile is present exactly once.

use crate::error::GameError;

/// An immutable, validated N-player normal-form game.
#[derive(Debug, Clone, PartialEq)]
pub struct Game {
    /// Number of pure strategies available to each player (length = players).
    strategy_counts: Vec<usize>,
    /// `payoffs[cell][player]` — payoff to `player` in the profile that
    /// mixed-radix-decodes to `cell`. Length = product of `strategy_counts`;
    /// each inner length = number of players.
    payoffs: Vec<Vec<f64>>,
}

impl Game {
    /// Build a game from its strategy counts and an explicit list of
    /// `(profile, utilities)` cells. Fails loudly (see [`GameError`]) unless:
    /// every player has ≥1 strategy; each profile names one in-range strategy
    /// per player; each utilities vector has one finite payoff per player; and
    /// the cells cover every profile **exactly once** (no gaps, no duplicates).
    pub fn from_cells(
        strategy_counts: Vec<usize>,
        cells: Vec<(Vec<usize>, Vec<f64>)>,
    ) -> Result<Game, GameError> {
        let players = strategy_counts.len();
        if players == 0 || strategy_counts.contains(&0) {
            return Err(GameError::Empty);
        }
        let total = strategy_counts.iter().product::<usize>();

        // Place each cell at its dense index; reject out-of-range profiles,
        // wrong-shaped utilities, non-finite payoffs, and duplicates.
        let mut payoffs: Vec<Option<Vec<f64>>> = vec![None; total];
        let mut filled = 0usize;
        for (profile, utilities) in cells {
            if profile.len() != players {
                return Err(GameError::InvalidProfile);
            }
            if utilities.len() != players {
                return Err(GameError::InvalidUtilities);
            }
            if utilities.iter().any(|u| !u.is_finite()) {
                return Err(GameError::NonFiniteUtility);
            }
            let cell = cell_index(&strategy_counts, &profile).ok_or(GameError::InvalidProfile)?;
            if payoffs[cell].is_some() {
                return Err(GameError::IncompleteGame(format!(
                    "duplicate cell for profile {profile:?}"
                )));
            }
            payoffs[cell] = Some(utilities);
            filled += 1;
        }
        if filled != total {
            return Err(GameError::IncompleteGame(format!(
                "got {filled} cells but the game has {total} strategy profiles"
            )));
        }

        Ok(Game {
            strategy_counts,
            // Every slot is `Some` (filled == total and no duplicates).
            payoffs: payoffs.into_iter().map(|p| p.unwrap()).collect(),
        })
    }

    /// Number of players.
    pub fn num_players(&self) -> usize {
        self.strategy_counts.len()
    }

    /// Number of strategies available to `player`.
    pub fn strategy_count(&self, player: usize) -> usize {
        self.strategy_counts[player]
    }

    /// Number of strategy profiles (cells) — the product of the strategy counts.
    pub fn cell_count(&self) -> usize {
        self.payoffs.len()
    }

    /// Payoff to `player` in `profile`.
    pub(crate) fn payoff(&self, profile: &[usize], player: usize) -> f64 {
        let cell = cell_index(&self.strategy_counts, profile)
            .expect("profile is in range (only internally-generated profiles reach here)");
        self.payoffs[cell][player]
    }

    /// The utility vector (one payoff per player) of `profile`.
    pub(crate) fn utilities(&self, profile: &[usize]) -> &[f64] {
        let cell = cell_index(&self.strategy_counts, profile)
            .expect("profile is in range (only internally-generated profiles reach here)");
        &self.payoffs[cell]
    }

    /// Every strategy profile, in dense cell order.
    pub(crate) fn profiles(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        (0..self.cell_count()).map(move |cell| profile_of(&self.strategy_counts, cell))
    }
}

/// Mixed-radix encode a strategy profile to its dense cell index, or `None` if
/// any strategy index is out of range for its player. Player 0 is the most
/// significant digit (row-major), so iteration over cells lists profiles in
/// lexicographic order of strategy indices.
pub(crate) fn cell_index(strategy_counts: &[usize], profile: &[usize]) -> Option<usize> {
    if profile.len() != strategy_counts.len() {
        return None;
    }
    let mut index = 0usize;
    for (p, &s) in profile.iter().enumerate() {
        if s >= strategy_counts[p] {
            return None;
        }
        index = index * strategy_counts[p] + s;
    }
    Some(index)
}

/// Mixed-radix decode a dense cell index back to its strategy profile.
pub(crate) fn profile_of(strategy_counts: &[usize], mut cell: usize) -> Vec<usize> {
    let mut profile = vec![0usize; strategy_counts.len()];
    for p in (0..strategy_counts.len()).rev() {
        profile[p] = cell % strategy_counts[p];
        cell /= strategy_counts[p];
    }
    profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_index_round_trips() {
        let counts = vec![2, 3];
        for cell in 0..6 {
            let profile = profile_of(&counts, cell);
            assert_eq!(cell_index(&counts, &profile), Some(cell));
        }
        // Row-major: player 0 most significant.
        assert_eq!(cell_index(&counts, &[0, 0]), Some(0));
        assert_eq!(cell_index(&counts, &[0, 2]), Some(2));
        assert_eq!(cell_index(&counts, &[1, 0]), Some(3));
        assert_eq!(cell_index(&counts, &[1, 2]), Some(5));
    }

    #[test]
    fn out_of_range_profile_is_none() {
        assert_eq!(cell_index(&[2, 2], &[2, 0]), None);
        assert_eq!(cell_index(&[2, 2], &[0]), None);
    }

    fn full_2x2() -> Vec<(Vec<usize>, Vec<f64>)> {
        vec![
            (vec![0, 0], vec![1.0, 1.0]),
            (vec![0, 1], vec![2.0, 2.0]),
            (vec![1, 0], vec![3.0, 3.0]),
            (vec![1, 1], vec![4.0, 4.0]),
        ]
    }

    #[test]
    fn builds_a_complete_game() {
        let g = Game::from_cells(vec![2, 2], full_2x2()).unwrap();
        assert_eq!(g.num_players(), 2);
        assert_eq!(g.cell_count(), 4);
        assert_eq!(g.payoff(&[1, 0], 1), 3.0);
        assert_eq!(g.utilities(&[1, 1]), &[4.0, 4.0]);
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(Game::from_cells(vec![], vec![]), Err(GameError::Empty));
        assert_eq!(Game::from_cells(vec![2, 0], vec![]), Err(GameError::Empty));
    }

    #[test]
    fn rejects_incomplete_and_duplicate() {
        // Missing the (1,1) cell.
        let mut cells = full_2x2();
        cells.pop();
        assert!(matches!(
            Game::from_cells(vec![2, 2], cells),
            Err(GameError::IncompleteGame(_))
        ));
        // Duplicate (0,0).
        let mut cells = full_2x2();
        cells.push((vec![0, 0], vec![9.0, 9.0]));
        assert!(matches!(
            Game::from_cells(vec![2, 2], cells),
            Err(GameError::IncompleteGame(_))
        ));
    }

    #[test]
    fn rejects_malformed_cells() {
        let mut cells = full_2x2();
        cells[0].1 = vec![f64::NAN, 1.0];
        assert_eq!(
            Game::from_cells(vec![2, 2], cells),
            Err(GameError::NonFiniteUtility)
        );

        let mut cells = full_2x2();
        cells[0].0 = vec![0, 0, 0]; // wrong profile length
        assert_eq!(
            Game::from_cells(vec![2, 2], cells),
            Err(GameError::InvalidProfile)
        );

        let mut cells = full_2x2();
        cells[0].1 = vec![1.0]; // wrong utilities length
        assert_eq!(
            Game::from_cells(vec![2, 2], cells),
            Err(GameError::InvalidUtilities)
        );
    }
}
