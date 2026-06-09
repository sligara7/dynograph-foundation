//! Error type for game construction.
//!
//! Following the foundation's core principle (fail loudly, never swallow a
//! semantic failure), [`Game::from_cells`](crate::Game::from_cells) rejects a
//! malformed game rather than analyzing a half-specified one. The service layer
//! maps these to `400`s.

use std::fmt;

/// Things that can go wrong building a [`Game`](crate::Game).
#[derive(Debug, Clone, PartialEq)]
pub enum GameError {
    /// No players, or a player with no strategies — there is nothing to analyze.
    Empty,
    /// A payoff utility was NaN or infinite; payoffs must be finite so the
    /// dominance/Pareto comparisons are well-defined.
    NonFiniteUtility,
    /// A cell's strategy profile had the wrong length (≠ number of players) or
    /// a strategy index out of range for that player.
    InvalidProfile,
    /// A cell's utility vector had the wrong length (≠ number of players).
    InvalidUtilities,
    /// The cells do not cover every strategy profile exactly once — the game is
    /// incomplete or has a duplicate cell. The string names the offending
    /// profile / count.
    IncompleteGame(String),
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::Empty => {
                write!(
                    f,
                    "a game needs at least one player, each with at least one strategy"
                )
            }
            GameError::NonFiniteUtility => {
                write!(f, "payoff utilities must be finite (no NaN/infinity)")
            }
            GameError::InvalidProfile => write!(
                f,
                "a cell's strategy profile must have one in-range strategy index per player"
            ),
            GameError::InvalidUtilities => {
                write!(f, "a cell's utilities must have one payoff per player")
            }
            GameError::IncompleteGame(detail) => write!(
                f,
                "payoffs must cover every strategy profile exactly once: {detail}"
            ),
        }
    }
}

impl std::error::Error for GameError {}
