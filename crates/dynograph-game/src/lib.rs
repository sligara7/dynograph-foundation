//! DynoGraph Game — domain-neutral, closed-form normal-form game-theory
//! analysis.
//!
//! The game-theory sibling of `dynograph-vector` / `dynograph-graph`: pure,
//! dependency-free math with no storage, service, or core dependency. The
//! caller builds a [`Game`] from a payoff matrix (via [`Game::from_cells`]) and
//! runs [`analyze`]; the result classifies the game's strategic structure.
//!
//! Scope is the **easy, high-value, O(matrix)** half:
//! - per-player **dominant strategies** (strict and weak),
//! - **pure-strategy Nash equilibria**,
//! - **Pareto-optimal** outcomes,
//! - the headline **`nash_is_pareto_suboptimal`** classification (the
//!   prisoner's-dilemma "rational play → collectively worse" signal) plus the
//!   Pareto outcome that dominates each suboptimal equilibrium, and
//! - the closed-form interior **2×2 mixed** Nash equilibrium.
//!
//! Deliberately **out of scope** (research-hard or not closed-form): general
//! N-player / non-2×2 mixed Nash (PPAD-complete), cooperative solution concepts
//! (Shapley value, core, nucleolus), and iterated-game simulation/learning —
//! those belong in a consumer layer, not in this stateless solver. Extracting a
//! payoff matrix from prose is likewise a caller concern; this crate is matrix
//! in, analysis out.
//!
//! Following the foundation's fail-loud principle, a malformed game (incomplete
//! payoff coverage, non-finite payoff, ill-shaped cell) is rejected via
//! [`GameError`] rather than analyzed half-specified.

mod analyze;
mod error;
mod game;

pub use analyze::{Analysis, Mixed2x2, NashDomination, analyze};
pub use error::GameError;
pub use game::Game;
