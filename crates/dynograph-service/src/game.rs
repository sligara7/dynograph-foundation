//! `POST /v1/util/game/analyze` — normal-form game-theory analysis.
//!
//! A stateless, domain-neutral pure-math endpoint (same shape as the
//! `util/pairwise_*` matrix ops): a payoff matrix goes in, the strategic
//! analysis comes out. The math lives in the `dynograph-game` crate; this
//! module is the HTTP/wire seam.
//!
//! ## What it computes (closed-form, O(matrix))
//!
//! Per-player **dominant strategies** (strict and weak), the **pure-strategy
//! Nash equilibria**, the **Pareto-optimal** outcomes, and the headline
//! **`nash_is_pareto_suboptimal`** classification — the prisoner's-dilemma
//! "rational play → collectively worse" signal — with the Pareto outcome that
//! dominates each suboptimal equilibrium. For a 2×2 two-player game it also
//! returns the closed-form interior **mixed** Nash equilibrium.
//!
//! ## Out of scope (by design)
//!
//! General N-player or non-2×2 **mixed** Nash (PPAD-complete), **cooperative**
//! solution concepts (Shapley value, core, nucleolus), and **iterated-game**
//! simulation/learning — those are research-hard or not closed-form and belong
//! in a consumer/LLM layer. Extracting a payoff matrix from prose is likewise a
//! caller concern: this endpoint is matrix in, analysis out.
//!
//! ## Wire shape
//!
//! ```json
//! POST /v1/util/game/analyze
//! {
//!   "players": [
//!     {"strategies": ["detonate", "wait"]},
//!     {"strategies": ["detonate", "wait"]}
//!   ],
//!   "payoffs": [
//!     {"profile": [0, 0], "utilities": [u_joker, u_citizens]},
//!     {"profile": [0, 1], "utilities": [...]},
//!     {"profile": [1, 0], "utilities": [...]},
//!     {"profile": [1, 1], "utilities": [...]}
//!   ]
//! }
//! ```
//!
//! `profile`/`utilities` index players in request order; a `profile` indexes
//! each player's `strategies` list. The `payoffs` list must cover every
//! strategy profile exactly once. Results report profiles as those same
//! strategy-index vectors (the caller maps them back to its labels).

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use dynograph_game::{Analysis, Game};

use crate::registry::RegistryError;

/// Safety cap on the number of strategy profiles (cells). A complete game must
/// send every cell, so the body size already bounds this, but the Pareto pass
/// is O(cells²·players); cap it (fail loud, same posture as the algo node/edge
/// caps) rather than risk pinning the blocking worker on a pathological matrix.
/// The narrative use cases are tiny (4–32 cells); 4096 is far past them.
pub(crate) const MAX_GAME_CELLS: usize = 4096;

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct GameAnalyzeRequest {
    /// Players in a fixed order; `profile`/`utilities` index them by position.
    pub players: Vec<PlayerSpec>,
    /// One cell per strategy profile, covering every profile exactly once.
    pub payoffs: Vec<CellSpec>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct PlayerSpec {
    /// This player's pure strategies; its length is the player's strategy count
    /// and a `profile` entry indexes into it. (The strategy labels document the
    /// index meaning; results report profiles as those indices.)
    pub strategies: Vec<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub(crate) struct CellSpec {
    /// One strategy index per player (in player order).
    pub profile: Vec<usize>,
    /// One payoff per player (in player order).
    pub utilities: Vec<f64>,
}

/// Per-player dominant-strategy result; a field is the dominant strategy's
/// index, or absent when none exists.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct DominantInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strictly_dominant: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weakly_dominant: Option<usize>,
}

/// A Pareto-suboptimal pure equilibrium and a Pareto outcome that dominates it.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct NashDominationInfo {
    pub nash: Vec<usize>,
    pub dominated_by: Vec<usize>,
}

/// The interior 2×2 mixed Nash equilibrium: each player's mixing probabilities.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct Mixed2x2Info {
    pub player0_probs: [f64; 2],
    pub player1_probs: [f64; 2],
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct GameAnalyzeResponse {
    /// Per player (request order): strict/weak dominant strategy, if any.
    pub dominant: Vec<DominantInfo>,
    /// Pure-strategy Nash equilibria, as strategy-index profiles.
    pub pure_nash: Vec<Vec<usize>>,
    /// Pareto-optimal outcomes, as strategy-index profiles.
    pub pareto_optimal: Vec<Vec<usize>>,
    /// True iff some pure equilibrium is Pareto-dominated by another outcome.
    pub nash_is_pareto_suboptimal: bool,
    /// For each suboptimal equilibrium, a Pareto outcome that dominates it.
    pub nash_domination: Vec<NashDominationInfo>,
    /// Interior 2×2 mixed equilibrium, when the game is 2×2 and one exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixed_2x2: Option<Mixed2x2Info>,
}

/// Validate the request, build the game, and analyze it. All failures are 400s
/// (the input is the caller's to fix).
pub(crate) fn run_analyze(req: GameAnalyzeRequest) -> Result<GameAnalyzeResponse, RegistryError> {
    if req.players.is_empty() {
        return Err(RegistryError::BadRequest(
            "players must be non-empty".to_string(),
        ));
    }
    let mut strategy_counts = Vec::with_capacity(req.players.len());
    for (p, player) in req.players.iter().enumerate() {
        if player.strategies.is_empty() {
            return Err(RegistryError::BadRequest(format!(
                "player {p} must have at least one strategy"
            )));
        }
        strategy_counts.push(player.strategies.len());
    }

    // Cap the profile count (with overflow-safe multiplication) before building.
    let mut cells_total: usize = 1;
    for &s in &strategy_counts {
        cells_total = cells_total
            .checked_mul(s)
            .filter(|&n| n <= MAX_GAME_CELLS)
            .ok_or_else(|| {
                RegistryError::BadRequest(format!(
                    "game has more than the {MAX_GAME_CELLS}-profile limit"
                ))
            })?;
    }

    let cells: Vec<(Vec<usize>, Vec<f64>)> = req
        .payoffs
        .into_iter()
        .map(|c| (c.profile, c.utilities))
        .collect();
    let game = Game::from_cells(strategy_counts, cells)
        .map_err(|e| RegistryError::BadRequest(e.to_string()))?;

    Ok(response_of(dynograph_game::analyze(&game)))
}

fn response_of(a: Analysis) -> GameAnalyzeResponse {
    let dominant = a
        .strictly_dominant
        .into_iter()
        .zip(a.weakly_dominant)
        .map(|(strictly_dominant, weakly_dominant)| DominantInfo {
            strictly_dominant,
            weakly_dominant,
        })
        .collect();
    GameAnalyzeResponse {
        dominant,
        pure_nash: a.pure_nash,
        pareto_optimal: a.pareto_optimal,
        nash_is_pareto_suboptimal: a.nash_is_pareto_suboptimal,
        nash_domination: a
            .nash_domination
            .into_iter()
            .map(|d| NashDominationInfo {
                nash: d.nash,
                dominated_by: d.dominated_by,
            })
            .collect(),
        mixed_2x2: a.mixed_2x2.map(|m| Mixed2x2Info {
            player0_probs: m.player0_probs,
            player1_probs: m.player1_probs,
        }),
    }
}
