//! The closed-form, O(matrix) analysis of a normal-form [`Game`].
//!
//! Produces, for the whole game: each player's strictly/weakly **dominant**
//! strategy (if any), the **pure-strategy Nash equilibria**, the
//! **Pareto-optimal** outcomes, and the headline classification —
//! `nash_is_pareto_suboptimal` (the prisoner's-dilemma "rational play →
//! collectively worse" signal) with the Pareto outcome that dominates each
//! suboptimal equilibrium. For a 2×2 two-player game it also returns the
//! closed-form interior **mixed** Nash equilibrium.
//!
//! Out of scope (by design — research-hard / not closed-form): general N-player
//! or non-2×2 mixed Nash, cooperative solution concepts (Shapley/core), and
//! iterated-game simulation.

use crate::game::Game;

/// The interior mixed-strategy Nash equilibrium of a 2×2 two-player game: each
/// player's mixing probability over their two strategies. Present only when the
/// game is 2×2 and a genuine interior mixed equilibrium exists (both players
/// mix strictly between their strategies).
#[derive(Debug, Clone, PartialEq)]
pub struct Mixed2x2 {
    /// Player 0's probabilities `[P(strategy 0), P(strategy 1)]`.
    pub player0_probs: [f64; 2],
    /// Player 1's probabilities `[P(strategy 0), P(strategy 1)]`.
    pub player1_probs: [f64; 2],
}

/// A pure Nash equilibrium that is Pareto-suboptimal, paired with a
/// Pareto-optimal outcome that dominates it (every player at least as well off,
/// at least one strictly better).
#[derive(Debug, Clone, PartialEq)]
pub struct NashDomination {
    pub nash: Vec<usize>,
    pub dominated_by: Vec<usize>,
}

/// The full analysis of a [`Game`]. Profiles are strategy-index vectors (one
/// per player); lists are in dense cell order for determinism.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// Per player: the strictly dominant strategy index, or `None`.
    pub strictly_dominant: Vec<Option<usize>>,
    /// Per player: the weakly dominant strategy index, or `None`.
    pub weakly_dominant: Vec<Option<usize>>,
    /// Pure-strategy Nash equilibria (profiles where no player can unilaterally
    /// improve).
    pub pure_nash: Vec<Vec<usize>>,
    /// Pareto-optimal outcomes (profiles whose utility vector is dominated by no
    /// other profile's).
    pub pareto_optimal: Vec<Vec<usize>>,
    /// True iff at least one pure Nash equilibrium is Pareto-suboptimal.
    pub nash_is_pareto_suboptimal: bool,
    /// For each Pareto-suboptimal pure equilibrium, a Pareto outcome that
    /// dominates it.
    pub nash_domination: Vec<NashDomination>,
    /// The interior 2×2 mixed Nash equilibrium, when applicable.
    pub mixed_2x2: Option<Mixed2x2>,
}

/// Analyze `game`. Pure: fully determined by the payoffs, no I/O.
pub fn analyze(game: &Game) -> Analysis {
    let players = game.num_players();
    let strictly_dominant = (0..players).map(|p| dominant(game, p, true)).collect();
    let weakly_dominant = (0..players).map(|p| dominant(game, p, false)).collect();

    let profiles: Vec<Vec<usize>> = game.profiles().collect();
    let pure_nash: Vec<Vec<usize>> = profiles
        .iter()
        .filter(|profile| is_nash(game, profile))
        .cloned()
        .collect();
    let pareto_optimal: Vec<Vec<usize>> = profiles
        .iter()
        .filter(|profile| {
            !profiles
                .iter()
                .any(|other| pareto_dominates(game.utilities(other), game.utilities(profile)))
        })
        .cloned()
        .collect();

    // A pure equilibrium is suboptimal when it isn't Pareto-optimal. Some
    // Pareto-optimal outcome then dominates it (domination is transitive, so a
    // maximal — Pareto-optimal — dominator always exists).
    let mut nash_domination = Vec::new();
    for ne in &pure_nash {
        if pareto_optimal.contains(ne) {
            continue;
        }
        if let Some(dom) = pareto_optimal
            .iter()
            .find(|po| pareto_dominates(game.utilities(po), game.utilities(ne)))
        {
            nash_domination.push(NashDomination {
                nash: ne.clone(),
                dominated_by: dom.clone(),
            });
        }
    }

    Analysis {
        strictly_dominant,
        weakly_dominant,
        pure_nash,
        pareto_optimal,
        nash_is_pareto_suboptimal: !nash_domination.is_empty(),
        nash_domination,
        mixed_2x2: mixed_2x2(game),
    }
}

/// The strategy that (strictly or weakly) dominates every other strategy for
/// `player`, or `None`. Strict: payoff strictly higher in every opponent
/// context. Weak: never lower, and strictly higher in at least one context.
fn dominant(game: &Game, player: usize, strict: bool) -> Option<usize> {
    let count = game.strategy_count(player);
    (0..count).find(|&s| (0..count).all(|t| t == s || beats(game, player, s, t, strict)))
}

/// Does strategy `s` (strictly or weakly) dominate strategy `t` for `player`?
/// Compares payoffs across every opponent context (every profile in which the
/// player plays `s`, substituting `t` to form the counterfactual).
fn beats(game: &Game, player: usize, s: usize, t: usize, strict: bool) -> bool {
    let mut strictly_better_somewhere = false;
    for profile in game.profiles() {
        if profile[player] != s {
            continue;
        }
        let with_s = game.payoff(&profile, player);
        let mut alt = profile.clone();
        alt[player] = t;
        let with_t = game.payoff(&alt, player);
        if strict {
            if with_s <= with_t {
                return false;
            }
        } else {
            if with_s < with_t {
                return false;
            }
            if with_s > with_t {
                strictly_better_somewhere = true;
            }
        }
    }
    strict || strictly_better_somewhere
}

/// Is `profile` a pure Nash equilibrium — no player can switch to a strategy
/// with a strictly higher payoff, holding the others fixed?
fn is_nash(game: &Game, profile: &[usize]) -> bool {
    (0..game.num_players()).all(|player| {
        let current = game.payoff(profile, player);
        (0..game.strategy_count(player)).all(|t| {
            let mut alt = profile.to_vec();
            alt[player] = t;
            game.payoff(&alt, player) <= current
        })
    })
}

/// Does outcome `a` Pareto-dominate outcome `b` — every player at least as well
/// off under `a`, and at least one strictly better?
fn pareto_dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai < bi {
            return false;
        }
        if ai > bi {
            strictly_better = true;
        }
    }
    strictly_better
}

/// The interior mixed Nash equilibrium of a 2×2 two-player game via the
/// indifference closed form: each player mixes to make the other indifferent.
/// `None` unless the game is 2×2 and both solved probabilities are strictly
/// interior (a genuine mixed equilibrium, not a degenerate one masking pure
/// dominance).
fn mixed_2x2(game: &Game) -> Option<Mixed2x2> {
    if game.num_players() != 2 || game.strategy_count(0) != 2 || game.strategy_count(1) != 2 {
        return None;
    }
    let u = |player: usize, i: usize, j: usize| game.payoff(&[i, j], player);

    // Player 0's mix p (prob of strategy 0) makes player 1 indifferent;
    // player 1's mix q makes player 0 indifferent.
    let d1 = u(1, 0, 0) - u(1, 0, 1) - u(1, 1, 0) + u(1, 1, 1);
    let d0 = u(0, 0, 0) - u(0, 0, 1) - u(0, 1, 0) + u(0, 1, 1);
    if d0 == 0.0 || d1 == 0.0 {
        return None;
    }
    let p = (u(1, 1, 1) - u(1, 1, 0)) / d1;
    let q = (u(0, 1, 1) - u(0, 0, 1)) / d0;
    if !is_interior(p) || !is_interior(q) {
        return None;
    }
    Some(Mixed2x2 {
        player0_probs: [p, 1.0 - p],
        player1_probs: [q, 1.0 - q],
    })
}

/// A genuine mixing probability lies strictly in `(0, 1)`.
fn is_interior(x: f64) -> bool {
    x.is_finite() && x > 0.0 && x < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `cells2x2[(i,j)] = [u0, u1]` for a 2-player 2×2 game.
    fn game_2x2(c: [[[f64; 2]; 2]; 2]) -> Game {
        let mut cells = Vec::new();
        for (i, row) in c.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                cells.push((vec![i, j], cell.to_vec()));
            }
        }
        Game::from_cells(vec![2, 2], cells).unwrap()
    }

    #[test]
    fn prisoners_dilemma() {
        // Strategy 0 = cooperate, 1 = defect. Payoffs (row, col):
        // (C,C)=3,3  (C,D)=0,5  (D,C)=5,0  (D,D)=1,1
        let g = game_2x2([[[3.0, 3.0], [0.0, 5.0]], [[5.0, 0.0], [1.0, 1.0]]]);
        let a = analyze(&g);
        // Defect (strategy 1) strictly dominates for both.
        assert_eq!(a.strictly_dominant, vec![Some(1), Some(1)]);
        // Unique pure NE = (D,D).
        assert_eq!(a.pure_nash, vec![vec![1, 1]]);
        // (D,D)=(1,1) is Pareto-dominated by (C,C)=(3,3); it is NOT Pareto-optimal.
        assert!(!a.pareto_optimal.contains(&vec![1, 1]));
        assert!(a.pareto_optimal.contains(&vec![0, 0]));
        // The headline.
        assert!(a.nash_is_pareto_suboptimal);
        assert_eq!(a.nash_domination.len(), 1);
        assert_eq!(a.nash_domination[0].nash, vec![1, 1]);
        assert_eq!(a.nash_domination[0].dominated_by, vec![0, 0]);
        // Degenerate mixed (strict dominance) => no interior mixed NE.
        assert_eq!(a.mixed_2x2, None);
    }

    #[test]
    fn stag_hunt_has_two_pure_equilibria_and_a_mixed_one() {
        // 0 = stag, 1 = hare. (S,S)=2,2 (S,H)=0,1 (H,S)=1,0 (H,H)=1,1
        let g = game_2x2([[[2.0, 2.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]);
        let a = analyze(&g);
        assert_eq!(a.pure_nash, vec![vec![0, 0], vec![1, 1]]);
        assert_eq!(a.strictly_dominant, vec![None, None]);
        // The (hare,hare) equilibrium is Pareto-dominated by (stag,stag) — the
        // payoff-dominant vs risk-dominant tension — so the game IS suboptimal.
        assert!(a.nash_is_pareto_suboptimal);
        assert!(
            a.nash_domination
                .iter()
                .any(|d| d.nash == vec![1, 1] && d.dominated_by == vec![0, 0])
        );
        // Interior mixed NE at (1/2, 1/2).
        let m = a.mixed_2x2.unwrap();
        assert!((m.player0_probs[0] - 0.5).abs() < 1e-9);
        assert!((m.player1_probs[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn chicken_two_pure_equilibria() {
        // 0 = swerve, 1 = straight. (Sw,Sw)=3,3 (Sw,St)=2,4 (St,Sw)=4,2 (St,St)=1,1
        let g = game_2x2([[[3.0, 3.0], [2.0, 4.0]], [[4.0, 2.0], [1.0, 1.0]]]);
        let a = analyze(&g);
        // Two asymmetric pure NE: (swerve,straight) and (straight,swerve).
        assert_eq!(a.pure_nash, vec![vec![0, 1], vec![1, 0]]);
        assert_eq!(a.strictly_dominant, vec![None, None]);
        // Both are Pareto-optimal.
        assert!(!a.nash_is_pareto_suboptimal);
        // Interior mixed NE exists.
        assert!(a.mixed_2x2.is_some());
    }

    #[test]
    fn matching_pennies_no_pure_ne_mixed_half() {
        // Zero-sum: player 0 matches, player 1 mismatches.
        // (H,H)=1,-1 (H,T)=-1,1 (T,H)=-1,1 (T,T)=1,-1
        let g = game_2x2([[[1.0, -1.0], [-1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]]);
        let a = analyze(&g);
        assert!(a.pure_nash.is_empty(), "no pure NE");
        assert!(!a.nash_is_pareto_suboptimal);
        let m = a.mixed_2x2.unwrap();
        assert!((m.player0_probs[0] - 0.5).abs() < 1e-9);
        assert!((m.player1_probs[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn weak_dominance_detected() {
        // Player 0: strategy 1 weakly dominates 0 (>= everywhere, > somewhere).
        // (0,0)=1,0 (0,1)=0,0 (1,0)=1,0 (1,1)=2,0
        let g = game_2x2([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]]);
        let a = analyze(&g);
        assert_eq!(a.weakly_dominant[0], Some(1));
        assert_eq!(a.strictly_dominant[0], None, "tie at col 0 blocks strict");
    }

    #[test]
    fn three_player_pure_nash() {
        // Minimal 3-player, each 2 strategies: every player prefers to match a
        // global "all play 0" coordination point. Build so (0,0,0) is the
        // unique NE and Pareto-best.
        let counts = vec![2, 2, 2];
        let mut cells = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let all_zero = i == 0 && j == 0 && k == 0;
                    let payoff = if all_zero { 1.0 } else { 0.0 };
                    cells.push((vec![i, j, k], vec![payoff, payoff, payoff]));
                }
            }
        }
        let g = Game::from_cells(counts, cells).unwrap();
        let a = analyze(&g);
        assert!(a.pure_nash.contains(&vec![0, 0, 0]));
        // Only the coordination point yields positive payoff, so it is the lone
        // Pareto-optimal outcome.
        assert_eq!(a.pareto_optimal, vec![vec![0, 0, 0]]);
        // Failed-coordination profiles (≥2 players off the point) are weak NE
        // that the coordination outcome Pareto-dominates → suboptimal.
        assert!(a.nash_is_pareto_suboptimal);
        // No mixed result for N != 2.
        assert_eq!(a.mixed_2x2, None);
    }
}
