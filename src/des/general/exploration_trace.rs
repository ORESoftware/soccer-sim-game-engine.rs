//! Versioned exploration profiles and honest selected/executed likelihood traces.
//!
//! The tactical selector remains responsible for candidate scores and feasibility.
//! This module owns reproducible sampling metadata and prevents a downstream
//! feasibility/MPC override from inheriting the probability of a different action.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const EXPLORATION_TRACE_SCHEMA_VERSION: u16 = 1;
const MIN_LOG_PROBABILITY_INPUT: f64 = 1.0e-300;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplorationMode {
    Training,
    Adaptation,
    Validation,
    Production,
}

impl ExplorationMode {
    pub const fn permits_stochastic_sampling(self) -> bool {
        matches!(self, Self::Training | Self::Adaptation)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExplorationStrategy {
    Greedy,
    EpsilonGreedy { epsilon: f64 },
    RankWeighted { weights: Vec<f64> },
    Boltzmann { temperature: f64 },
    UncertaintyDirected { beta: f64, temperature: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExplorationProfile {
    pub schema_version: u16,
    pub name: String,
    pub mode: ExplorationMode,
    pub strategy: ExplorationStrategy,
    pub seed: u64,
}

impl ExplorationProfile {
    pub fn greedy(name: impl Into<String>, mode: ExplorationMode, seed: u64) -> Self {
        Self {
            schema_version: EXPLORATION_TRACE_SCHEMA_VERSION,
            name: name.into(),
            mode,
            strategy: ExplorationStrategy::Greedy,
            seed,
        }
    }

    pub fn rank_70_20_10(seed: u64) -> Self {
        Self {
            schema_version: EXPLORATION_TRACE_SCHEMA_VERSION,
            name: "rank_70_20_10".to_owned(),
            mode: ExplorationMode::Training,
            strategy: ExplorationStrategy::RankWeighted {
                weights: vec![0.70, 0.20, 0.10],
            },
            seed,
        }
    }

    pub fn validate(&self) -> Result<(), SelectionError> {
        if self.schema_version != EXPLORATION_TRACE_SCHEMA_VERSION {
            return Err(SelectionError::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: EXPLORATION_TRACE_SCHEMA_VERSION,
            });
        }
        if self.name.trim().is_empty() {
            return Err(SelectionError::InvalidProfile(
                "profile name must not be empty".to_owned(),
            ));
        }
        match &self.strategy {
            ExplorationStrategy::Greedy => Ok(()),
            ExplorationStrategy::EpsilonGreedy { epsilon } => {
                require_unit_interval("epsilon", *epsilon)
            }
            ExplorationStrategy::RankWeighted { weights } => {
                if weights.is_empty() {
                    return Err(SelectionError::InvalidProfile(
                        "rank-weighted strategy requires at least one weight".to_owned(),
                    ));
                }
                let mut total = 0.0;
                for weight in weights {
                    require_non_negative("rank weight", *weight)?;
                    total += weight;
                }
                if !total.is_finite() || total <= 0.0 {
                    return Err(SelectionError::InvalidProfile(
                        "rank weights must contain positive finite mass".to_owned(),
                    ));
                }
                Ok(())
            }
            ExplorationStrategy::Boltzmann { temperature } => {
                require_positive("temperature", *temperature)
            }
            ExplorationStrategy::UncertaintyDirected { beta, temperature } => {
                require_non_negative("beta", *beta)?;
                require_positive("temperature", *temperature)
            }
        }
    }

    pub fn effective_strategy(&self) -> ExplorationStrategy {
        if self.mode.permits_stochastic_sampling() {
            self.strategy.clone()
        } else {
            ExplorationStrategy::Greedy
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionCandidate {
    pub action_id: u32,
    pub score: f64,
    pub uncertainty: f64,
    pub feasible: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutedLikelihoodStatus {
    Exact,
    UnknownAfterDownstreamOverride,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActionSelectionTrace {
    pub schema_version: u16,
    pub profile_name: String,
    pub mode: ExplorationMode,
    pub decision_nonce: u64,
    pub selected_action_id: u32,
    pub selected_rank: usize,
    pub selected_probability: f64,
    pub selected_log_probability: f64,
    pub executed_action_id: u32,
    pub executed_rank: usize,
    pub executed_probability: Option<f64>,
    pub executed_log_probability: Option<f64>,
    pub likelihood_status: ExecutedLikelihoodStatus,
}

impl ActionSelectionTrace {
    /// Reconcile the action that the authoritative engine actually applied.
    ///
    /// When downstream feasibility, rules, or MPC logic changes the action, the
    /// transformed distribution is unknown. Returning `None` is safer than
    /// fabricating an on-policy likelihood from the originally selected action.
    pub fn reconcile_executed_action(
        &mut self,
        executed_action_id: u32,
        candidates: &[ActionCandidate],
    ) -> Result<(), SelectionError> {
        let ranked = rank_feasible_candidates(candidates)?;
        let executed_rank = ranked
            .iter()
            .position(|candidate| candidate.action_id == executed_action_id)
            .ok_or(SelectionError::ExecutedActionNotFeasible(executed_action_id))?;

        self.executed_action_id = executed_action_id;
        self.executed_rank = executed_rank;
        if executed_action_id == self.selected_action_id {
            self.executed_probability = Some(self.selected_probability);
            self.executed_log_probability = Some(self.selected_log_probability);
            self.likelihood_status = ExecutedLikelihoodStatus::Exact;
        } else {
            self.executed_probability = None;
            self.executed_log_probability = None;
            self.likelihood_status =
                ExecutedLikelihoodStatus::UnknownAfterDownstreamOverride;
        }
        Ok(())
    }

    pub const fn has_exact_executed_likelihood(&self) -> bool {
        matches!(self.likelihood_status, ExecutedLikelihoodStatus::Exact)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    UnsupportedSchemaVersion { found: u16, expected: u16 },
    InvalidProfile(String),
    NoFeasibleCandidates,
    DuplicateActionId(u32),
    NonFiniteScore(u32),
    InvalidUncertainty(u32),
    ExecutedActionNotFeasible(u32),
}

impl Display for SelectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found, expected } => {
                write!(formatter, "unsupported schema version {found}; expected {expected}")
            }
            Self::InvalidProfile(message) => formatter.write_str(message),
            Self::NoFeasibleCandidates => formatter.write_str("no feasible action candidates"),
            Self::DuplicateActionId(action_id) => {
                write!(formatter, "duplicate action id {action_id}")
            }
            Self::NonFiniteScore(action_id) => {
                write!(formatter, "action {action_id} has a non-finite score")
            }
            Self::InvalidUncertainty(action_id) => {
                write!(formatter, "action {action_id} has invalid uncertainty")
            }
            Self::ExecutedActionNotFeasible(action_id) => {
                write!(formatter, "executed action {action_id} is not feasible")
            }
        }
    }
}

impl Error for SelectionError {}

/// Return feasible candidates in deterministic score-descending/action-id order.
pub fn rank_feasible_candidates(
    candidates: &[ActionCandidate],
) -> Result<Vec<ActionCandidate>, SelectionError> {
    let mut ranked = Vec::new();
    for candidate in candidates.iter().copied().filter(|candidate| candidate.feasible) {
        if !candidate.score.is_finite() {
            return Err(SelectionError::NonFiniteScore(candidate.action_id));
        }
        if !candidate.uncertainty.is_finite() || candidate.uncertainty < 0.0 {
            return Err(SelectionError::InvalidUncertainty(candidate.action_id));
        }
        if ranked
            .iter()
            .any(|existing: &ActionCandidate| existing.action_id == candidate.action_id)
        {
            return Err(SelectionError::DuplicateActionId(candidate.action_id));
        }
        ranked.push(candidate);
    }
    if ranked.is_empty() {
        return Err(SelectionError::NoFeasibleCandidates);
    }
    ranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.action_id.cmp(&right.action_id))
    });
    Ok(ranked)
}

/// Calculate the exact behavior distribution over deterministically ranked candidates.
pub fn behavior_distribution(
    profile: &ExplorationProfile,
    candidates: &[ActionCandidate],
) -> Result<Vec<(ActionCandidate, f64)>, SelectionError> {
    profile.validate()?;
    let ranked = rank_feasible_candidates(candidates)?;
    let probabilities = strategy_probabilities(&profile.effective_strategy(), &ranked);
    Ok(ranked.into_iter().zip(probabilities).collect())
}

pub fn select_action(
    profile: &ExplorationProfile,
    decision_nonce: u64,
    candidates: &[ActionCandidate],
) -> Result<ActionSelectionTrace, SelectionError> {
    let distribution = behavior_distribution(profile, candidates)?;
    let probabilities: Vec<f64> = distribution.iter().map(|(_, probability)| *probability).collect();
    let selected_rank = sample_index(&probabilities, deterministic_unit_draw(profile.seed, decision_nonce));
    let (selected, selected_probability) = distribution[selected_rank];
    let selected_log_probability = selected_probability.max(MIN_LOG_PROBABILITY_INPUT).ln();

    Ok(ActionSelectionTrace {
        schema_version: EXPLORATION_TRACE_SCHEMA_VERSION,
        profile_name: profile.name.clone(),
        mode: profile.mode,
        decision_nonce,
        selected_action_id: selected.action_id,
        selected_rank,
        selected_probability,
        selected_log_probability,
        executed_action_id: selected.action_id,
        executed_rank: selected_rank,
        executed_probability: Some(selected_probability),
        executed_log_probability: Some(selected_log_probability),
        likelihood_status: ExecutedLikelihoodStatus::Exact,
    })
}

fn strategy_probabilities(
    strategy: &ExplorationStrategy,
    ranked: &[ActionCandidate],
) -> Vec<f64> {
    match strategy {
        ExplorationStrategy::Greedy => one_hot(ranked.len(), 0),
        ExplorationStrategy::EpsilonGreedy { epsilon } => {
            if ranked.len() == 1 {
                return vec![1.0];
            }
            let exploratory_mass = epsilon / ranked.len() as f64;
            let mut probabilities = vec![exploratory_mass; ranked.len()];
            probabilities[0] += 1.0 - epsilon;
            probabilities
        }
        ExplorationStrategy::RankWeighted { weights } => {
            let covered = ranked.len().min(weights.len());
            let total: f64 = weights[..covered].iter().sum();
            let mut probabilities = vec![0.0; ranked.len()];
            for index in 0..covered {
                probabilities[index] = weights[index] / total;
            }
            probabilities
        }
        ExplorationStrategy::Boltzmann { temperature } => {
            softmax(ranked, *temperature, 0.0)
        }
        ExplorationStrategy::UncertaintyDirected { beta, temperature } => {
            softmax(ranked, *temperature, *beta)
        }
    }
}

fn softmax(ranked: &[ActionCandidate], temperature: f64, beta: f64) -> Vec<f64> {
    let adjusted: Vec<f64> = ranked
        .iter()
        .map(|candidate| candidate.score + beta * candidate.uncertainty)
        .collect();
    let maximum = adjusted.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<f64> = adjusted
        .iter()
        .map(|value| ((value - maximum) / temperature).exp())
        .collect();
    let total: f64 = probabilities.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return one_hot(ranked.len(), 0);
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn one_hot(count: usize, selected: usize) -> Vec<f64> {
    let mut probabilities = vec![0.0; count];
    probabilities[selected] = 1.0;
    probabilities
}

fn sample_index(probabilities: &[f64], draw: f64) -> usize {
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

fn deterministic_unit_draw(seed: u64, nonce: u64) -> f64 {
    let mixed = splitmix64(seed ^ nonce.rotate_left(29));
    (mixed >> 11) as f64 / (1_u64 << 53) as f64
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn require_unit_interval(name: &str, value: f64) -> Result<(), SelectionError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(SelectionError::InvalidProfile(format!(
            "{name} must be finite and in [0, 1]"
        )))
    }
}

fn require_positive(name: &str, value: f64) -> Result<(), SelectionError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SelectionError::InvalidProfile(format!(
            "{name} must be finite and positive"
        )))
    }
}

fn require_non_negative(name: &str, value: f64) -> Result<(), SelectionError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(SelectionError::InvalidProfile(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates() -> [ActionCandidate; 4] {
        [
            ActionCandidate { action_id: 10, score: 5.0, uncertainty: 0.1, feasible: true },
            ActionCandidate { action_id: 20, score: 4.0, uncertainty: 0.2, feasible: true },
            ActionCandidate { action_id: 30, score: 3.0, uncertainty: 2.0, feasible: true },
            ActionCandidate { action_id: 40, score: 100.0, uncertainty: 0.0, feasible: false },
        ]
    }

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-12, "left={left}, right={right}");
    }

    #[test]
    fn rank_profile_exposes_exact_distribution() {
        let distribution = behavior_distribution(&ExplorationProfile::rank_70_20_10(1), &candidates())
            .expect("valid distribution");
        assert_eq!(distribution.iter().map(|(item, _)| item.action_id).collect::<Vec<_>>(), vec![10, 20, 30]);
        assert_close(distribution[0].1, 0.70);
        assert_close(distribution[1].1, 0.20);
        assert_close(distribution[2].1, 0.10);
    }

    #[test]
    fn seed_and_nonce_replay_exactly() {
        let profile = ExplorationProfile::rank_70_20_10(99);
        assert_eq!(
            select_action(&profile, 1234, &candidates()).unwrap(),
            select_action(&profile, 1234, &candidates()).unwrap()
        );
    }

    #[test]
    fn validation_and_production_force_greedy_distribution() {
        for mode in [ExplorationMode::Validation, ExplorationMode::Production] {
            let profile = ExplorationProfile {
                schema_version: EXPLORATION_TRACE_SCHEMA_VERSION,
                name: "evaluation".to_owned(),
                mode,
                strategy: ExplorationStrategy::RankWeighted { weights: vec![0.1, 0.2, 0.7] },
                seed: 1,
            };
            let distribution = behavior_distribution(&profile, &candidates()).unwrap();
            assert_eq!(distribution.iter().map(|(_, probability)| *probability).collect::<Vec<_>>(), vec![1.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn downstream_override_never_reuses_selected_probability() {
        let profile = ExplorationProfile::rank_70_20_10(7);
        let mut trace = select_action(&profile, 5, &candidates()).unwrap();
        let replacement = if trace.selected_action_id == 10 { 20 } else { 10 };
        trace.reconcile_executed_action(replacement, &candidates()).unwrap();
        assert_eq!(trace.executed_action_id, replacement);
        assert_eq!(trace.executed_probability, None);
        assert_eq!(trace.executed_log_probability, None);
        assert_eq!(trace.likelihood_status, ExecutedLikelihoodStatus::UnknownAfterDownstreamOverride);
        assert!(!trace.has_exact_executed_likelihood());
    }

    #[test]
    fn unchanged_execution_keeps_exact_likelihood() {
        let profile = ExplorationProfile::rank_70_20_10(7);
        let mut trace = select_action(&profile, 5, &candidates()).unwrap();
        trace.reconcile_executed_action(trace.selected_action_id, &candidates()).unwrap();
        assert_eq!(trace.executed_probability, Some(trace.selected_probability));
        assert_eq!(trace.executed_log_probability, Some(trace.selected_log_probability));
        assert!(trace.has_exact_executed_likelihood());
    }

    #[test]
    fn uncertainty_can_promote_a_lower_scored_action() {
        let profile = ExplorationProfile {
            schema_version: EXPLORATION_TRACE_SCHEMA_VERSION,
            name: "uncertainty".to_owned(),
            mode: ExplorationMode::Training,
            strategy: ExplorationStrategy::UncertaintyDirected { beta: 2.0, temperature: 0.05 },
            seed: 1,
        };
        let distribution = behavior_distribution(&profile, &candidates()).unwrap();
        let most_likely = distribution
            .iter()
            .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap())
            .unwrap();
        assert_eq!(most_likely.0.action_id, 30);
    }

    #[test]
    fn invalid_candidates_and_profiles_are_rejected() {
        let duplicate = [
            ActionCandidate { action_id: 1, score: 1.0, uncertainty: 0.0, feasible: true },
            ActionCandidate { action_id: 1, score: 0.0, uncertainty: 0.0, feasible: true },
        ];
        assert_eq!(
            select_action(&ExplorationProfile::rank_70_20_10(1), 1, &duplicate),
            Err(SelectionError::DuplicateActionId(1))
        );
        let invalid_profile = ExplorationProfile {
            schema_version: EXPLORATION_TRACE_SCHEMA_VERSION,
            name: "invalid".to_owned(),
            mode: ExplorationMode::Training,
            strategy: ExplorationStrategy::EpsilonGreedy { epsilon: 1.5 },
            seed: 1,
        };
        assert!(invalid_profile.validate().is_err());
    }

    #[test]
    fn rank_weights_renormalize_for_short_candidate_lists() {
        let short = [ActionCandidate { action_id: 7, score: 2.0, uncertainty: 0.0, feasible: true }];
        let distribution = behavior_distribution(&ExplorationProfile::rank_70_20_10(1), &short).unwrap();
        assert_eq!(distribution, vec![(short[0], 1.0)]);
    }
}
