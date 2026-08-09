//! Versioned soccer reward/penalty contract with anti-specification-gaming checks.
//!
//! Penalties are negative rewards. Cost-style consumers use `cost = -reward`.
//! Raw component contributions are retained before clipping or normalization.

use std::{
    collections::HashSet,
    error::Error,
    fmt::{Display, Formatter},
};

pub const REWARD_CONTRACT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RewardComponentId {
    GoalFor,
    GoalAgainst,
    ExpectedGoalsDelta,
    PossessionValueDelta,
    ProgressiveAction,
    DefensiveIntegrity,
    Foul,
    Offside,
    Turnover,
    InfeasibleAction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardGroup {
    Terminal,
    Shaping,
    Penalty,
    Safety,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewardPolarity {
    Reward,
    Penalty,
    Signed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardComponentSpec {
    pub id: RewardComponentId,
    pub group: RewardGroup,
    pub polarity: RewardPolarity,
    pub minimum: f64,
    pub maximum: f64,
    pub weight: f64,
}

impl RewardComponentSpec {
    pub fn weighted_bounds(self) -> (f64, f64) {
        let left = self.minimum * self.weight;
        let right = self.maximum * self.weight;
        (left.min(right), left.max(right))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RewardContract {
    pub schema_version: u16,
    pub name: String,
    pub terminal_dominance_margin: f64,
    pub components: Vec<RewardComponentSpec>,
}

impl RewardContract {
    pub fn soccer_v1() -> Self {
        use RewardComponentId as Id;
        use RewardGroup as Group;
        use RewardPolarity as Polarity;

        Self {
            schema_version: REWARD_CONTRACT_SCHEMA_VERSION,
            name: "soccer-reward-v1".to_owned(),
            terminal_dominance_margin: 20.0,
            components: vec![
                RewardComponentSpec {
                    id: Id::GoalFor,
                    group: Group::Terminal,
                    polarity: Polarity::Reward,
                    minimum: 1.0,
                    maximum: 1.0,
                    weight: 100.0,
                },
                RewardComponentSpec {
                    id: Id::GoalAgainst,
                    group: Group::Terminal,
                    polarity: Polarity::Penalty,
                    minimum: 1.0,
                    maximum: 1.0,
                    weight: -100.0,
                },
                RewardComponentSpec {
                    id: Id::ExpectedGoalsDelta,
                    group: Group::Shaping,
                    polarity: Polarity::Signed,
                    minimum: -3.0,
                    maximum: 3.0,
                    weight: 2.0,
                },
                RewardComponentSpec {
                    id: Id::PossessionValueDelta,
                    group: Group::Shaping,
                    polarity: Polarity::Signed,
                    minimum: -1.0,
                    maximum: 1.0,
                    weight: 0.5,
                },
                RewardComponentSpec {
                    id: Id::ProgressiveAction,
                    group: Group::Shaping,
                    polarity: Polarity::Signed,
                    minimum: -1.0,
                    maximum: 1.0,
                    weight: 0.75,
                },
                RewardComponentSpec {
                    id: Id::DefensiveIntegrity,
                    group: Group::Shaping,
                    polarity: Polarity::Signed,
                    minimum: -1.0,
                    maximum: 1.0,
                    weight: 0.75,
                },
                RewardComponentSpec {
                    id: Id::Foul,
                    group: Group::Penalty,
                    polarity: Polarity::Penalty,
                    minimum: 0.0,
                    maximum: 1.0,
                    weight: -2.0,
                },
                RewardComponentSpec {
                    id: Id::Offside,
                    group: Group::Penalty,
                    polarity: Polarity::Penalty,
                    minimum: 0.0,
                    maximum: 1.0,
                    weight: -1.0,
                },
                RewardComponentSpec {
                    id: Id::Turnover,
                    group: Group::Penalty,
                    polarity: Polarity::Penalty,
                    minimum: 0.0,
                    maximum: 1.0,
                    weight: -2.0,
                },
                RewardComponentSpec {
                    id: Id::InfeasibleAction,
                    group: Group::Safety,
                    polarity: Polarity::Penalty,
                    minimum: 0.0,
                    maximum: 1.0,
                    weight: -5.0,
                },
            ],
        }
    }

    pub fn component(&self, id: RewardComponentId) -> Option<&RewardComponentSpec> {
        self.components.iter().find(|component| component.id == id)
    }

    pub fn validate(&self) -> Result<RewardContractValidation, RewardContractError> {
        if self.schema_version != REWARD_CONTRACT_SCHEMA_VERSION {
            return Err(RewardContractError::UnsupportedSchemaVersion {
                found: self.schema_version,
                expected: REWARD_CONTRACT_SCHEMA_VERSION,
            });
        }
        if self.name.trim().is_empty() {
            return Err(RewardContractError::InvalidContract(
                "contract name must not be empty".to_owned(),
            ));
        }
        if !self.terminal_dominance_margin.is_finite()
            || self.terminal_dominance_margin < 0.0
        {
            return Err(RewardContractError::InvalidContract(
                "terminal dominance margin must be finite and non-negative".to_owned(),
            ));
        }
        if self.components.is_empty() {
            return Err(RewardContractError::InvalidContract(
                "contract must contain reward components".to_owned(),
            ));
        }

        let mut seen = HashSet::new();
        let mut terminal_ids = HashSet::new();
        for component in &self.components {
            if !seen.insert(component.id) {
                return Err(RewardContractError::DuplicateComponent(component.id));
            }
            if !component.minimum.is_finite()
                || !component.maximum.is_finite()
                || !component.weight.is_finite()
            {
                return Err(RewardContractError::InvalidContract(format!(
                    "{:?} bounds and weight must be finite",
                    component.id
                )));
            }
            if component.minimum > component.maximum {
                return Err(RewardContractError::InvalidContract(format!(
                    "{:?} minimum exceeds maximum",
                    component.id
                )));
            }

            let (weighted_minimum, weighted_maximum) = component.weighted_bounds();
            match component.polarity {
                RewardPolarity::Reward if weighted_minimum < 0.0 => {
                    return Err(RewardContractError::InvalidContract(format!(
                        "{:?} reward component can become negative",
                        component.id
                    )));
                }
                RewardPolarity::Penalty if weighted_maximum > 0.0 => {
                    return Err(RewardContractError::InvalidContract(format!(
                        "{:?} penalty component can become positive",
                        component.id
                    )));
                }
                _ => {}
            }

            if component.group == RewardGroup::Terminal {
                terminal_ids.insert(component.id);
            }
        }

        let expected_terminals = HashSet::from([
            RewardComponentId::GoalFor,
            RewardComponentId::GoalAgainst,
        ]);
        if terminal_ids != expected_terminals {
            return Err(RewardContractError::InvalidContract(
                "terminal components must be exactly GoalFor and GoalAgainst".to_owned(),
            ));
        }

        let goal_for = *self
            .component(RewardComponentId::GoalFor)
            .ok_or(RewardContractError::MissingComponent(
                RewardComponentId::GoalFor,
            ))?;
        let goal_against = *self
            .component(RewardComponentId::GoalAgainst)
            .ok_or(RewardContractError::MissingComponent(
                RewardComponentId::GoalAgainst,
            ))?;
        let (goal_for_minimum, goal_for_maximum) = goal_for.weighted_bounds();
        let (goal_against_minimum, goal_against_maximum) = goal_against.weighted_bounds();
        if (goal_for_minimum - goal_for_maximum).abs() > 1.0e-12
            || (goal_against_minimum - goal_against_maximum).abs() > 1.0e-12
            || goal_for_minimum <= 0.0
            || goal_against_maximum >= 0.0
            || (goal_for_minimum + goal_against_maximum).abs() > 1.0e-12
        {
            return Err(RewardContractError::InvalidContract(
                "GoalFor and GoalAgainst must be fixed, non-zero, and symmetric".to_owned(),
            ));
        }

        let nonterminal_absolute_envelope: f64 = self
            .components
            .iter()
            .filter(|component| component.group != RewardGroup::Terminal)
            .map(|component| {
                let (minimum, maximum) = component.weighted_bounds();
                minimum.abs().max(maximum.abs())
            })
            .sum();
        let actual_margin = goal_for_minimum - nonterminal_absolute_envelope;
        if actual_margin + 1.0e-12 < self.terminal_dominance_margin {
            return Err(RewardContractError::TerminalDominanceViolation {
                terminal_magnitude: goal_for_minimum,
                nonterminal_envelope: nonterminal_absolute_envelope,
                required_margin: self.terminal_dominance_margin,
            });
        }

        let infeasible = *self
            .component(RewardComponentId::InfeasibleAction)
            .ok_or(RewardContractError::MissingComponent(
                RewardComponentId::InfeasibleAction,
            ))?;
        let (infeasible_minimum, infeasible_maximum) = infeasible.weighted_bounds();
        if infeasible_minimum >= 0.0 || infeasible_maximum > 0.0 {
            return Err(RewardContractError::InvalidContract(
                "InfeasibleAction must be a strictly negative optional penalty".to_owned(),
            ));
        }

        Ok(RewardContractValidation {
            terminal_reward_magnitude: goal_for_minimum,
            nonterminal_absolute_envelope,
            required_margin: self.terminal_dominance_margin,
            actual_margin,
        })
    }

    /// Score one transition while retaining its raw component breakdown.
    ///
    /// Missing components contribute zero. Duplicate or out-of-range values are
    /// rejected instead of silently double-counted or clipped.
    pub fn score(
        &self,
        values: &[(RewardComponentId, f64)],
    ) -> Result<RewardBreakdown, RewardContractError> {
        self.validate()?;
        let mut seen = HashSet::new();
        let mut contributions = Vec::with_capacity(values.len());
        let mut reward = 0.0;

        for (id, raw_value) in values.iter().copied() {
            if !seen.insert(id) {
                return Err(RewardContractError::DuplicateRuntimeValue(id));
            }
            let component = *self
                .component(id)
                .ok_or(RewardContractError::MissingComponent(id))?;
            if !raw_value.is_finite()
                || raw_value < component.minimum
                || raw_value > component.maximum
            {
                return Err(RewardContractError::RuntimeValueOutOfRange {
                    id,
                    value: raw_value,
                    minimum: component.minimum,
                    maximum: component.maximum,
                });
            }
            let weighted_value = raw_value * component.weight;
            reward += weighted_value;
            contributions.push(RewardContribution {
                id,
                raw_value,
                weight: component.weight,
                weighted_value,
            });
        }

        Ok(RewardBreakdown {
            reward,
            cost: -reward,
            contributions,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardContractValidation {
    pub terminal_reward_magnitude: f64,
    pub nonterminal_absolute_envelope: f64,
    pub required_margin: f64,
    pub actual_margin: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardContribution {
    pub id: RewardComponentId,
    pub raw_value: f64,
    pub weight: f64,
    pub weighted_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RewardBreakdown {
    pub reward: f64,
    pub cost: f64,
    pub contributions: Vec<RewardContribution>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RewardContractError {
    UnsupportedSchemaVersion {
        found: u16,
        expected: u16,
    },
    InvalidContract(String),
    DuplicateComponent(RewardComponentId),
    MissingComponent(RewardComponentId),
    TerminalDominanceViolation {
        terminal_magnitude: f64,
        nonterminal_envelope: f64,
        required_margin: f64,
    },
    DuplicateRuntimeValue(RewardComponentId),
    RuntimeValueOutOfRange {
        id: RewardComponentId,
        value: f64,
        minimum: f64,
        maximum: f64,
    },
}

impl Display for RewardContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { found, expected } => {
                write!(formatter, "unsupported schema version {found}; expected {expected}")
            }
            Self::InvalidContract(message) => formatter.write_str(message),
            Self::DuplicateComponent(id) => write!(formatter, "duplicate component {id:?}"),
            Self::MissingComponent(id) => write!(formatter, "missing component {id:?}"),
            Self::TerminalDominanceViolation {
                terminal_magnitude,
                nonterminal_envelope,
                required_margin,
            } => write!(
                formatter,
                "non-terminal envelope {nonterminal_envelope} can overpower terminal magnitude {terminal_magnitude} with required margin {required_margin}"
            ),
            Self::DuplicateRuntimeValue(id) => {
                write!(formatter, "duplicate runtime value for {id:?}")
            }
            Self::RuntimeValueOutOfRange {
                id,
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "runtime value {value} for {id:?} is outside [{minimum}, {maximum}]"
            ),
        }
    }
}

impl Error for RewardContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1.0e-12, "left={left}, right={right}");
    }

    #[test]
    fn soccer_v1_has_terminal_dominance_margin() {
        let validation = RewardContract::soccer_v1().validate().unwrap();
        assert_close(validation.terminal_reward_magnitude, 100.0);
        assert_close(validation.nonterminal_absolute_envelope, 18.0);
        assert_close(validation.required_margin, 20.0);
        assert_close(validation.actual_margin, 82.0);
    }

    #[test]
    fn score_preserves_breakdown_and_cost_is_negative_reward() {
        let breakdown = RewardContract::soccer_v1()
            .score(&[
                (RewardComponentId::ExpectedGoalsDelta, 0.5),
                (RewardComponentId::ProgressiveAction, 1.0),
                (RewardComponentId::Turnover, 1.0),
            ])
            .unwrap();
        assert_close(breakdown.reward, -0.25);
        assert_close(breakdown.cost, 0.25);
        assert_eq!(breakdown.contributions.len(), 3);
    }

    #[test]
    fn duplicate_runtime_values_are_rejected() {
        let error = RewardContract::soccer_v1()
            .score(&[
                (RewardComponentId::Turnover, 1.0),
                (RewardComponentId::Turnover, 0.0),
            ])
            .unwrap_err();
        assert_eq!(
            error,
            RewardContractError::DuplicateRuntimeValue(RewardComponentId::Turnover)
        );
    }

    #[test]
    fn out_of_range_runtime_values_are_rejected() {
        let error = RewardContract::soccer_v1()
            .score(&[(RewardComponentId::Turnover, 2.0)])
            .unwrap_err();
        assert!(matches!(
            error,
            RewardContractError::RuntimeValueOutOfRange {
                id: RewardComponentId::Turnover,
                ..
            }
        ));
    }

    #[test]
    fn positive_penalty_contract_is_rejected() {
        let mut contract = RewardContract::soccer_v1();
        contract
            .components
            .iter_mut()
            .find(|component| component.id == RewardComponentId::Foul)
            .unwrap()
            .weight = 2.0;
        assert!(matches!(
            contract.validate(),
            Err(RewardContractError::InvalidContract(message))
                if message.contains("penalty component can become positive")
        ));
    }

    #[test]
    fn shaping_cannot_overpower_match_outcome() {
        let mut contract = RewardContract::soccer_v1();
        contract
            .components
            .iter_mut()
            .find(|component| component.id == RewardComponentId::ExpectedGoalsDelta)
            .unwrap()
            .weight = 40.0;
        assert!(matches!(
            contract.validate(),
            Err(RewardContractError::TerminalDominanceViolation { .. })
        ));
    }

    #[test]
    fn asymmetric_terminal_rewards_are_rejected() {
        let mut contract = RewardContract::soccer_v1();
        contract
            .components
            .iter_mut()
            .find(|component| component.id == RewardComponentId::GoalAgainst)
            .unwrap()
            .weight = -90.0;
        assert!(matches!(
            contract.validate(),
            Err(RewardContractError::InvalidContract(message))
                if message.contains("symmetric")
        ));
    }

    #[test]
    fn infeasible_action_must_remain_a_negative_penalty() {
        let mut contract = RewardContract::soccer_v1();
        contract
            .components
            .iter_mut()
            .find(|component| component.id == RewardComponentId::InfeasibleAction)
            .unwrap()
            .weight = 0.0;
        assert!(matches!(
            contract.validate(),
            Err(RewardContractError::InvalidContract(message))
                if message.contains("InfeasibleAction")
        ));
    }
}
