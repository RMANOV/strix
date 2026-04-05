use serde::{Deserialize, Serialize};

use crate::param_space::{self, ParamDef, ParamKind, ParamSpace, ParamVec};

pub const BASE_PARAM_DIM: usize = 54;
pub const HETEROGENEOUS_PARAM_DIM: usize = 62;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Role {
    Scout,
    Relay,
    Strike,
    Decoy,
}

impl Role {
    pub const ALL: [Role; 4] = [Role::Scout, Role::Relay, Role::Strike, Role::Decoy];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scout => "scout",
            Self::Relay => "relay",
            Self::Strike => "strike",
            Self::Decoy => "decoy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Echelon {
    Pair,
    Squad,
    Platoon,
}

impl Echelon {
    pub const ALL: [Echelon; 3] = [Echelon::Pair, Echelon::Squad, Echelon::Platoon];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pair => "pair",
            Self::Squad => "squad",
            Self::Platoon => "platoon",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleEchelonGain {
    pub role: Role,
    pub echelon: Echelon,
    pub exploration_gain: f64,
    pub coordination_gain: f64,
    pub relay_weight: f64,
    pub strike_weight: f64,
    pub deception_weight: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeterogeneousPolicy {
    pub gains: Vec<RoleEchelonGain>,
}

impl HeterogeneousPolicy {
    pub fn gain_for(&self, role: Role, echelon: Echelon) -> Option<&RoleEchelonGain> {
        self.gains
            .iter()
            .find(|gain| gain.role == role && gain.echelon == echelon)
    }

    pub fn adjust_aggression(&self, base: f64, role: Role, echelon: Echelon) -> f64 {
        self.gain_for(role, echelon)
            .map(|gain| base * gain.strike_weight)
            .unwrap_or(base)
    }
}

pub fn strix_heterogeneous() -> ParamSpace {
    let mut space = param_space::strix_full();
    let extras = vec![
        continuous("role_gain_scout", 0.6, 1.6, 1.15),
        continuous("role_gain_relay", 0.6, 1.6, 1.05),
        continuous("role_gain_strike", 0.6, 1.6, 1.20),
        continuous("role_gain_decoy", 0.6, 1.6, 1.10),
        continuous("echelon_gain_pair", 0.7, 1.5, 0.95),
        continuous("echelon_gain_squad", 0.7, 1.5, 1.05),
        continuous("echelon_gain_platoon", 0.7, 1.5, 1.15),
        continuous("heterogeneity_spread", 0.0, 0.5, 0.2),
    ];
    space.params.extend(extras);
    space
}

pub fn decode_heterogeneous_policy(params: &ParamVec) -> HeterogeneousPolicy {
    if params.len() < HETEROGENEOUS_PARAM_DIM {
        return HeterogeneousPolicy::default();
    }

    let role_gains = [params[54], params[55], params[56], params[57]];
    let echelon_gains = [params[58], params[59], params[60]];
    let spread = params[61].clamp(0.0, 0.5);

    let mut gains = Vec::new();
    for (role_index, role) in Role::ALL.iter().copied().enumerate() {
        for (echelon_index, echelon) in Echelon::ALL.iter().copied().enumerate() {
            let role_gain = role_gains[role_index];
            let echelon_gain = echelon_gains[echelon_index];
            let exploration_gain = clamp_gain(
                role_gain
                    * match role {
                        Role::Scout => 1.0 + spread * 0.8,
                        Role::Relay => 1.0 + spread * 0.2,
                        Role::Strike => 1.0 - spread * 0.2,
                        Role::Decoy => 1.0 + spread * 0.5,
                    },
            );
            let coordination_gain = clamp_gain(
                echelon_gain
                    * match echelon {
                        Echelon::Pair => 1.0 - spread * 0.1,
                        Echelon::Squad => 1.0 + spread * 0.15,
                        Echelon::Platoon => 1.0 + spread * 0.35,
                    },
            );
            let relay_weight = clamp_gain(
                role_gain
                    * match role {
                        Role::Relay => 1.25,
                        _ => 0.90,
                    }
                    * echelon_gain,
            );
            let strike_weight = clamp_gain(
                role_gain
                    * match role {
                        Role::Strike => 1.30,
                        Role::Scout => 0.85,
                        _ => 1.0,
                    }
                    * (1.0 + spread * 0.1),
            );
            let deception_weight = clamp_gain(
                role_gain
                    * match role {
                        Role::Decoy => 1.25,
                        _ => 0.95,
                    },
            );
            gains.push(RoleEchelonGain {
                role,
                echelon,
                exploration_gain,
                coordination_gain,
                relay_weight,
                strike_weight,
                deception_weight,
            });
        }
    }

    HeterogeneousPolicy { gains }
}

fn continuous(name: &'static str, min: f64, max: f64, default: f64) -> ParamDef {
    ParamDef {
        name,
        kind: ParamKind::Continuous { min, max },
        default,
    }
}

fn clamp_gain(value: f64) -> f64 {
    value.clamp(0.5, 1.8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heterogeneous_space_extends_base_space() {
        let base = param_space::strix_full();
        let hetero = strix_heterogeneous();
        assert_eq!(base.len(), BASE_PARAM_DIM);
        assert_eq!(hetero.len(), HETEROGENEOUS_PARAM_DIM);
    }

    #[test]
    fn decode_policy_produces_all_role_echelon_pairs() {
        let params = strix_heterogeneous().defaults();
        let policy = decode_heterogeneous_policy(&params);
        assert_eq!(policy.gains.len(), Role::ALL.len() * Echelon::ALL.len());
    }

    #[test]
    fn strike_roles_get_higher_aggression_than_scouts() {
        let params = strix_heterogeneous().defaults();
        let policy = decode_heterogeneous_policy(&params);
        let strike = policy.gain_for(Role::Strike, Echelon::Squad).unwrap();
        let scout = policy.gain_for(Role::Scout, Echelon::Squad).unwrap();
        assert!(strike.strike_weight > scout.strike_weight);
    }
}
