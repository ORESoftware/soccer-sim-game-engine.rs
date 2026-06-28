//! Headless A/B harness for the quick-release forward-pass work.
//!
//! Runs full deterministic matches and measures PASSING TEMPO: how long carriers
//! dwell on the ball before releasing, how often they release a forward pass when an
//! OPEN FORWARD option is already on, and how the action mix shifts. Run it twice with
//! identical seeds — once with the gates unset (baseline) and once with them set — and
//! diff the output:
//!
//!   cargo run --release --bin measure_quick_release -- [ticks] [seeds]
//!   DD_SOCCER_ENABLE_QUICK_RELEASE_PASS_BIAS=1 \
//!   DD_SOCCER_ENABLE_QUICK_RELEASE_PASS_REWARD=1 \
//!     cargo run --release --bin measure_quick_release -- [ticks] [seeds]

use soccer_engine::des::general::soccer::{
    enable_deterministic_formation_lp, MatchConfig, SoccerMatch,
};

// "Open forward option exists" threshold — mirrors the decision-bias gate
// (QUICK_RELEASE_BIAS_MIN_OPENNESS = 0.45).
const OPEN_FWD_OPENNESS: f64 = 0.45;

fn is_floor_pass_release(action: &str) -> bool {
    action.starts_with("pass") // ranked floor passes pass1/pass2/pass3
        || action == "first-time-pass"
        || action == "killer-pass"
        || action == "surprise-pass"
        || action == "wall-pass"
}

fn is_hold_or_dribble(action: &str) -> bool {
    matches!(
        action,
        "dribble"
            | "carry-forward"
            | "protect-ball"
            | "side-step"
            | "left-cut"
            | "right-cut"
            | "nutmeg"
            | "fake-left-cut-right"
            | "fake-right-cut-left"
            | "shield"
            | "hold"
    )
}

fn main() {
    enable_deterministic_formation_lp();
    let args: Vec<String> = std::env::args().collect();
    let ticks: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    let bias = std::env::var("DD_SOCCER_ENABLE_QUICK_RELEASE_PASS_BIAS").is_ok();
    let reward = std::env::var("DD_SOCCER_ENABLE_QUICK_RELEASE_PASS_REWARD").is_ok();

    let mut on_ball: u64 = 0;
    let mut floor_pass_releases: u64 = 0;
    let mut first_time_passes: u64 = 0;
    let mut dribble_holds: u64 = 0;

    // Hold time (actual_time_on_ball_seconds) at the moment of a floor-pass release.
    let mut release_hold_sum: f64 = 0.0;

    // "Open forward option exists" situations and what the carrier did.
    let mut open_fwd: u64 = 0;
    let mut open_fwd_released: u64 = 0;
    let mut open_fwd_held: u64 = 0;
    let mut open_fwd_hold_sum: f64 = 0.0;

    let mut goals_home: u32 = 0;
    let mut goals_away: u32 = 0;

    for s in 0..seeds {
        let config = MatchConfig {
            seed: 0x5EED_0000u32.wrapping_add(s as u32),
            ..MatchConfig::default()
        };
        let mut sim = SoccerMatch::default_11v11(config);
        for _ in 0..ticks {
            sim.run_time_step();
            for p in sim.players.iter() {
                let Some(d) = p.last_decision.as_ref() else {
                    continue;
                };
                if !d.observation.has_ball {
                    continue;
                }
                on_ball += 1;
                let action = d.action.as_str();
                let hold = d.observation.actual_time_on_ball_seconds.max(0.0);

                if is_floor_pass_release(action) {
                    floor_pass_releases += 1;
                    release_hold_sum += hold;
                }
                if action == "first-time-pass" {
                    first_time_passes += 1;
                }
                if is_hold_or_dribble(action) {
                    dribble_holds += 1;
                }

                if d.observation.best_forward_pass_receiver_openness >= OPEN_FWD_OPENNESS {
                    open_fwd += 1;
                    open_fwd_hold_sum += hold;
                    if is_floor_pass_release(action) {
                        open_fwd_released += 1;
                    } else if is_hold_or_dribble(action) {
                        open_fwd_held += 1;
                    }
                }
            }
        }
        goals_home += sim.score_home;
        goals_away += sim.score_away;
    }

    let pct = |n: u64, d: u64| 100.0 * n as f64 / d.max(1) as f64;
    let mean = |sum: f64, d: u64| sum / d.max(1) as f64;

    println!("===== QUICK-RELEASE TEMPO ({seeds} matches x {ticks} ticks) =====");
    println!(
        "gates: BIAS={} REWARD={}   aggregate score {goals_home}-{goals_away}",
        bias, reward
    );
    println!("on-ball decisions:            {on_ball}");
    println!(
        "floor-pass releases:          {floor_pass_releases}  ({:.3}% of on-ball)",
        pct(floor_pass_releases, on_ball)
    );
    println!(
        "  mean hold @ release:        {:.4} s   <-- lower = released quicker",
        mean(release_hold_sum, floor_pass_releases)
    );
    println!(
        "first-time (one-touch) passes:{first_time_passes}  ({:.3}% of on-ball)",
        pct(first_time_passes, on_ball)
    );
    println!(
        "dribble/hold decisions:       {dribble_holds}  ({:.3}% of on-ball)",
        pct(dribble_holds, on_ball)
    );
    println!("--- when an OPEN FORWARD option exists (openness >= {OPEN_FWD_OPENNESS}) ---");
    println!(
        "open-forward situations:      {open_fwd}  ({:.3}% of on-ball)",
        pct(open_fwd, on_ball)
    );
    println!(
        "  RELEASED a forward pass:    {open_fwd_released}  ({:.3}% of open-fwd)  <-- higher = \"pass already!\"",
        pct(open_fwd_released, open_fwd)
    );
    println!(
        "  held / dribbled instead:    {open_fwd_held}  ({:.3}% of open-fwd)",
        pct(open_fwd_held, open_fwd)
    );
    println!(
        "  mean hold in open-fwd:      {:.4} s   <-- lower = less dwelling",
        mean(open_fwd_hold_sum, open_fwd)
    );
}
