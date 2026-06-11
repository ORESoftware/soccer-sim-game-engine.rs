//! # soccer-sim-game-engine
//!
//! The soccer 2D sim/game domain, layered on the generic
//! [`des_engine`](::des_engine) (`discrete-event-system.rs`) which supplies the
//! optimization (LP simplex/dual/IPM/Clarabel) and learning (neural nets,
//! MDP/POMDP, evolution) machinery the soccer engine drives.
//!
//! ## Dependency direction
//!
//! ```text
//! discrete-event-system.rs (des_engine)   generic DES + optimization + learning
//!         ▲ depends on
//! soccer-sim-game-engine.rs (this crate)  soccer domain: match engine, learning, planner
//!         ▲ depends on
//!   dd-des-rs / dd-soccer-rs servers       run games (uuid-keyed) + serve the UIs
//! ```
//!
//! ## Transport-agnostic by design
//!
//! This crate is a pure **domain engine** — it has **no HTTP server**. It is
//! imported both by the Rust web servers (`dd-soccer-rs`, `dd-des-rs`) *and* by
//! a desktop gaming system, so it must stay transport-agnostic. Consumers drive
//! the engine through the plain session API ([`soccer::SoccerRealtimeSession`]):
//! step the match, apply inputs, read a frame/snapshot — no sockets involved.
//!
//! The HTTP glue (`SoccerLiveHttpBridge`, which translates method/path/body into
//! session calls) is a **server concern**, not part of the agnostic engine. It
//! currently still lives inside `des_engine::des::general::soccer` for
//! historical reasons; the extraction **relocates it into the server layer**
//! (or a thin `soccer-realtime-http` shim) so the engine a desktop links against
//! carries zero web dependencies. Until then the servers reach it transitionally
//! via [`des_engine`] directly — never presented as part of this crate's API.
//!
//! ## Migration note (phase 1)
//!
//! The soccer source still physically lives inside `des_engine`. This crate
//! currently **re-exports** those modules so dependents can target a *stable*
//! `soccer_sim_game_engine::*` API today. The physical code move (engine → here)
//! happens later, behind the repo-automation freeze, **without changing this
//! public surface**: the `pub use` lines below become `pub mod` definitions
//! owning the code, and the HTTP bridge is left behind in the server layer.

/// Re-export of the generic engine, for code that needs the optimization /
/// learning primitives directly (LP, neural nets, MDP/POMDP).
pub use des_engine;

/// The soccer match engine: `SoccerMatch`, the live HTTP bridge
/// (`SoccerLiveHttpBridge`), playback-artifact rendering, formation LP, neural
/// value model, moment embeddings, set-piece training, and the gameplay model.
pub mod soccer {
    pub use des_engine::des::general::soccer::*;
}

/// Squad-rotation scheduling for soccer (the playing-time rotation problem).
pub mod soccer_rotation {
    pub use des_engine::des::general::soccer_rotation::*;
}

/// The 11-a-side rotation/formation planner (IP/MIP solved via the engine).
pub mod soccer_planner {
    pub use des_engine::des::soccer_planner::*;
}

/// Learning glue: Q-policies, tactical/neural evolution, completed-game scoring,
/// and the moment-embedding types persisted for retrieval.
pub mod soccer_learning {
    pub use des_engine::des::soccer_learning::*;
}

/// Postgres persistence + pgvector moment-embedding RAG store for soccer
/// learning (policy versions, completed runs, set-play artifacts, embeddings).
pub mod soccer_learning_pg {
    pub use des_engine::des::soccer_learning_pg::*;
}
