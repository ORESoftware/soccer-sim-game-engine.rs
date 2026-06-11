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
//! This crate is a pure **domain engine** — **no HTTP server**. It is imported
//! both by the Rust web servers (`dd-soccer-rs`, `dd-des-rs`) *and* by a desktop
//! gaming system, so consumers drive it through the plain session API
//! ([`soccer::SoccerRealtimeSession`]) — step the match, apply inputs, read a
//! frame — with no sockets involved. (The remaining HTTP glue still embedded in
//! `soccer` is slated to move into the server layer so this engine carries zero
//! web dependencies.)
//!
//! ## Status: owns the soccer code (extraction in progress)
//!
//! These modules are the soccer source **moved out of `des_engine`**. The
//! engine-side deletion + registry-reference cuts are what remove the dangling
//! soccer refs from `discrete-event-system.rs`; see the crate-split memory.

/// Re-export of the generic engine, for code that needs the optimization /
/// learning primitives directly (LP, neural nets, MDP/POMDP, DES stations).
pub use des_engine;

/// The soccer match engine: `SoccerMatch`, the realtime session, playback-
/// artifact rendering, formation LP, neural value model, moment embeddings,
/// set-piece training, and the gameplay model.
pub mod soccer;

/// Squad-rotation scheduling for soccer (the playing-time rotation problem).
pub mod soccer_rotation;

/// The 11-a-side rotation/formation planner (IP/MIP solved via the engine).
pub mod soccer_planner;

/// Learning glue: Q-policies, tactical/neural evolution, completed-game scoring,
/// and the moment-embedding types persisted for retrieval.
pub mod soccer_learning;

/// Postgres persistence + pgvector moment-embedding RAG store for soccer
/// learning (policy versions, completed runs, set-play artifacts, embeddings).
pub mod soccer_learning_pg;
