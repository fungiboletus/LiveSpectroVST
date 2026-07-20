# Contributing

Use Rust 1.92 or newer. Before opening a pull request, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo xtask bundle live-spectro-vst --release
```

The plugin crate forbids unsafe Rust. Keep audio-thread processing allocation
free and avoid blocking synchronization in `process()`.

Do not change the VST3 class ID, CLAP ID, or parameter IDs without a migration
plan because existing DAW sessions depend on them.
