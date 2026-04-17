<h1 align="center">CHIME rfi receiver</h1>

Deals with a few tasks related to RFI flagging:
- Provides a per-feed likelihood that the feed is corrupted using per-feed SK data
- Enables and disables RFI zeroing around solar noon
- Exports some useful Prometheus metrics

This is intended to run on an auxiliary node. It is tightly coupled to the data being sent from
kotekan - namely, the packet structure defined in
[kotekan](https://github.com/kotekan/kotekan/blob/chord/lib/utils/rfi_functions.h#L14)

# Installation
Installing is easy using [cargo](https://github.com/rust-lang/cargo).
```
$ cargo install --git https://github.com/ljgray/rfi-receiver.git [--branch main] [--tag v1.0.0]
```
Installing to a system path (which is probably what you want) is also easy. Just note that
cargo automatically appends `bin/` to the `--root` path.
```
$ [sudo] cargo install --git https://github.com/ljgray/rfi-receiver.git [<version args>] --root /usr/local [--locked]
```
`cargo install` builds in `--release` mode by default. You can choose to use dependency versions
specified in `Cargo.lock` by including the `--locked` argument.

## Build from source
```
$ git clone https://github.com/ljgray/rfi-receiver.git
$ cargo build [--release]
$ cargo run [--release]
```

# Running
```
$ ./path/to/binary --addr <http address> --udp_addr <udp recv address> [--config <path to config>] [--threads <num threads>]
```

If bulding from source, you can also just use `cargo run -- [args]`.

# Tests
Unit tests can be run using
```
$ cargo test
```
or
```
$ cargo nextest run
```
if using [nextest](https://nexte.st/). Integration testing is a WIP.
