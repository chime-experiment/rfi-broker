<h1 align="center">CHIME rfi receiver</h1>

Deals with a few tasks related to RFI flagging:
- Provides a per-feed likelihood that the feed is corrupted using spectral kurtosis data
- Enables and disables RFI zeroing around solar noon
- Exports some useful Prometheus metrics (TBD)

This is intended to run on an auxiliary node. It is tightly coupled to the data being sent from
kotekan - namely, the packet structure defined in
[kotekan](https://github.com/kotekan/kotekan/blob/chord/lib/utils/rfi_functions.h#L14)

# Installation
Install using [cargo](https://github.com/rust-lang/cargo).
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
$ cargo build [--release] [--profile <profile>]
$ cargo run [--release] [--profile <profile>]
```

## Profiles
The following profiles are included:
- `debug`
- `release-with-debug-info`
- `release`

# Running
```
$ [RUST_LOG=<log_level>] ./path/to/binary --addr <http address> --udp_addr <udp recv address> [--config <path to config>] [--threads <num threads>]
```

If bulding from source, you can also just use `cargo run -- [args]`.

## Logging
Log level is controlled by the `RUST_LOG` environment variable, and defaults to `INFO`. However, `DEBUG` statements (and below)
are entirely remove in a `release` build.

## Endpoints
The following endpoints are exposed in all build profiles:
- `/metadata`: most recent packet header
- `/metrics`: prometheus metrics
- `/human-metrics`: basic human-readable metrics
- `/bad_input_likelihood`: per-input metric containing the likelihood of the input being corrupted
- `/`: prints `bad_input_likelihood`

Debug-only:
- `/last-frame`: prints the most recent frame in each ringbuffer, and buffer length and shape
- `/write-buffers`: POST - writes both spectral kurtosis buffers and masks to a provided directory

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

# Metrics
## Bad Input Likelihood
Represents the likelihood that an input is "bad", based on per-input median-absolute-deviations
excursion measurements. Represented as a percentage (0 - 100) using the base endpoint (`/`) and a
0 - 1 scale when calling the `/bad_input_likelihood` endpoint.

### How is it computed?
Each packet includes a per-feed, per-frequency count of the number of kotekan frames in which that
input exceeded a `N`-sigma median-absolute-deviations test, computed across inputs for each frequecy
independently, using a per-input spectral kurtosis metric. Thus, these counts range from 0 to the number
of kotekan frames accumulated per packet.

1. Sum counts across frequencies to produce a per-element trial success count.
2. Compute likelihood of a feed being bad based on the CDF of a Poisson distribution for the number of
trials (num_freq * num_frames_per_packet) and whatever `N` sigma value was used by the MAD test.
3. Feed this likelihood into a Beta distribution, which acts as a ramp function to suppress high likelihoods
produced by a handful of excursions (since `p` is small and `n` is large, a dozen or so excursions results in
a likelihood of ~0.5 from the Poisson CDF alone, which may not be representative of the metric that we want
to produce).
4. Feed the likelihood in an exponentially-weighted moving average with a 64-sample lookback period in order
to smooth over short-duration broad-spectrum contamination.
