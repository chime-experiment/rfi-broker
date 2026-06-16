<h1 align="center">CHIME rfi broker</h1>

Deals with a few tasks related to RFI flagging:
- Provides a per-feed likelihood that the feed is corrupted based on spectral kurtosis data
- Enables and disables RFI zeroing around solar noon
- Exports some useful Prometheus metrics

This is intended to run on an auxiliary node. It is tightly coupled to the data being sent from
kotekan - namely, the packet structure defined in
[kotekan](https://github.com/kotekan/kotekan/blob/chord/lib/utils/rfi_functions.h#L14)

# Installation
Install using [cargo](https://github.com/rust-lang/cargo).
```
$ cargo install --git https://github.com/chime-experiment/rfi-broker.git [--branch main] [--tag v1.0.0]
```
Installing to a system path (which is probably what you want) is also straightforward. Note that
cargo automatically appends `bin/` to the `--root` path.
```
$ [sudo] cargo install --git https://github.com/chime-experiment/rfi-broker.git [<version args>] --root /usr/local [--locked]
```
`cargo install` builds in `--release` mode by default. You can choose to use dependency versions
specified in `Cargo.lock` by including the `--locked` argument.

## Build from source
```
$ git clone https://github.com/ljgray/rfi-broker.git
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

If building from source, you can also just use `cargo run -- [args]`.

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
Represents the likelihood that an input is "bad", based on per-input spectral kurtosis integrated at
a slowish cadence (<1 second). Represented as a percentage (0 - 100) using the base endpoint (`/`) and a
0 - 1 scale when calling the `/bad_input_likelihood` endpoint.

### How is it computed?
Each packet includes a per-feed, per-frequency spectral kurtosis (SK) value, integrated for some number of
kotekan frames.

Each sample is centred according to the median across feeds, normalized by the standard deviation over feeds,
then converted to a p-value. p-values are reduced across frequencies using Fisher's method, and a likelihood
is produced from the CDF of the resulting Fisher test statistic with chi-squared distribution.

Per-sample likelihood is fed into an exponentially-weighted moving average with a 32-sample lookback period
in order to smooth over short-duration broad-spectrum contamination.

## Prometheus
The following metrics are exported to [Prometheus](https://prometheus.io/) via the `/metrics` endpoint:
- `rfibroker_bad_input_likelihood`: input likelihood defined above. Labels: `[feed_index]`
- `rfibroker_frac_flagged`: fraction of flagged samples per frequency. Labels: `[freq_id]`
- `rfibroker_sktilde_avg`: feed-averaged spectral kurtosis accumulated over the integration period. Labels: `[freq_id]`
- `rfibroker_rfi_zeroing_first_stage_enabled`: whether or not first-stage zeroing should currently be set
- `rfibroker_rfi_zeroing_second_stage_enabled`: whether or not second-stage zeroing should currently be set
- `rfibroker_packets_received_total`: count of packets received from `kotekan`. Does *not* include OS-level losses
- `rfibroker_packets_dropped_total`: count of packets dropped within the broker. does *not* include OS-level losses
