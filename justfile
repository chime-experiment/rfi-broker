set shell := ["bash", "-eo", "pipefail", "-c"]

base_dir := justfile_directory()

# list all available bindings (default)
default:
    just --list

# --- Checks ---

# lint the entire workspace with clippy
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# run all workspace tests
test:
    cargo nextest run --profile ci --no-tests="warn"

# --- CI ---

ci-clippy: clippy
    cargo clippy --all-targets --all-features --release -- -D warnings

# also test in release build for CI
ci-test: test
    cargo nextest run --profile ci --no-tests="warn" --release

# --- Docs ---

# build all workspace docs
doc:
    cargo doc --release --no-deps --document-private-items

# build all workspace docs and open
doc-open: doc
    open "target/doc/rfi_broker/index.html"

# build all workspace docs and modify build for GitHub pages
doc-hosted: doc
    echo "<meta http-equiv=\"refresh\" content=\"0; url=rfi_broker/index.html\">" > "target/doc/index.html"
    touch "target/doc/.nojekyll"
