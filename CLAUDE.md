# Project Instructions

## Running tests

Run the test suite via offload, which parallelizes execution across Modal cloud sandboxes:

```bash
./scripts/offload-tests.sh
```

Or directly:

```bash
offload run --copy-dir ".:/app"
```

Prerequisites: offload (`cargo install offload@0.3.0`) and Modal credentials (`modal token new`).
