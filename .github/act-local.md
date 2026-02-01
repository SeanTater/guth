# Running GitHub Actions Locally with `act`

## Installation

```bash
# Arch Linux
sudo pacman -S act

# macOS
brew install act

# Nix
nix-env -iA nixpkgs.act

# Go
go install github.com/nektos/act@latest

# Direct download
curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

## Quick Start

```bash
# List available workflows and jobs
act -l

# Run a specific workflow (dry-run list jobs)
act -l -W .github/workflows/release.yml

# Run a specific job
act -j build-deb-x86_64

# Run workflow_dispatch workflows (manual triggers)
act workflow_dispatch

# Run with dry-run input
act workflow_dispatch --input dry_run=true
```

## Common Commands

### Test Package Builds

```bash
# Build and test .deb package (x86_64)
act -j build-deb-x86_64 -j test-deb-x86_64

# Build and test .rpm package
act -j build-rpm -j test-rpm-x86_64

# Build Docker image
act -j build-and-test -W .github/workflows/container.yml
```

### Test Binary Releases

```bash
# Build Linux x86_64 binary
act -j build-linux -W .github/workflows/release.yml

# Note: macOS and Windows jobs won't work locally (need native runners)
```

## Image Selection

`act` uses Docker images to simulate GitHub runners. On first run, it will ask which image size to use:

- **Micro** (~200MB): Minimal, may miss some tools
- **Medium** (~500MB): Good balance, recommended for most workflows
- **Large** (~17GB): Full GitHub runner environment, slow to download

For Rust workflows, medium usually works. If you hit missing tool errors, try:

```bash
act -P ubuntu-latest=catthehacker/ubuntu:act-latest
```

## Limitations

- macOS jobs (`macos-*`) cannot run locally
- Windows jobs (`windows-*`) cannot run locally
- Some GitHub-specific features (secrets, GITHUB_TOKEN) need manual setup
- Container jobs may have networking differences

## Secrets

For workflows needing secrets:

```bash
# Create a .secrets file (add to .gitignore!)
echo "CARGO_REGISTRY_TOKEN=your_token" > .secrets

# Run with secrets
act --secret-file .secrets
```

## Debugging

```bash
# Verbose output
act -v

# Very verbose
act -vv

# Keep containers running after failure for inspection
act --reuse
```
