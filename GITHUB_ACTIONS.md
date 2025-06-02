# GitHub Actions Setup

## Badges for README

Replace `USERNAME` and `REPOSITORY` in the README badges with your actual GitHub username and repository name:

```markdown
[![Tests](https://github.com/USERNAME/REPOSITORY/workflows/Tests/badge.svg)](https://github.com/USERNAME/REPOSITORY/actions)
[![CI/CD](https://github.com/USERNAME/REPOSITORY/workflows/CI%2FCD/badge.svg)](https://github.com/USERNAME/REPOSITORY/actions)
[![codecov](https://codecov.io/gh/USERNAME/REPOSITORY/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/REPOSITORY)
```

## Example for common repository names:

If your GitHub username is `johndoe` and repository is `minesweeper-ai`:

```markdown
[![Tests](https://github.com/johndoe/minesweeper-ai/workflows/Tests/badge.svg)](https://github.com/johndoe/minesweeper-ai/actions)
[![CI/CD](https://github.com/johndoe/minesweeper-ai/workflows/CI%2FCD/badge.svg)](https://github.com/johndoe/minesweeper-ai/actions)
[![codecov](https://codecov.io/gh/johndoe/minesweeper-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/johndoe/minesweeper-ai)
```

## Workflows Created

1. **`.github/workflows/tests.yml`** - Basic test runner
   - Runs on Python 3.8-3.13
   - Executes all tests on push/PR
   - Generates coverage report (Python 3.13 only)
   - Uploads to Codecov

2. **`.github/workflows/ci.yml`** - Comprehensive CI/CD
   - Includes linting
   - Coverage artifacts
   - Build verification
   - Import testing

## Setting up Codecov (Optional)

1. Go to [codecov.io](https://codecov.io)
2. Sign in with GitHub
3. Add your repository
4. Copy the upload token (if private repo)
5. Add as GitHub secret: `CODECOV_TOKEN`

## Branch Protection (Recommended)

In your GitHub repository settings:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Require status checks: "test (3.13)"
4. Require branches to be up to date
