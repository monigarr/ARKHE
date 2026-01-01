# Release Management Guide

This guide explains how to manage releases for the ARKHE Framework using the automated release tools.

## Overview

The ARKHE Framework uses semantic versioning (MAJOR.MINOR.PATCH) and automated release scripts to manage versioning and tagging.

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Current version: **0.1.0**

## Release Scripts

### `scripts/release.py`

Main release management script that handles:
- Version bumping
- Updating version in source files
- Creating git tags
- CHANGELOG validation

#### Usage Examples

**Bump patch version (0.1.0 → 0.1.1):**
```bash
python scripts/release.py --bump patch
```

**Bump minor version (0.1.0 → 0.2.0):**
```bash
python scripts/release.py --bump minor
```

**Bump major version (0.1.0 → 1.0.0):**
```bash
python scripts/release.py --bump major
```

**Set specific version:**
```bash
python scripts/release.py --version 0.2.0
```

**Create tag only (no version bump):**
```bash
python scripts/release.py --tag-only --version 0.1.0
```

**Skip CHANGELOG validation:**
```bash
python scripts/release.py --bump patch --skip-validation
```

**Skip git tag creation:**
```bash
python scripts/release.py --bump patch --skip-tag
```

**Custom tag message:**
```bash
python scripts/release.py --bump patch --message "Hotfix release"
```

### `scripts/generate_release_notes.py`

Generates formatted release notes from CHANGELOG.md.

#### Usage Examples

**Generate release notes for version 0.1.2:**
```bash
python scripts/generate_release_notes.py 0.1.2
```

**Save to file:**
```bash
python scripts/generate_release_notes.py 0.1.2 --output RELEASE_NOTES.md
```

## Release Workflow

### Standard Release Process

1. **Update CHANGELOG.md**
   - Add a new section for the release version
   - Document all changes (Added, Changed, Fixed, Removed, etc.)
   - Follow the format in existing changelog entries

2. **Run Release Script**
   ```bash
   python scripts/release.py --bump patch
   ```
   This will:
   - Update version in `src/math_research/__init__.py`
   - Update version in `src/apps/cli/__init__.py`
   - Validate CHANGELOG.md has an entry for the new version
   - Create a git tag (e.g., `v0.1.1`)

3. **Review Changes**
   ```bash
   git diff
   ```

4. **Commit Version Updates**
   ```bash
   git add .
   git commit -m "chore: bump version to 0.1.1"
   ```

5. **Push Changes and Tags**
   ```bash
   git push origin main
   git push origin v0.1.1
   ```

6. **Generate Release Notes (Optional)**
   ```bash
   python scripts/generate_release_notes.py 0.1.1 --output RELEASE_NOTES.md
   ```

7. **Create GitHub Release (Manual)**
   - Go to GitHub repository → Releases → Draft a new release
   - Tag: `v0.1.1`
   - Title: `ARKHE Framework v0.1.1`
   - Description: Copy from `RELEASE_NOTES.md` or CHANGELOG.md

### Hotfix Release

For urgent bug fixes:

```bash
# Create hotfix branch
git checkout -b hotfix/0.1.1

# Make fixes and commit
git add .
git commit -m "fix: critical bug description"

# Update CHANGELOG.md
# Run release script
python scripts/release.py --bump patch

# Commit and push
git add .
git commit -m "chore: bump version to 0.1.1"
git push origin hotfix/0.1.1

# Create PR and merge to main
# Then push tag
git push origin v0.1.1
```

## Version Files

The following files contain version information and are automatically updated:

- `src/math_research/__init__.py` - Main package version
- `src/apps/cli/__init__.py` - CLI version

## Git Tags

All releases are tagged with the format `v{version}` (e.g., `v0.1.0`, `v0.1.1`).

**List all tags:**
```bash
git tag -l
```

**View tag details:**
```bash
git show v0.1.0
```

**Delete a tag (if needed):**
```bash
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0
```

## CHANGELOG.md Format

The CHANGELOG.md follows [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [0.1.1] - 2025-01-09

### Added
- New feature description

### Changed
- Change description

### Fixed
- Bug fix description

### Removed
- Removed feature description
```

## Best Practices

1. **Always update CHANGELOG.md before releasing**
   - Document all user-facing changes
   - Group changes by type (Added, Changed, Fixed, etc.)

2. **Use semantic versioning correctly**
   - PATCH: Bug fixes, minor improvements
   - MINOR: New features, backward-compatible changes
   - MAJOR: Breaking changes, major refactoring

3. **Test before releasing**
   - Run all tests: `pytest`
   - Verify installation: `pip install -e .`
   - Test CLI: `python -m src.apps.cli --version`

4. **Create meaningful tag messages**
   - Use `--message` flag for descriptive tag messages
   - Include key highlights of the release

5. **Keep working directory clean**
   - Commit or stash changes before running release script
   - The script will warn if there are uncommitted changes

## Troubleshooting

### "Version not found in CHANGELOG.md"
- Make sure you've added a section for the new version in CHANGELOG.md
- Use `--skip-validation` to bypass this check (not recommended)

### "Tag already exists"
- The tag already exists in the repository
- Use `git tag -l` to list existing tags
- Delete the tag if it was created incorrectly

### "Working directory is not clean"
- Commit or stash your changes before running the release script
- The script requires a clean working directory to prevent accidental commits

## Related Documentation

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [CHANGELOG.md](../CHANGELOG.md)

