# Installation Modernization Design

**Date**: 2026-03-31  
**Goal**: Fix immediate installation issues and modernize dependency management by moving from requirements.txt to pyproject.toml-only configuration.

## Problem Statement

The pirlygenes package has installation issues that prevent clean installation in fresh environments:

1. **Dependency specification errors**: Typo in numpy version (`1.26.09` should be `1.26.0`)
2. **Version constraint conflicts**: `pandas<=2.2.1` blocks modern pandas 3.0+ installations
3. **Missing dependencies**: Untracked plotting dependencies (seaborn, matplotlib) required by imports
4. **Dual dependency management**: Both requirements.txt and pyproject.toml dependency specifications
5. **pyensembl fork dependency**: Package relies on a fork with setuptools fixes (PR #322)

## Solution: Conservative Migration Approach

### Dependency Management Strategy

**Remove requirements.txt entirely** and consolidate all dependencies in pyproject.toml:

```toml
[project]
dependencies = [
    "numpy>=1.24.0,<2.0",    # Fix typo, support modern versions
    "pandas>=2.0.0",         # Remove restrictive upper bound
    "pyensembl>=2.4.0",      # Baseline for fork compatibility  
    "argh",                  # CLI framework
    "openpyxl",              # Excel file support
    # Additional plotting dependencies discovered in testing:
    "matplotlib>=3.5.0",     # Core plotting
    "seaborn>=0.11.0",       # Statistical plotting
]
```

**Version bound rationale**:
- numpy: Lower bound ensures modern features, upper bound prevents major breaking changes
- pandas: No upper bound to support pandas 3.0+ while maintaining 2.0+ compatibility
- pyensembl: Minimum version aligns with fork baseline

### pyproject.toml Configuration Updates

**Remove dynamic dependencies**:
```toml
# Remove this section:
[tool.setuptools.dynamic]
dependencies = {file = ["requirements.txt"]}
```

**Update project configuration**:
```toml
[project]
name = "pirlygenes"
requires-python = ">=3.7"
dynamic = ["version"]  # Only version remains dynamic
dependencies = [
    # Static dependency list as above
]
```

**Preserve existing configuration**:
- Keep `[tool.setuptools.dynamic]` version specification
- Maintain `[tool.setuptools.packages]` and package data configuration
- Preserve build system and project metadata

### Testing Strategy

**Fresh environment validation**:
1. Create clean virtual environment
2. Install package from updated pyproject.toml only
3. Test critical imports: `from pirlygenes.gene_sets_cancer import CTA_gene_names`
4. Run CLI commands to verify plotting dependencies
5. Iterate dependency list based on import failures

**Success criteria**:
- Clean installation in fresh venv without missing dependency errors
- All documented imports work correctly
- CLI plotting functionality works without additional manual installs

### pyensembl Fork Handling

**Short-term approach** (until PR #322 merges):
- Specify `pyensembl>=2.4.0` in dependencies
- Add installation note in README about potential need for fork
- Avoid git+ URLs to keep installation simple

**Long-term**: Once upstream PR merges, no changes needed - version constraint will work automatically.

## Implementation Plan

### Phase 1: Update Configuration
1. Update pyproject.toml with new dependencies section
2. Remove dynamic dependencies configuration  
3. Delete requirements.txt

### Phase 2: Fresh Environment Testing
1. Create isolated test environment
2. Install and test iteratively
3. Add missing dependencies discovered during testing

### Phase 3: Validation
1. Test key import commands
2. Verify CLI functionality
3. Test installation on multiple Python versions (3.7+)

## Risk Assessment

**Low risk changes**:
- Removing requirements.txt (redundant with pyproject.toml)
- Fixing numpy version typo
- Removing pandas upper bound

**Moderate risk**:
- pyensembl version specification (depends on fork until PR merges)
- Discovering all implicit dependencies through testing

**Mitigation**:
- Conservative version bounds
- Thorough testing in fresh environments
- Clear documentation of pyensembl fork requirement if needed