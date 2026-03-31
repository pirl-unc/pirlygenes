# Installation Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix immediate installation issues and modernize dependency management by migrating from requirements.txt to pyproject.toml-only configuration.

**Architecture:** Conservative migration approach - update pyproject.toml with static dependencies, remove requirements.txt, and validate in fresh environment to discover missing plotting dependencies.

**Tech Stack:** Python packaging (setuptools, pyproject.toml), dependency management, virtual environments

---

## File Structure

**Files to modify:**
- `pyproject.toml` - Add static dependencies, remove dynamic dependency configuration
- `README.md` - Add installation notes about pyensembl fork requirement

**Files to delete:**
- `requirements.txt` - Consolidate into pyproject.toml

**Files to create:**
- `test_fresh_install.py` - Temporary test script for validation

### Task 1: Update pyproject.toml Configuration

**Files:**
- Modify: `pyproject.toml:15-19`

- [ ] **Step 1: Remove dynamic dependencies configuration**

Replace the dynamic dependencies line in pyproject.toml:

```toml
# Change from:
dynamic = ["version", "dependencies"]

# To:
dynamic = ["version"]
```

- [ ] **Step 2: Remove setuptools.dynamic dependencies section**

Remove these lines from pyproject.toml:

```toml
[tool.setuptools.dynamic]
version = {attr = "pirlygenes.version.__version__"}
# Remove this line:
# dependencies = {file = ["requirements.txt"]}
```

Keep only:

```toml
[tool.setuptools.dynamic]
version = {attr = "pirlygenes.version.__version__"}
```

- [ ] **Step 3: Add static dependencies to project section**

Add dependencies array to the [project] section after line 14:

```toml
[project]
name = "pirlygenes"
requires-python = ">=3.7"
authors = [ {name="Alex Rubinsteyn", email="alex.rubinsteyn@unc.edu" } ]
description = "Gene lists for cancer immunotherapy expression analysis"
classifiers = [
    "Programming Language :: Python :: Implementation :: CPython",
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
]
readme = "README.md"
dynamic = ["version"]
dependencies = [
    "numpy>=1.24.0,<2.0",
    "pandas>=2.0.0",
    "pyensembl>=2.4.0",
    "argh",
    "openpyxl",
]
```

- [ ] **Step 4: Verify pyproject.toml syntax**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`
Expected: No error output

- [ ] **Step 5: Commit configuration changes**

```bash
git add pyproject.toml
git commit -m "Update pyproject.toml: migrate to static dependencies"
```

### Task 2: Remove requirements.txt

**Files:**
- Delete: `requirements.txt`

- [ ] **Step 1: Backup current requirements.txt content**

Run: `cat requirements.txt`
Expected output to verify:
```
numpy<=1.26.09
pandas<=2.2.1
pyensembl
argh
openpyxl
```

- [ ] **Step 2: Delete requirements.txt**

Run: `rm requirements.txt`

- [ ] **Step 3: Verify file is deleted**

Run: `ls requirements.txt`
Expected: `ls: requirements.txt: No such file or directory`

- [ ] **Step 4: Commit deletion**

```bash
git add -A
git commit -m "Remove requirements.txt - dependencies now in pyproject.toml"
```

### Task 3: Test Installation in Fresh Environment

**Files:**
- Create: `test_fresh_install.py`

- [ ] **Step 1: Create test script for import validation**

```python
#!/usr/bin/env python3
"""
Test script to validate pirlygenes installation and imports in fresh environment.
"""

def test_basic_imports():
    """Test core module imports."""
    try:
        import pirlygenes
        print("✓ pirlygenes imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pirlygenes: {e}")
        return False
    return True

def test_gene_sets_import():
    """Test gene sets module imports."""
    try:
        from pirlygenes.gene_sets_cancer import CTA_gene_names
        print("✓ CTA_gene_names imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CTA_gene_names: {e}")
        return False
    return True

def test_cta_functionality():
    """Test CTA gene functionality."""
    try:
        from pirlygenes.gene_sets_cancer import CTA_gene_names
        genes = CTA_gene_names()
        print(f"✓ CTA_gene_names() returned {len(genes)} genes")
        return len(genes) > 200  # Should be ~257
    except Exception as e:
        print(f"✗ CTA_gene_names() failed: {e}")
        return False

def test_cli_import():
    """Test CLI module imports."""
    try:
        from pirlygenes import cli
        print("✓ CLI module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CLI: {e}")
        return False
    return True

def test_plotting_dependencies():
    """Test plotting-related imports."""
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib missing: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ seaborn missing: {e}")
        return False
    return True

if __name__ == "__main__":
    print("Testing pirlygenes installation...")
    tests = [
        test_basic_imports,
        test_gene_sets_import,
        test_cta_functionality,
        test_cli_import,
        test_plotting_dependencies,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("✓ All tests passed - installation successful!")
    else:
        print("✗ Some tests failed - missing dependencies detected")
```

- [ ] **Step 2: Create fresh virtual environment**

Run: `python -m venv test_env`
Expected: Directory `test_env` created

- [ ] **Step 3: Activate test environment**

Run: `source test_env/bin/activate`
Expected: Prompt changes to show `(test_env)`

- [ ] **Step 4: Install package in test environment**

Run: `pip install -e .`
Expected: Installation completes without errors

- [ ] **Step 5: Run test script to identify missing dependencies**

Run: `python test_fresh_install.py`
Expected: Some tests may fail, revealing missing dependencies

- [ ] **Step 6: Document test results and missing dependencies**

Record output in terminal for next task - particularly any ImportError messages for matplotlib, seaborn, or other missing packages.

### Task 4: Update Dependencies Based on Test Results

**Files:**
- Modify: `pyproject.toml:16-22` (dependencies array)

- [ ] **Step 1: Add matplotlib dependency if missing**

If test revealed matplotlib ImportError, add to dependencies array:

```toml
dependencies = [
    "numpy>=1.24.0,<2.0",
    "pandas>=2.0.0", 
    "pyensembl>=2.4.0",
    "argh",
    "openpyxl",
    "matplotlib>=3.5.0",
]
```

- [ ] **Step 2: Add seaborn dependency if missing**

If test revealed seaborn ImportError, add to dependencies array:

```toml
dependencies = [
    "numpy>=1.24.0,<2.0",
    "pandas>=2.0.0",
    "pyensembl>=2.4.0", 
    "argh",
    "openpyxl",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
]
```

- [ ] **Step 3: Add any other missing dependencies**

Based on test results, add other missing dependencies with appropriate version bounds.

- [ ] **Step 4: Reinstall in test environment**

Run: `pip install -e . --upgrade`
Expected: New dependencies installed

- [ ] **Step 5: Rerun tests to verify fixes**

Run: `python test_fresh_install.py`
Expected: All tests now pass

- [ ] **Step 6: Commit dependency updates**

```bash
git add pyproject.toml
git commit -m "Add missing plotting dependencies discovered in fresh environment testing"
```

### Task 5: Clean Up Test Environment

**Files:**
- Delete: `test_fresh_install.py`
- Delete: `test_env/` directory

- [ ] **Step 1: Deactivate test environment**

Run: `deactivate`
Expected: Prompt returns to normal (no `(test_env)`)

- [ ] **Step 2: Remove test environment directory**

Run: `rm -rf test_env`

- [ ] **Step 3: Remove test script**

Run: `rm test_fresh_install.py`

- [ ] **Step 4: Verify cleanup**

Run: `ls test_*`
Expected: `ls: test_*: No such file or directory`

### Task 6: Update README with Installation Notes

**Files:**
- Modify: `README.md:1-10` (add installation section after title)

- [ ] **Step 1: Add installation section after title**

Add after line 4 (before "## TCR-T" section):

```markdown
# pirlygenes

Gene lists related to cancer immunotherapy

## Installation

Install from PyPI:

```bash
pip install pirlygenes
```

Or install from source:

```bash
git clone https://github.com/pirl-unc/pirlygenes.git
cd pirlygenes
pip install -e .
```

### pyensembl Dependency Note

This package requires `pyensembl>=2.4.0`. If you encounter installation issues with pyensembl, you may need to use a fork with setuptools compatibility fixes until [PR #322](https://github.com/openvax/pyensembl/pull/322) is merged:

```bash
pip install git+https://github.com/bnelsj/pyensembl.git@worktree-feature%2Bupdate_setuptools
```

### Dependencies

Core dependencies:
- numpy>=1.24.0,<2.0
- pandas>=2.0.0  
- pyensembl>=2.4.0
- matplotlib>=3.5.0 (for plotting)
- seaborn>=0.11.0 (for statistical plots)

```

- [ ] **Step 2: Verify markdown formatting**

Run: `head -30 README.md`
Expected: Installation section appears correctly formatted

- [ ] **Step 3: Commit README updates**

```bash
git add README.md
git commit -m "Add installation instructions and dependency notes to README"
```

### Task 7: Final Validation

**Files:**
- Test all modified files

- [ ] **Step 1: Test package installation from scratch**

Create new test environment:
```bash
python -m venv final_test
source final_test/bin/activate
pip install -e .
```

- [ ] **Step 2: Test key functionality**

Run: `python -c "from pirlygenes.gene_sets_cancer import CTA_gene_names; print(f'Success: {len(CTA_gene_names())} genes')"`
Expected: Output like "Success: 257 genes"

- [ ] **Step 3: Test CLI functionality**

Run: `python -m pirlygenes.cli --help`
Expected: Help message displays without errors

- [ ] **Step 4: Verify package metadata**

Run: `pip show pirlygenes`
Expected: Shows correct version and dependencies

- [ ] **Step 5: Clean up final test**

```bash
deactivate
rm -rf final_test
```

- [ ] **Step 6: Create final commit**

```bash
git add -A
git commit -m "Installation modernization complete: migrated to pyproject.toml-only dependencies"
```

### Task 8: Documentation and Cleanup

**Files:**
- Verify all changes are committed

- [ ] **Step 1: Review all changes**

Run: `git log --oneline -5`
Expected: Shows all commits from this implementation

- [ ] **Step 2: Verify no uncommitted changes**

Run: `git status`
Expected: "working tree clean"

- [ ] **Step 3: Push changes if desired**

Run: `git push origin feature+update_installation`
Expected: Changes pushed to remote branch

- [ ] **Step 4: Validate pyproject.toml final state**

Run: `cat pyproject.toml`
Expected: Shows updated configuration with static dependencies and no requirements.txt references

- [ ] **Step 5: Document completion**

Installation modernization complete. Key changes:
- Migrated from requirements.txt to pyproject.toml-only dependencies  
- Fixed numpy version typo (1.26.09 → ≥1.24.0,<2.0)
- Removed restrictive pandas upper bound (≤2.2.1 → ≥2.0.0)
- Added missing plotting dependencies (matplotlib, seaborn)
- Updated README with installation instructions and pyensembl fork notes