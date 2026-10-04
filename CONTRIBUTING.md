# Contributing to People Analytics Toolkit

Thank you for your interest in contributing to `people-analytics-toolkit`! This project provides production-grade, mathematically rigorous feature engineering methodologies for People Analytics, Workforce Intelligence, and Human Capital Science.

We welcome contributions of new analytical methods, performance optimizations, documentation improvements, and bug fixes.

---

## Code of Conduct

We are committed to providing a welcoming, inclusive, and professional environment for all contributors. Please be respectful, constructive, and collaborative in all discussions, issues, and pull requests.

---

## Getting Started

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- Git

### Development Environment Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/sam-tritto/people-analytics-toolkit.git
   cd people-analytics-toolkit
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using uv (recommended)
   uv venv .venv --python 3.11
   source .venv/bin/activate

   # Or using standard venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   # Using uv
   uv pip install -e ".[all,dev]"

   # Or using pip
   pip install -e ".[all,dev]"
   ```

---

## Development Workflow

### Code Style and Formatting

- **PEP 8**: Follow standard Python conventions.
- **Type Annotations**: All public APIs must have complete PEP 484 type annotations. The package is PEP 561 compliant (`py.typed`).
- **Docstrings**: Document all classes and functions using NumPy / SciPy docstring style, specifying:
  - Formula / mathematical formulation
  - Parameters with types
  - Return types and shapes
  - Business/HCM interpretation guidelines

### Type Checking

Verify typing compliance with `mypy`:
```bash
mypy src/
```

### Running the Test Suite

Run the full pytest suite:
```bash
pytest tests/ -v
```

Run specific test modules:
```bash
pytest tests/test_features.py -v
pytest tests/test_polymorphism_and_properties.py -v
pytest tests/test_testing_gaps.py -v
```

---

## Writing Tests

When contributing a new feature or method, please ensure:

1. **Unit & Edge-Case Tests**:
   - Test with typical inputs, empty DataFrames, zero-variance inputs, and missing/NaN values.
2. **Polymorphic Input Tests**:
   - Parametrize inputs to ensure analytical functions accept `pd.Series`, `np.ndarray`, Python `list`, and primitive scalars (`float`/`int`) where appropriate.
3. **Property-Based Invariants (Hypothesis)**:
   - For functions with mathematical guarantees (homogeneity, monotonicity, shift invariance, conservation of mass), write Hypothesis tests in `tests/test_polymorphism_and_properties.py`.
4. **Shared Fixtures**:
   - Use or extend standardized fixtures in `tests/conftest.py` rather than creating redundant synthetic data generation within test files.

---

## Pull Request Guidelines

1. **Create a topic branch:**
   ```bash
   git checkout -b feature/my-new-method
   ```

2. **Commit your changes:**
   - Write clear, concise commit messages.
   - Ensure all tests pass locally before pushing.

3. **Verify the build:**
   ```bash
   python -m build --wheel
   ```

4. **Submit a Pull Request:**
   - Reference any relevant issues.
   - Describe what the change accomplishes and include mathematical rationale or citations where applicable.
   - Ensure GitHub Actions CI checks pass.

---

## License

By contributing to `people-analytics-toolkit`, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
