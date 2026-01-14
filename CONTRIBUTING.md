# Contributing to Meta-Cognitive Reasoning

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/metacog-reasoning.git
   cd metacog-reasoning
   ```
3. **Set up the development environment**:
   ```bash
   bash scripts/setup_environment.sh
   source venv/bin/activate
   ```

## Development Workflow

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Run tests** to ensure everything works:
   ```bash
   pytest tests/
   ```

4. **Format your code**:
   ```bash
   black src/ scripts/ tests/
   isort src/ scripts/ tests/
   ```

5. **Commit your changes** with a descriptive message:
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## Code Style Guidelines

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Add type hints to function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and concise (< 50 lines when possible)

## Commit Message Guidelines

Use conventional commit messages:

- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for updates to existing features
- `Refactor:` for code refactoring
- `Docs:` for documentation changes
- `Test:` for adding or updating tests

Example:
```
Add: Multi-component reward function for student training

- Implement R_answer, R_strategy, R_process, R_plan
- Add BERTScore computation for process similarity
- Include unit tests for all reward components
```

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

Run tests:
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- Update README.md if you add new features
- Add docstrings to new functions and classes
- Update configuration examples if needed
- Add examples to notebooks/ if appropriate

## Areas for Contribution

We welcome contributions in these areas:

1. **Core Framework**
   - Improving training algorithms
   - Adding new reasoning strategies
   - Optimizing reward functions

2. **Data & Evaluation**
   - Adding new benchmarks
   - Improving translation quality
   - Creating new evaluation metrics

3. **Analysis**
   - Strategy transfer analysis
   - Error taxonomy
   - Visualization tools

4. **Documentation**
   - Tutorials and examples
   - API documentation
   - Research paper summaries

5. **Infrastructure**
   - CI/CD pipelines
   - Docker containers
   - Deployment scripts

## Questions or Issues?

- Open an issue on GitHub for bugs or feature requests
- Start a discussion for questions or ideas
- Contact the maintainers directly for sensitive issues

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming environment for all contributors

Thank you for contributing! 🎉
