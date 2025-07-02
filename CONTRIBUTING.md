# Contributing to AgentCheck

Thank you for your interest in contributing to AgentCheck! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions from the community! Here are the main ways you can contribute:

### **🐛 Bug Reports**
- Use the [GitHub Issues](https://github.com/hvardhan878/agentcheck/issues) page
- Include a clear description, reproduction steps, and expected vs actual behavior
- Add the `bug` label and relevant component labels

### **💡 Feature Requests**
- Submit feature requests via [GitHub Issues](https://github.com/hvardhan878/agentcheck/issues)
- Use the `enhancement` label
- Describe the use case and expected benefits
- Consider if it aligns with our [roadmap](README.md#-roadmap)

### **📝 Documentation**
- Improve README, docstrings, or add new documentation
- Fix typos, clarify explanations, or add examples
- Submit via pull request

### **🔧 Code Contributions**
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests
- Submit a pull request

## 🛠️ Development Setup

### **Prerequisites**
- Python 3.9 or higher
- Git
- OpenAI API key (for testing)

### **Local Development Environment**

```bash
# 1. Clone the repository
git clone https://github.com/hvardhan878/agentcheck.git
cd agentcheck

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Set up pre-commit hooks
pre-commit install

# 5. Set your OpenAI API key for testing
export OPENAI_API_KEY=sk-your-key-here
```

### **Development Dependencies**

The project uses several development tools:

```bash
# Install all development dependencies
pip install -e ".[dev]"

# Or install individually:
pip install pytest pytest-cov pytest-asyncio
pip install black ruff mypy
pip install pre-commit
pip install streamlit plotly pandas numpy  # For dashboard
```

## 📋 Code Standards

### **Python Style Guide**
We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `ruff` for automatic sorting
- **Type hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings

### **Code Formatting**

```bash
# Format code with Black
black agentcheck/ tests/ demo/

# Sort imports with ruff
ruff check --fix agentcheck/ tests/ demo/

# Type checking with mypy
mypy agentcheck/

# Run all formatting checks
ruff check agentcheck/ tests/ demo/
```

### **Pre-commit Hooks**
The project uses pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🧪 Testing

### **Running Tests**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentcheck --cov-report=html

# Run specific test file
pytest tests/test_deterministic.py

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### **Test Structure**

```
tests/
├── unit/                    # Unit tests
│   ├── test_trace.py       # Trace functionality
│   ├── test_deterministic.py # Deterministic testing
│   └── test_utils.py       # Utility functions
├── integration/            # Integration tests
│   ├── test_cli.py         # CLI commands
│   └── test_dashboard.py   # Dashboard functionality
├── fixtures/               # Test fixtures and data
└── conftest.py            # Pytest configuration
```

### **Writing Tests**

```python
# Example test structure
import pytest
from agentcheck import Trace, DeterministicReplayer

class TestTrace:
    def test_trace_creation(self):
        """Test that traces can be created successfully."""
        with Trace() as trace:
            assert trace is not None
            assert trace.trace_id is not None
    
    def test_llm_call_recording(self):
        """Test that LLM calls are properly recorded."""
        with Trace() as trace:
            trace.add_llm_call(
                messages=[{"role": "user", "content": "Hello"}],
                response={"content": "Hi there!"},
                model="gpt-4o-mini"
            )
            
            assert len(trace.steps) == 1
            assert trace.steps[0]["type"] == "llm_call"

    @pytest.mark.asyncio
    async def test_async_trace(self):
        """Test async trace functionality."""
        async with Trace() as trace:
            # Async operations here
            pass
```

### **Test Guidelines**

- **Test coverage**: Aim for >90% coverage
- **Test names**: Descriptive names that explain what is being tested
- **Test isolation**: Each test should be independent
- **Fixtures**: Use pytest fixtures for common setup
- **Mocking**: Mock external dependencies (API calls, file system)

## 🏗️ Architecture Guidelines

### **Core Principles**

1. **Simplicity**: Keep the API simple and intuitive
2. **Composability**: Components should work together seamlessly
3. **Extensibility**: Easy to add new features and integrations
4. **Performance**: Efficient for production use
5. **Observability**: Built-in tracing and monitoring

### **Code Organization**

```
agentcheck/
├── __init__.py           # Main package exports
├── trace.py             # Core tracing functionality
├── deterministic.py     # Deterministic testing
├── cli.py              # Command-line interface
├── utils.py            # Utility functions
├── dashboard/          # Dashboard components
│   ├── __init__.py
│   ├── app.py
│   └── components.py
└── integrations/       # Third-party integrations
    ├── __init__.py
    ├── openai.py
    └── langchain.py
```

### **Adding New Features**

1. **Plan the feature**: Create an issue describing the feature
2. **Design the API**: Consider how it fits with existing functionality
3. **Implement**: Follow the code standards and add tests
4. **Document**: Update README and add docstrings
5. **Test**: Ensure all tests pass and add new tests

## 📦 Building and Distribution

### **Local Build**

```bash
# Build the package
python -m build

# Install from local build
pip install dist/agentcheck-*.whl
```

### **Testing the Build**

```bash
# Test the CLI
agentcheck --help

# Test basic functionality
agentcheck trace "echo 'Hello, World!'" --output test.json
agentcheck show test.json
```

## 🚀 Release Process

### **Version Management**
We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### **Release Checklist**

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version is bumped
- [ ] Release notes are written
- [ ] PyPI package is built and uploaded

## 🤝 Pull Request Process

### **Before Submitting**

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run all checks**:
   ```bash
   pytest
   black agentcheck/ tests/
   ruff check agentcheck/ tests/
   mypy agentcheck/
   ```

### **Pull Request Guidelines**

1. **Title**: Clear, descriptive title
2. **Description**: Explain what the PR does and why
3. **Related issues**: Link to relevant issues
4. **Tests**: Ensure all tests pass
5. **Documentation**: Update docs if needed

### **Review Process**

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Discussion** of any concerns
4. **Approval** and merge

## 🐛 Debugging

### **Common Issues**

#### **Import Errors**
```bash
# Make sure you're in the right directory
cd agentcheck

# Install in development mode
pip install -e ".[dev]"

# Check Python path
python -c "import agentcheck; print(agentcheck.__file__)"
```

#### **Test Failures**
```bash
# Run tests with more detail
pytest -v -s

# Run specific failing test
pytest tests/test_specific.py::test_function -v -s

# Check test coverage
pytest --cov=agentcheck --cov-report=term-missing
```

#### **CLI Issues**
```bash
# Test CLI installation
which agentcheck

# Check CLI help
agentcheck --help

# Debug CLI commands
python -m agentcheck.cli --help
```

## 📚 Documentation

### **Documentation Standards**

- **README.md**: Main project documentation
- **Docstrings**: Google-style docstrings for all public functions
- **Examples**: Include working examples in docstrings
- **Type hints**: All public functions should have type hints

### **Example Docstring**

```python
def deterministic_replay(
    consistency_threshold: float = 0.8,
    baseline_runs: int = 5,
    baseline_name: Optional[str] = None
) -> Callable:
    """Decorator for deterministic replay testing.
    
    Args:
        consistency_threshold: Minimum consistency score (0.0-1.0).
        baseline_runs: Number of runs to establish baseline.
        baseline_name: Name for the baseline (auto-generated if None).
    
    Returns:
        Decorated function with deterministic replay capabilities.
    
    Example:
        >>> @deterministic_replay(consistency_threshold=0.8)
        ... def my_agent(input_text: str) -> str:
        ...     return "Hello, " + input_text
    """
```

## 🎯 Contribution Areas

### **High Priority**
- **Bug fixes**: Critical issues affecting core functionality
- **Performance improvements**: Faster execution, lower memory usage
- **Test coverage**: Improving test coverage and quality
- **Documentation**: Better examples and explanations

### **Medium Priority**
- **New integrations**: Support for more LLM providers
- **Dashboard improvements**: Better UI/UX and new features
- **CLI enhancements**: More commands and options
- **Enterprise features**: Security, compliance, monitoring

### **Low Priority**
- **Nice-to-have features**: Convenience functions
- **Experimental features**: Proof-of-concept implementations
- **Documentation**: Additional guides and tutorials

## 📞 Getting Help

### **Community Resources**
- **GitHub Issues**: [Report bugs and request features](https://github.com/hvardhan878/agentcheck/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/hvardhan878/agentcheck/discussions)
- **Documentation**: [README.md](README.md) and inline docs

### **Before Asking for Help**
1. **Check existing issues**: Search for similar problems
2. **Read the documentation**: Check README and docstrings
3. **Try debugging**: Use the debugging tips above
4. **Provide context**: Include error messages, code examples, and environment details

## 📄 License

By contributing to AgentCheck, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Thank you to all contributors who have helped make AgentCheck better! Your contributions are greatly appreciated.

---

**Happy coding! 🚀** 