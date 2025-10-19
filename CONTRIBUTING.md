# Contributing to FloorMind 🤝

Thank you for your interest in contributing to FloorMind! We welcome contributions from the community and are excited to see what you'll build.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- CUDA-compatible GPU (recommended for training)
- Basic knowledge of PyTorch and diffusion models

### Development Setup
1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/yourusername/FloorMind.git
   cd FloorMind
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install jupyter matplotlib seaborn pytest black flake8
   ```

3. **Verify installation**
   ```bash
   # Test model loading
   python -c "from backend.services.model_service import ModelService; print('✅ Setup complete')"
   
   # Test notebooks
   jupyter notebook
   # Open and run a few cells from notebooks/FloorMind_Base_Training.ipynb
   ```

## 🎯 Ways to Contribute

### 🐛 Bug Reports
Found a bug? Please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages and logs

### ✨ Feature Requests
Have an idea? We'd love to hear it! Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant examples or mockups

### 🔧 Code Contributions

#### Areas We Need Help With:
- **🎨 New Constraint Types**: Accessibility, sustainability, feng shui
- **📊 Dataset Improvements**: More diverse floor plans, better annotations
- **🔧 Performance Optimization**: Faster generation, lower memory usage
- **🎯 Evaluation Metrics**: Better architectural quality assessment
- **📱 Frontend Enhancements**: Improved UI/UX, mobile support
- **📚 Documentation**: Tutorials, examples, API docs

## 📝 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure notebooks run cleanly

### 3. Code Quality Checks
```bash
# Format code
black .

# Check style
flake8 .

# Run tests
python -m pytest backend/tests/

# Test notebooks (run all cells)
jupyter nbconvert --execute notebooks/*.ipynb
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add new constraint type for accessibility"
# Use conventional commit format:
# feat: new feature
# fix: bug fix
# docs: documentation
# style: formatting
# refactor: code restructuring
# test: adding tests
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
# Then create PR on GitHub
```

## 🎨 Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where possible
- Write docstrings for functions and classes
- Keep functions focused and small
- Use meaningful variable names

```python
def generate_floorplan(
    prompt: str, 
    constraints: Dict[str, Any],
    num_inference_steps: int = 20
) -> Dict[str, Any]:
    """
    Generate a floor plan from text prompt with constraints.
    
    Args:
        prompt: Text description of desired floor plan
        constraints: Dictionary of architectural constraints
        num_inference_steps: Number of diffusion steps
        
    Returns:
        Dictionary containing generated image and metadata
    """
    # Implementation here
    pass
```

### Jupyter Notebooks
- Clear markdown explanations for each section
- Well-commented code cells
- Include output examples
- Test with "Restart & Run All"
- Keep cells focused and not too long

### Documentation
- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep README and guides up to date

## 🧪 Testing Guidelines

### Unit Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest backend/tests/test_model_service.py

# Run with coverage
python -m pytest --cov=backend
```

### Integration Tests
- Test API endpoints
- Verify model loading and generation
- Check notebook execution

### Manual Testing
- Test frontend demo
- Verify training notebooks
- Check generated floor plan quality

## 📊 Dataset Contributions

### Adding New Floor Plans
1. **Image Requirements**:
   - PNG format, 512x512 resolution
   - Clean, professional floor plans
   - Proper room labels and dimensions

2. **Metadata Format**:
   ```csv
   filename,description,rooms,area_sqft,style
   plan_001.png,"Modern 2BR apartment",2,850,"Contemporary"
   ```

3. **Quality Standards**:
   - Architecturally accurate
   - Clear room boundaries
   - Proper scale and proportions
   - No copyrighted content

### Dataset Processing
```python
# Add to data/process_datasets.py
def process_new_dataset(source_dir: str, output_dir: str):
    """Process and validate new floor plan dataset."""
    # Implementation here
```

## 🤖 Model Contributions

### New Constraint Types
1. **Define constraint logic**:
   ```python
   class AccessibilityConstraint(BaseConstraint):
       def calculate_loss(self, predicted, target, metadata):
           # Implement accessibility compliance loss
           pass
   ```

2. **Add to training notebooks**:
   - Update constraint configuration
   - Add loss function integration
   - Include evaluation metrics

3. **Test thoroughly**:
   - Verify constraint effectiveness
   - Check generation quality
   - Measure performance impact

### Model Architecture Changes
- Discuss major changes in issues first
- Provide benchmarks and comparisons
- Ensure backward compatibility
- Update documentation

## 📚 Documentation Contributions

### Types of Documentation
- **API Documentation**: Function/class docstrings
- **User Guides**: How-to tutorials and examples
- **Developer Docs**: Architecture and contribution guides
- **Notebook Documentation**: Inline explanations and markdown

### Documentation Standards
- Clear, beginner-friendly language
- Include practical examples
- Keep information current
- Add screenshots for visual features

## 🎉 Recognition

Contributors will be:
- Listed in the README contributors section
- Mentioned in release notes
- Invited to join the core team (for significant contributions)

## ❓ Questions?

- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Create an Issue
- **Feature Ideas**: Start with a Discussion, then create an Issue
- **Development Help**: Join our community chat

## 📋 Pull Request Checklist

Before submitting your PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Notebooks execute cleanly
- [ ] Commit messages are clear
- [ ] PR description explains changes
- [ ] No large files committed
- [ ] Sensitive data removed

## 🏆 Types of Contributors

### 🌟 Code Contributors
- Implement new features
- Fix bugs and issues
- Optimize performance
- Improve architecture

### 📊 Data Contributors
- Provide new floor plan datasets
- Improve data quality
- Create data processing tools
- Validate architectural accuracy

### 📚 Documentation Contributors
- Write tutorials and guides
- Improve API documentation
- Create video tutorials
- Translate documentation

### 🧪 Testing Contributors
- Write comprehensive tests
- Perform manual testing
- Report bugs and issues
- Validate model quality

### 🎨 Design Contributors
- Improve UI/UX design
- Create visual assets
- Design better workflows
- Enhance user experience

---

**Thank you for contributing to FloorMind! Together, we're building the future of AI-powered architectural design.** 🏠✨