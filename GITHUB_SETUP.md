# GitHub Setup Instructions

This document provides step-by-step instructions for uploading the project to GitHub.

## Prerequisites

- GitHub account
- Git installed locally
- SSH key configured with GitHub (recommended) or HTTPS access

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon in the top right → "New repository"
3. Fill in the details:
   - **Repository name:** `metacog-reasoning`
   - **Description:** "Meta-Cognitive Self-Play with Cross-Lingual Reasoning Distillation"
   - **Visibility:** Public (or Private if preferred)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click "Create repository"

## Step 2: Link Local Repository to GitHub

The local Git repository is already initialized. Now connect it to GitHub:

### Option A: Using SSH (Recommended)

```bash
cd ~/metacog-reasoning
git remote add origin git@github.com:YOUR_USERNAME/metacog-reasoning.git
git push -u origin main
```

### Option B: Using HTTPS

```bash
cd ~/metacog-reasoning
git remote add origin https://github.com/YOUR_USERNAME/metacog-reasoning.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Verify Upload

1. Go to your repository on GitHub: `https://github.com/YOUR_USERNAME/metacog-reasoning`
2. Verify that all files are present
3. Check that README.md is displayed correctly

## Step 4: Configure Repository Settings (Optional)

### Add Topics

Go to repository → About (gear icon) → Add topics:
- `machine-learning`
- `nlp`
- `reasoning`
- `reinforcement-learning`
- `multilingual`
- `indic-languages`
- `meta-cognition`
- `knowledge-distillation`

### Enable GitHub Actions (Future)

Go to Settings → Actions → General → Allow all actions

### Add Collaborators

Go to Settings → Collaborators → Add people

## Step 5: Update README with Your Information

Before pushing, update these placeholders in `README.md`:

```bash
# Replace placeholders
sed -i 's/yourusername/YOUR_ACTUAL_USERNAME/g' README.md
sed -i 's/your.email@example.com/YOUR_ACTUAL_EMAIL/g' README.md

# Commit the changes
git add README.md
git commit -m "Update: Replace placeholder information with actual details"
git push
```

## Step 6: Set Up Branch Protection (Recommended)

For collaborative work:

1. Go to Settings → Branches
2. Add rule for `main` branch:
   - ✅ Require pull request before merging
   - ✅ Require status checks to pass
   - ✅ Require conversation resolution before merging

## Step 7: Create Development Branch

For active development:

```bash
git checkout -b develop
git push -u origin develop
```

## Workflow for Collaboration

### For Team Members:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/metacog-reasoning.git
   cd metacog-reasoning
   ```

2. **Set up environment:**
   ```bash
   bash scripts/setup_environment.sh
   source venv/bin/activate
   ```

3. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make changes, commit, and push:**
   ```bash
   git add .
   git commit -m "Add: Your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request on GitHub**

### For Maintainers:

1. **Review Pull Requests**
2. **Merge to main after approval**
3. **Create releases for major milestones:**
   ```bash
   git tag -a v0.1.0 -m "Phase 0: Initial setup complete"
   git push origin v0.1.0
   ```

## GitHub Repository Structure

Your repository will have:

```
metacog-reasoning/
├── .github/              # (Future: CI/CD workflows)
├── config/               # Configuration files
├── data/                 # Data directories (empty, .gitignored)
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks
├── scripts/              # Utility scripts
├── src/                  # Source code
├── tests/                # Unit tests
├── README.md             # Main documentation
├── CONTRIBUTING.md       # Contribution guidelines
├── LICENSE               # MIT License
├── requirements.txt      # Dependencies
└── setup.py              # Package setup
```

## Next Steps After Upload

1. **Share the repository** with your advisor and collaborators
2. **Set up Weights & Biases** for experiment tracking:
   ```bash
   wandb login
   ```
3. **Start Phase 1 implementation:**
   ```bash
   python scripts/train_teacher.py
   ```

## Troubleshooting

### Permission Denied (SSH)

If you get "Permission denied (publickey)":
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your.email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
3. Copy key: `cat ~/.ssh/id_ed25519.pub`

### Large Files

If you have large model checkpoints:
1. Install Git LFS: `git lfs install`
2. Track large files: `git lfs track "*.bin" "*.pth"`
3. Commit: `git add .gitattributes && git commit -m "Add: Git LFS tracking"`

### Authentication Failed (HTTPS)

Use a Personal Access Token instead of password:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use token as password when pushing

---

**Status:** Ready to upload to GitHub!
**Last Updated:** January 2026
