# GitHub Publish Checklist

> Pre-publication checklist for Cheng Zhang's postdoctoral research portfolio repository.

**Author:** Cheng Zhang, PhD
**Repository:** MPLNet-Cultural-Heritage-GeoAI

---

## Pre-Publication Review

### 1. Privacy & Security

- [ ] **REMOVED** Personal phone number from public files (body text)
- [ ] **REMOVED** ID numbers, passport information
- [ ] **REMOVED** Private CV, transcripts, certificates
- [ ] **ADDED** `POSTDOC_PORTFOLIO_PRIVATE.md` to `.gitignore`
- [ ] **VERIFIED** No GPS coordinates of villages in code/comments
- [ ] **VERIFIED** No village residents' personal information
- [ ] **VERIFIED** No unauthorized photography from field work

### 2. GitHub Username Privacy

> **⚠️ IMPORTANT:** The current GitHub username appears to contain a phone number: `Jack13026212687`.

If the author wants to avoid public phone-number exposure, consider:

- Creating a professional GitHub username (e.g., `zhangcheng-geoai`)
- Creating a GitHub organization account for academic portfolios
- Changing the username before public release

**Current exposure:**
- GitHub profile URL: https://github.com/Jack13026212687
- GitHub Pages URL: https://Jack13026212687.github.io/MPLNet-Cultural-Heritage-GeoAI/

如果作者希望避免公开暴露手机号，建议在正式投递前更换为更专业的 GitHub 用户名或新建组织账号。

### 3. Copyright & Licensing

- [ ] **REVIEWED** Published journal PDFs are allowed for academic portfolio demonstration
- [ ] **BLOCKED** Patent application drafts, certificates, and confidential materials
- [ ] **ADDED** LICENSE file with scope clarification
- [ ] **ADDED** copyright notice in README.md
- [ ] **UNDERSTOOD** MIT license applies to source code ONLY

### 4. Content Accuracy

- [ ] **VERIFIED** All DOI links are correct and accessible
- [ ] **VERIFIED** Author names match actual publication
- [ ] **VERIFIED** Journal names, volumes, pages are accurate
- [ ] **REMOVED** All unverified impact factor claims
- [ ] **REMOVED** All unverified JCR quartile claims
- [ ] **REMOVED** All unverified citation counts
- [ ] **FLAGGED** Any remaining unverified metrics

### 5. Identity

- [ ] **CONFIRMED** Identity wording: "Cheng Zhang, PhD"

### 6. Repository Quality

- [ ] **UPDATED** README.md with clear structure and professional tone
- [ ] **CREATED** MODEL_CARD.md for model documentation
- [ ] **CREATED** DATASET_CARD.md for dataset documentation
- [ ] **CREATED** papers/README.md with publication metadata
- [ ] **CREATED** docs/index.html for GitHub Pages
- [ ] **ORGANIZED** file structure is logical and consistent

### 7. Code Functionality

- [ ] **TESTED** `python tools/train.py --help` works
- [ ] **TESTED** `python tools/evaluate.py --help` works
- [ ] **VERIFIED** All imports work
- [ ] **VERIFIED** requirements.txt is complete

### 8. Git Configuration

- [ ] **ADDED** sensitive patterns to `.gitignore`
- [ ] **REMOVED** any committed sensitive data
- [ ] **CHECKED** `git status` for unintended files
- [ ] **VERIFIED** private files are not tracked

---

## Privacy Verification Commands

### Git Bash

```bash
# Check for phone numbers in public files (body text only)
grep -RInE "13026212687|130-2621-2687|\+86 130|1302621" --include="*.md" --include="*.html" --include="*.css" . 2>/dev/null

# Check for sensitive patterns
grep -r "身份证\|身份证号" --include="*.md" --include="*.py" .

# Verify private file is not tracked
git ls-files | grep -i "POSTDOC_PORTFOLIO_PRIVATE"

# Check which PDF files will be committed
git status --short papers/
```

### PowerShell

```powershell
# Check for phone numbers in body text
Select-String -Path (Get-ChildItem -Recurse -Include *.md,*.html,*.css | Select-Object -ExpandProperty FullName) -Pattern "13026212687|130-2621-2687|\+86 130|1302621"

# Check for sensitive files tracked
git ls-files | Select-String -Pattern "POSTDOC_PORTFOLIO_PRIVATE|\.docx$|certificate|cert|证书"
```

---

## Sensitive File Check Command

### Git Bash

```bash
git ls-files | grep -Ei "POSTDOC_PORTFOLIO_PRIVATE|\.docx$|resume|cv|certificate|cert|证书|专利|审查|答复|\.pth$|\.pt$|\.ckpt$|\.zip$|\.rar$"
```

### PowerShell

```powershell
git ls-files | Select-String -Pattern "POSTDOC_PORTFOLIO_PRIVATE|\.docx$|resume|cv|certificate|cert|证书|专利|审查|答复|\.pth$|\.pt$|\.ckpt$|\.zip$|\.rar$"
```

**Expected Result:** Should NOT include published journal PDFs (now allowed) or POSTDOC_PORTFOLIO_PRIVATE.md (in .gitignore)

**If sensitive files are found:**
```bash
# Remove from git tracking (does not delete local files)
git rm --cached path/to/file
```

---

## Publication Steps

### Step 1: Local Review

```bash
# Navigate to repository
cd MPLNet-Cultural-Heritage-GeoAI

# Review all changes
git status
git diff --stat

# Check phone numbers in body text
grep -RInE "13026212687|130-2621-2687" --include="*.md" --include="*.html" --include="*.css" . 2>/dev/null

# Check papers folder
git status --short papers/

# Test code
python tools/train.py --help
python tools/evaluate.py --help
```

### Step 2: Stage and Commit

```bash
# Stage all public files (including published journal PDFs)
git add .

# OR stage selectively (exclude any unintended files)
git add README.md POSTDOC_PORTFOLIO.md MODEL_CARD.md DATASET_CARD.md
git add papers/ docs/
git add models/ datasets/ configs/ tools/
git add requirements.txt .gitignore LICENSE GITHUB_PUBLISH_CHECKLIST.md

# Review staged files
git status

# Commit
git commit -m "docs: publish bilingual Cultural Heritage GeoAI postdoc portfolio

- Complete bilingual documentation
- GitHub Pages with target collaboration tracks
- MIT LICENSE with scope clarification
- Papers/README with DOI references and published PDFs
- Model and Dataset cards"

# Push
git push origin main
```

---

## GitHub Pages Deployment

1. Push the repository to GitHub.
2. Open repository Settings.
3. Go to Pages.
4. Build and deployment:
   - Source: Deploy from a branch
   - Branch: main
   - Folder: /docs
5. Save.
6. Wait for deployment (~2 minutes).
7. Expected URL:
   https://Jack13026212687.github.io/MPLNet-Cultural-Heritage-GeoAI/

---

## Journal Information Note

> **Important:** Journal-level metrics (impact factor, JCR quartile, citations, etc.) change over time and should be verified from official indexing databases before any public release. This repository does not make claims about:
>
> - Impact factors (IF)
> - JCR quartiles (Q1/Q2/Q3/Q4)
> - CAS分区 (Chinese Academy of Sciences ranking)
> - Citation counts
> - Any ranking claims
>
> Users should verify this information independently from Web of Science, Scopus, or journal websites.

---

## papers/ Folder Policy

### Allowed for Public GitHub Release

| File | Type | Status |
|------|------|--------|
| `Zhang_et_al-2026-npj_Heritage_Science.pdf` | Journal Article (Published) | Allowed |
| `中观-1MPLNet Mamba prompt learning networks.pdf` | Journal Article (Published) | Allowed |
| `宏观-张成2025Ecological suitability evaluation .pdf` | Journal Article (Published) | Allowed |
| `微观- 2024 EPANet KD Efficient.pdf` | Journal Article (Published) | Allowed |

### Must Be Excluded

| Pattern | Type | Reason |
|---------|------|--------|
| `*.docx` | Patent application drafts | Confidential |
| `*patent*`, `*Patent*` | Patent documents | Not for distribution |
| `*专利*`, `*审查*`, `*答复*` | Patent review documents | Confidential |
| `*证书*`, `*certificate*` | Patent certificates | Not required for repo |

---

## Emergency: Remove Sensitive Data

If sensitive data is accidentally committed:

```bash
# Remove from git history (use with caution)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch PATH/TO/SENSITIVE/FILE' \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: rewrites history)
git push origin --force --all
```

**Better:** Immediately contact GitHub Support for repository cleanup if credentials were exposed.

---

## Final Checklist Before Push

- [ ] No phone numbers in body text of public files
- [ ] No unverified impact factors or JCR claims
- [ ] MIT license scope is clear
- [ ] Private file (POSTDOC_PORTFOLIO_PRIVATE.md) is in .gitignore
- [ ] Only published journal PDFs are in papers/ folder
- [ ] PhD identity is correctly stated as "Cheng Zhang, PhD"
- [ ] All DOI links are valid
- [ ] Code is functional
- [ ] GitHub username privacy risk assessed

---

## Final Git Commands

```bash
# 1. Check status
git status

# 2. Check for sensitive files tracked (excluding allowed PDFs)
git ls-files | grep -Ei "POSTDOC_PORTFOLIO_PRIVATE|\.docx$|resume|cv|certificate|cert|证书|专利|审查|答复"

# 3. Check papers folder
git status --short papers/

# 4. If no sensitive files found, stage all:
git add .

# 5. Commit
git commit -m "docs: publish bilingual Cultural Heritage GeoAI postdoc portfolio"

# 6. Push
git push origin main
```

---

*Last Updated: 2026-05-26*
