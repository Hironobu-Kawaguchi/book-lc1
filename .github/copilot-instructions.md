# AI Agent Instructions for book-template

## Project Architecture

This is a **Markdown-to-EPUB publishing pipeline** for Kindle (KDP), using Pandoc and GitHub Actions. The flow: `manuscript/*.md` → sanitization (removes TODO lines) → Pandoc GFM→EPUB3 → `build/book.epub`.

**Key components:**
- `manuscript/`: Book chapters with numbered prefixes (`00-`, `01-`, `02-`) to enforce ordering
- `metadata.yml`: Single source of truth for book metadata (title, author, lang, cover)
- `epub.css`: Typography and layout styling (chapter breaks, code blocks, Japanese text spacing)
- `scripts/sanitize_manuscript.sh`: Pre-build step that strips lines matching `^\s*TODO(\s|\[|:)` using BSD-compatible sed
- `.github/workflows/`: CI automation—`build.yml` for main push, `release.yml` for v* tags

## Critical Developer Workflows

**Build locally:**
```bash
make build  # Runs sanitize → Pandoc → build/book.epub
```

**Manual build (step-by-step):**
```bash
# 1) Create directories
mkdir -p build/sanitized

# 2) Remove TODO lines from manuscript
bash scripts/sanitize_manuscript.sh

# 3) Generate EPUB with Pandoc
pandoc --from=gfm --standalone \
  --metadata-file=metadata.yml \
  --output=build/book.epub \
  --toc \
  --css=epub.css \
  --syntax-highlighting=kate \
  build/sanitized/*.md
```

**Prerequisites:** 
```bash
brew install pandoc git-lfs
git lfs install  # Enable Git LFS for this repository
# Images (jpg, png, svg, gif) are auto-tracked per .gitattributes
```

**Clean build artifacts:** `make clean` (removes `build/` directory)

**Test:** Open `build/book.epub` in Apple Books/Calibre—verify TOC, images, chapter breaks

**Release:** `git tag v1.0.0 && git push origin v1.0.0` triggers `release.yml` to attach EPUB to GitHub Release

## Project-Specific Conventions

### One Sentence Per Line (1文1行)
**Non-negotiable:** Each sentence must be on its own line in `manuscript/*.md`. This makes Git diffs readable and enables precise review comments. Example:
```markdown
従来の WYSIWYG エディタは差分が読みにくくレビューが困難である。
Git はテキスト差分に最適化されており Markdown と相性が良い。
```

### Chapter Ordering
Files in `manuscript/` use numbered prefixes to guarantee build order: `00-frontmatter.md`, `01-introduction.md`, `02-chapter-one.md`. Never rely on alphabetical sorting alone.

### TODO Comment Strategy
Lines starting with `TODO`, `TODO:`, or `TODO [...]` are **automatically removed** by `sanitize_manuscript.sh` before build. Use freely for WIP notes that should never appear in the final EPUB.

### Image References
Place images in `images/` and reference with relative paths: `![Description](images/filename.png)`. Large images are tracked via Git LFS (jpg, png, svg, gif per `.gitattributes`).

### Metadata and Cover
Edit `metadata.yml` to set title, author, language, publisher, and date. To add a cover image: place `images/cover.jpg` and uncomment the `cover-image` field in `metadata.yml`.

### Styling
Modify `epub.css` for typography changes—**never use inline styles** in Markdown. The CSS includes Kindle/KFX optimizations (page breaks on h1/h2, Japanese text spacing hints).

## Key Files That Define Patterns

- `manuscript/01-introduction.md`: Exemplifies 1-sentence-per-line and heading hierarchy
- `scripts/sanitize_manuscript.sh`: Shows BSD sed pattern for TODO removal (`^\s*TODO(\s|\[|:)`)
- `.github/workflows/build.yml`: Pandoc Docker image usage (`pandoc/core:3.8`) with exact args
- `epub.css`: Typography rules for Japanese books (text-spacing, line-break, page-break-before)

## Integration Points

**CI/CD:** GitHub Actions workflows are triggered on `push` (build.yml) and `tags: v*` (release.yml). Both use Docker-based Pandoc (`pandoc/core:3.8`) for consistency.

**Pandoc flags:** Always use `--from=gfm --standalone --toc --css=epub.css --syntax-highlighting=kate` for GFM support and syntax highlighting.

**KDP:** EPUB files must be manually uploaded to KDP Bookshelf (no API integration). Download from GitHub Releases after tagging.
