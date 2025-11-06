# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Markdown-to-EPUB book publishing template for Kindle (KDP) that uses Pandoc and GitHub Actions following a Docs-as-Code approach. Authors write in GFM (GitHub Flavored Markdown) with a "one sentence per line" convention to make Git diffs more readable. The build process sanitizes TODO comments, generates EPUB3 files with custom CSS styling, and can automatically publish to GitHub Releases.

## Build Commands

**Prerequisites:**
- Install Pandoc: `brew install pandoc`
- Install Git LFS (for images): `brew install git-lfs && git lfs install`

**Build the book locally:**
```bash
make build
```
This runs `scripts/sanitize_manuscript.sh` to remove TODO lines, then generates `build/book.epub` using Pandoc.

**Manual build (if needed):**
```bash
mkdir -p build/sanitized
bash scripts/sanitize_manuscript.sh
pandoc --from=gfm --standalone \
  --metadata-file=metadata.yml \
  --output=build/book.epub \
  --toc \
  --css=epub.css \
  --syntax-highlighting=kate \
  build/sanitized/*.md
```

**Clean build artifacts:**
```bash
make clean
```

**Test the EPUB:**
Open `build/book.epub` in an EPUB viewer (e.g., Apple Books, Calibre) and verify:
- Table of contents navigation works
- Images render correctly
- Code blocks are readable
- Chapter breaks appear correctly

## Repository Structure

```
manuscript/           # Book chapters (numbered 00-, 01-, etc.)
images/              # Figures and cover (Git LFS managed)
scripts/             # Build utilities (sanitize_manuscript.sh)
build/               # Generated artifacts (gitignored)
.github/workflows/   # CI automation (build.yml, release.yml)
metadata.yml         # Book metadata (title, author, lang, cover)
epub.css             # EPUB styling
Makefile             # Build commands
```

## Writing Guidelines

**One sentence per line:** Each sentence must be on its own line to make Git diffs more readable during review. This is critical for collaboration and change tracking and dramatically improves review efficiency.

**Chapter organization:**
- Files in `manuscript/` use numbered prefixes to ensure stable ordering: `00-frontmatter.md`, `01-introduction.md`, `02-chapter-one.md`
- Use GFM syntax: `#` for chapter titles (h1), `##` for sections (h2), and maintain hierarchical structure
- Images go in `images/` and are referenced with relative paths: `![図1](images/ch01-diagram.png)`

**TODO comments:** Lines starting with `TODO`, `TODO:`, or `TODO [...]` are automatically removed during build by `scripts/sanitize_manuscript.sh`. Use them freely for WIP notes that should not appear in the final EPUB.

**Metadata and cover:**
- Edit `metadata.yml` to set title, author, language (lang), publisher, date, and rights
- Add cover image as `images/cover.jpg` and uncomment `cover-image` field in `metadata.yml` to enable it

**Styling:** Modify `epub.css` for typography and layout (includes Japanese text optimization, chapter page breaks, code block styling). Avoid inline styles in Markdown.

## CI/CD Workflows

**Continuous build (`.github/workflows/build.yml`):**
- Triggered on push to `main`
- Builds EPUB and uploads as GitHub Actions artifact
- Uses `pandoc/core:3.8` Docker image

**Release publishing (`.github/workflows/release.yml`):**
- Triggered when pushing tags matching `v*` pattern
- Builds EPUB and attaches to GitHub Release
- Example: `git tag v1.0.0 && git push origin v1.0.0`

## Git LFS Setup

Large images can bloat the repository. Images (jpg, jpeg, png, gif, svg) are configured to be tracked with Git LFS in `.gitattributes`. Before adding images:
```bash
brew install git-lfs  # if not already installed
git lfs install
# Tracking rules already configured in .gitattributes (jpg/png/svg/gif)
```

Then add images normally - they will be automatically tracked by LFS according to the rules in `.gitattributes`.

## Architecture Notes

**Build pipeline:** `manuscript/*.md` → `sanitize_manuscript.sh` (removes TODO lines) → Pandoc (GFM to EPUB3) → `build/book.epub`

**Pandoc configuration:**
- Source format: GFM (GitHub Flavored Markdown)
- Target format: EPUB3 (default output format for .epub files)
- Metadata sourced from `metadata.yml`
- Styling from `epub.css` (applied with `--css` option)
- Syntax highlighting: kate theme (applied with `--syntax-highlighting` option)
- TOC: Automatically generated with `--toc` flag

**Sanitization:** The sanitize script uses BSD-compatible sed to remove lines matching `^\s*TODO(\s|\[|:)` before building, allowing authors to leave WIP notes that won't appear in the final EPUB.

## KDP Publishing

KDP only accepts manual EPUB uploads. To publish:
1. Create a release: `git tag v1.0.0 && git push origin v1.0.0`
2. Download `book.epub` from the GitHub Release
3. Upload to KDP Bookshelf
