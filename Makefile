.PHONY: build clean sanitize

SANITIZED_DIR=build/sanitized

sanitize:
	bash scripts/sanitize_manuscript.sh

build: sanitize
	pandoc \
		--from=gfm \
		--standalone \
		--metadata-file=metadata.yml \
		--output=build/book.epub \
		--toc \
		--css=epub.css \
		--syntax-highlighting=kate \
		$(SANITIZED_DIR)/*.md

clean:
	rm -rf build
