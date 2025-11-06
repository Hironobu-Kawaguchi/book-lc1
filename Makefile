PANDOC ?= pandoc
OUT_DIR := build
SANITIZED_DIR := $(OUT_DIR)/sanitized
OUT := $(OUT_DIR)/book.epub

.PHONY: build clean

build: $(OUT)

$(OUT): metadata.yml epub.css $(wildcard manuscript/*.md) scripts/sanitize_manuscript.sh
	@bash scripts/sanitize_manuscript.sh
	@$(PANDOC) --from=gfm --standalone \
	  --metadata-file=metadata.yml --css=epub.css \
	  --syntax-highlighting=kate --toc -o $(OUT) $(SANITIZED_DIR)/*.md
	@echo "\n✓ Built $(OUT)"

clean:
	rm -rf $(OUT_DIR)

