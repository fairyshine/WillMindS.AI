clear:
	find . -name "__pycache__" | xargs rm -rf
	find . -name ".DS_Store" | xargs rm -rf

.PHONY: clear