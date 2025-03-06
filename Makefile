clear:
	find . -name "__pycache__" | xargs rm -rf
	find . -name ".DS_Store" | xargs rm -rf

git-pull:
	git reset --hard HEAD
	git pull origin master

.PHONY: clear git-pull