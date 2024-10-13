.PHONY: install run

install:
	python3 -m pip install -r requirements.txt

run:
	lsof -ti:3000 | xargs kill -9 || true
	python3 app.py