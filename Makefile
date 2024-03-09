run:
	FLASK_APP=src/_molotov/flask_api.py flask run

install:
	python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt

