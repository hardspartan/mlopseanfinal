install:
	pip install -r requirements.txt

train:
	python src/train.py --config config.yaml