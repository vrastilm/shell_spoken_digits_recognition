create_shell:
	pipenv shell
	pipenv sync

develop_model:
	clear;
	python model_training/train_model.py -d $(shell readlink -f Data) -tdc  $(shell readlink -f model_training/train_data_cache.npy) -vdc $(shell readlink -f model_training/validation_data_cache.npy)

build:
	clear;
	pip install --editable shell_asr/

run:
	clear;
	shell_asr

test:
	clear;
	python -m unittest discover -s shell_asr/test