PYTHON?=python
PYTHONPATH=./
LIB_DIR=lib
TESTS_DIR=tests

ls:
	ls $(LIB_DIR)

lint:
	flake8 ./$(LIB_DIR)

fmt:
	black  $(LIB_DIR) $(TESTS_DIR) $(BIN_DIR)


test:
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest $(TESTS_DIR) -v