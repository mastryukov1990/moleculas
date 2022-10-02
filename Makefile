PYTHON?=python
PYTHONPATH=./
LIB_DIR=lib

ls:
	ls $(LIB_DIR)

lint: ls
	pycodestyle lint $(LIB_DIR)

fmt:
	black  $(LIB_DIR) $(TESTS_DIR) $(BIN_DIR)