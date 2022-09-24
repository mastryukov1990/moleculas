PYTHON?=python3
PYTHONPATH=./
LIB_DIR=lib

ls:
	ls $(LIB_DIR)

lint: ls
	pystyle lint $(LIB_DIR)

fmt:
	pystyle fmt $(LIB_DIR) $(TESTS_DIR) $(BIN_DIR)