SYSTEM_PYTHON ?= python3
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3
PIP ?= $(VENV)/bin/pip3

TRAIN_DATA ?= data/dataset_train.csv
DESCRIBE_TEST ?= test/describe_pandas.py

.PHONY: all help venv install describe_test clean fclean re

# Run the first mandatory program.
all: describe_test

# Show available commands.
help:
	@echo "Available targets:"
	@echo "  make all       - run the current mandatory workflow"
	@echo "  make venv      - create the local Python virtual environment"
	@echo "  make install   - install Python dependencies if requirements.txt exists"
	@echo "  make describe_test  - display numerical statistics for the train dataset"
	@echo "  make clean     - remove Python cache files"
	@echo "  make fclean    - clean and remove the virtual environment"
	@echo "  make re        - rebuild the environment, then run all"

# Create the local Python virtual environment.
venv: $(VENV)/bin/python3

$(VENV)/bin/python3:
	$(SYSTEM_PYTHON) -m venv $(VENV)

# Install dependencies only after we add a requirements.txt.
install: venv
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	else \
		echo "No requirements.txt yet."; \
	fi

# First mandatory program: dataset numerical summary.
describe_test: venv
	$(PYTHON) $(DESCRIBE_TEST) $(TRAIN_DATA)

# Remove Python cache files.
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Remove generated local environment.
fclean: clean
	rm -rf $(VENV)

# Rebuild and run again.
re: fclean all
