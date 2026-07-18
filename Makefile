SYSTEM_PYTHON ?= python3
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3
PIP ?= $(VENV)/bin/pip3

TRAIN_DATA ?= data/dataset_train.csv
DESCRIBE ?= src/describe.py
HISTOGRAM ?= src/histogram.py
SCATTER_PLOT ?= src/scatter_plot.py
DESCRIBE_TEST ?= test/describe_pandas.py
DISPLAY_DB ?= test/display_db.py
DB_JSON ?= db.json

.PHONY: all help venv install describe histogram scatter_plot describe_test display_db clean fclean re

# Run the first mandatory program.
all: describe

# Show available commands.
help:
	@echo "Available targets:"
	@echo "  make all             - run the current mandatory workflow"
	@echo "  make venv            - create the local Python virtual environment"
	@echo "  make install         - install Python dependencies if requirements.txt exists"
	@echo "  make describe        - run the current describe implementation"
	@echo "  make histogram       - display one feature histogram"
	@echo "  make scatter_plot    - display one simple scatter plot"
	@echo "  make describe_test   - run the pandas reference describe"
	@echo "  make display_db      - write database JSON to db.json"
	@echo "  make clean           - remove Python cache files"
	@echo "  make fclean          - clean and remove the virtual environment"
	@echo "  make re              - rebuild the environment, then run all"

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

# First mandatory program, built slowly.
describe: venv
	$(PYTHON) $(DESCRIBE) $(TRAIN_DATA)

# Display one feature histogram.
histogram: venv
	$(PYTHON) $(HISTOGRAM) $(TRAIN_DATA)

# Display one simple scatter plot.
scatter_plot: venv
	$(PYTHON) $(SCATTER_PLOT) $(TRAIN_DATA)

# Pandas reference output for comparison only.
describe_test: venv
	$(PYTHON) $(DESCRIBE_TEST) $(TRAIN_DATA)

# Debug helper: write parsed database as JSON.
display_db: venv
	PYTHONPATH=src $(PYTHON) $(DISPLAY_DB) $(TRAIN_DATA) > $(DB_JSON)

# Remove Python cache files.
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Remove generated local environment.
fclean: clean
	rm -rf $(VENV)

# Rebuild and run again.
re: fclean all
