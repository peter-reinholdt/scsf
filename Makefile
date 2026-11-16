PYTHON ?= python3
.PHONY: all test
all:
	$(PYTHON) setup.py build_ext --inplace
test:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) -m pytest -q
