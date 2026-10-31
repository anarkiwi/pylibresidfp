FROM python:3.12-slim AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build git clang-format \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "scikit-build-core>=0.10" "pybind11>=2.13" numpy pytest pytest-cov \
    pytest-xdist black pylint "vagrant-moses[cpu]"

FROM base AS test
WORKDIR /src
COPY . .
RUN pip install --no-build-isolation -e .
RUN black --check src tests \
    && pylint src tests \
    && clang-format --dry-run --Werror src/binding.cpp \
    && OMP_NUM_THREADS=2 pytest -n auto --cov=pylibresidfp --cov-fail-under=85
