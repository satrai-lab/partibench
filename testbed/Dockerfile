FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the partibench/ root so we can reference files from both
# testbed/ (modified) and the original source tree (unchanged).
COPY testbed/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Header-only net_rules.tsv so core.py import never crashes at pod startup.
# The real per-node file is mounted over this via a ConfigMap by deploy_hardcoded_split.sh.
RUN printf 'DEST_NAME\tIP\tBANDWIDTH\tDELAY\n' > net_rules.tsv

# Modified files — live in testbed/
COPY testbed/core.py          .
COPY testbed/local_deploy.py  .
COPY testbed/place.py         .
COPY testbed/merge_bench.py   .
COPY data/                      ./assets/

# Unchanged files — referenced directly from the original source tree
COPY benchmarking/benchmark.py       .
COPY benchmarking/memory_profiler.py .

EXPOSE 6999

CMD ["python", "local_deploy.py"]
