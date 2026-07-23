FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e .

# Runs the full Phase 1-9 cognitive loop on synthetic sensor data and prints
# a real summary (throughput, audit chain validity, guard verdicts) - the
# same command from the README's "for engineers" quick start, just packaged
# so there's nothing to install locally first.
CMD ["python3", "demo_run.py", "--frames", "2000", "--fresh"]
