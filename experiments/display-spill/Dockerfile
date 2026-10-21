# Build from the repository root so both feature folders are in the context:
#   docker build -f experiments/display-spill/Dockerfile -t tensorfold-dsv41:display-spill .
# Copy only the files the transforms read. A whole-folder COPY would also take a private bundle-launch-* folder
# (it holds a .env file) from the build context into the image.
ARG BASE_IMAGE=tensorfold-dsv41:07aa02585f64
FROM ${BASE_IMAGE}
COPY experiments/display-kv/apply.py experiments/display-kv/probe.py /opt/display-kv/
COPY experiments/display-kv/overlay/display_kv.py /opt/display-kv/overlay/display_kv.py
COPY experiments/display-spill/apply.py experiments/display-spill/compose.py /opt/display-spill/
COPY experiments/display-spill/overlay/spill_tier.py /opt/display-spill/overlay/spill_tier.py
RUN python /opt/display-spill/compose.py /opt/tensorfold /opt/display-kv/apply.py /opt/display-spill/apply.py
