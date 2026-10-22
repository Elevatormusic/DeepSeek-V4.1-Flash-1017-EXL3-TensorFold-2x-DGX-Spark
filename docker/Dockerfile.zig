# TensorFold Zig engine (the dsv41 family) for the GB10 (sm_121, aarch64, CUDA 13). It builds the branch dsv41-zig-dev of
# the fork at one commit with Zig 0.17.0, and puts three programs in the image:
#   /opt/tensorfold/native/bin/tensorfold-native   rank 0 (head): the OpenAI-compatible server
#   /opt/tensorfold/bin/tf-dsv41-lanes             rank 1 (worker): the follower
#   /opt/tensorfold/bin/tf-dsv41-rank-cache        builds the weight file of a rank (once for each view)
# The base image is the Python engine image of the recipe at ZIG_BASE_COMMIT: it supplies the CUDA toolkit (nvcc), NCCL,
# libibverbs and PyTorch. The Zig engine does not run the Python engine. Build with scripts/zig_build_image.sh
# (it builds the base image first when it is missing, and passes the arguments below).
ARG BASE_IMAGE=tensorfold-dsv41:3c45bdaeab7d
FROM ${BASE_IMAGE} AS src
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && /opt/zig/zig version
# xgrammar v0.2.8 at the one commit that the annotated tag v0.2.8 points to (a tag can move; a commit cannot). The
# submodules come from the commits that this tree records.
ARG XGRAMMAR_REF=97787376faee5ed8466cfad57c99855e4ce2f6aa
RUN git init -q /opt/xgrammar && cd /opt/xgrammar && git remote add origin https://github.com/mlc-ai/xgrammar && \
    git fetch -q --depth 1 origin "${XGRAMMAR_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${XGRAMMAR_REF}" && \
    git submodule update -q --init --recursive --depth 1
ARG TF_REPO=https://github.com/Elevatormusic/TensorFold.git
ARG TF_REF
RUN test -n "${TF_REF}" || { echo "set --build-arg TF_REF=<full commit>"; exit 1; } \
 && git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "${TF_REF}"

FROM src AS build
ARG SM=121
ARG JOBS=8
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=${SM} -Doptimize=ReleaseFast \
      --prefix /opt/tensorfold -j${JOBS} install native && \
    /opt/zig/zig build -Dxgrammar=/opt/xgrammar --prefix /opt/tensorfold -j${JOBS} grammar && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes && \
    mkdir -p /opt/tensorfold/native/lib && cp /opt/tensorfold/lib/libtfgrammar.so /opt/tensorfold/native/lib/ && \
    mkdir -p /opt/tensorfold/tools && cp -a tools/dsv41-zig /opt/tensorfold/tools/ && \
    mkdir -p /opt/tensorfold/share/doc/tensorfold && cp -a LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES /opt/tensorfold/share/doc/tensorfold/ && \
    ls -la /opt/tensorfold/bin /opt/tensorfold/native/bin /opt/tensorfold/lib 2>&1 | head -40 && rm -rf /root/.cache/zig

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
ARG TF_REF
LABEL tf.engine=zig tf.ref=${TF_REF}
ENV PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH}
WORKDIR /workspace
ENTRYPOINT []
