FROM rocm/dev-ubuntu-24.04:7.2-complete

RUN apt-get update && apt-get install -y --no-install-recommends zstd curl ca-certificates \
    && curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar x --zstd -C /usr \
    && curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst | tar x --zstd -C /usr \
    && rm -rf /var/lib/apt/lists/*

CMD ["ollama", "serve"]
