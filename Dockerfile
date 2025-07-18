FROM python:3.11-slim-bookworm

# Set environment variable to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-tk \
    libgl1-mesa-glx \
    curl \
    iptables \
    dnsutils \
    openssl \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install NLTK
RUN pip install --no-cache-dir nltk==3.8.1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Generate secure vault passphrase and export to env script
RUN openssl rand -hex 32 > /app/.vault_pass && \
    echo "export VAULT_PASSPHRASE=$(cat /app/.vault_pass)" > /app/set_env.sh && \
    chmod +x /app/set_env.sh

# Create startup script using heredoc (reliable)
RUN cat << 'EOF' > /app/firewall_start.sh
#!/bin/bash
set -e
source /app/set_env.sh

echo "[INFO] Setting up iptables firewall..."
iptables -F OUTPUT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

resolve_and_allow() {
  DOMAIN=$1
  echo "[INFO] Resolving $DOMAIN..."
  getent ahosts $DOMAIN | awk '/STREAM/ {print $1}' | sort -u | while read ip; do
    clean_ip=$(echo $ip | tr -d '"')
    if [[ $clean_ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "[INFO] Allowing $clean_ip for $DOMAIN"
      iptables -A OUTPUT -d $clean_ip -j ACCEPT
    else
      echo "[WARN] Skipping invalid IP: $clean_ip"
    fi
  done
}

resolve_and_allow huggingface.co
resolve_and_allow objects.githubusercontent.com

iptables -A OUTPUT -j REJECT
echo "[INFO] Firewall active. Continuing..."

if [ ! -f /data/llama-2-7b-chat.ggmlv3.q8_0.bin ]; then
  echo "Downloading model file..."
  curl -L -o /data/llama-2-7b-chat.ggmlv3.q8_0.bin \
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --progress-bar
  echo "Verifying model file..."
  echo "3bfdde943555c78294626a6ccd40184162d066d39774bd2c98dae24943d32cc3  /data/llama-2-7b-chat.ggmlv3.q8_0.bin" | sha256sum -c -
else
  echo "Model file already exists, skipping download."
fi

ls -lh /data/llama-2-7b-chat.ggmlv3.q8_0.bin
export DISPLAY=:0
exec python main.py
EOF

# Make the script executable
RUN chmod +x /app/firewall_start.sh

# Generate config.json with randomized API key
RUN python -c 'import random, string, json; print(json.dumps({ \
  "DB_NAME": "story_generator.db", \
  "WEAVIATE_ENDPOINT": "http://localhost:8079", \
  "WEAVIATE_QUERY_PATH": "/v1/graphql", \
  "LLAMA_MODEL_PATH": "/data/llama-2-7b-chat.ggmlv3.q8_0.bin", \
  "IMAGE_GENERATION_URL": "http://127.0.0.1:7860/sdapi/v1/txt2img", \
  "MAX_TOKENS": 3999, \
  "CHUNK_SIZE": 1250, \
  "API_KEY": "".join(random.choices(string.ascii_letters + string.digits, k=32)), \
  "WEAVIATE_API_URL": "http://localhost:8079/v1/objects", \
  "ELEVEN_LABS_KEY": "apikyhere" \
}))' > /app/config.json

# Entrypoint to the assistant app
CMD ["/app/firewall_start.sh"]
