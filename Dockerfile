# Use an appropriate Python base image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-tk \
    libgl1-mesa-glx \
    curl \
    iptables \
    dnsutils \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Install NLTK version 3.8.1 (or any desired version)
RUN pip install --no-cache-dir nltk==3.8.1

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire current directory contents into the container at /app
COPY . .

# Create a script to restrict outbound traffic
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "[INFO] Setting up iptables firewall..."\n\
\n\
# Flush existing rules\n\
iptables -F OUTPUT\n\
\n\
# Allow localhost and DNS\n\
iptables -A OUTPUT -o lo -j ACCEPT\n\
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT\n\
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT\n\
\n\
# Allow resolved IPs for Hugging Face and GitHub (objects.githubusercontent.com)\n\
resolve_and_allow() {\n\
  DOMAIN=$1\n\
  echo "[INFO] Resolving $DOMAIN..."\n\
  getent ahosts $DOMAIN | awk \"/STREAM/ {print \\$1}\" | sort -u | while read ip; do\n\
    echo "[INFO] Allowing $ip for $DOMAIN"\n\
    iptables -A OUTPUT -d \"$ip\" -j ACCEPT\n\
  done\n\
}\n\
\n\
resolve_and_allow huggingface.co\n\
resolve_and_allow objects.githubusercontent.com\n\
\n\
# Drop all other outbound traffic\n\
iptables -A OUTPUT -j REJECT\n\
\n\
echo "[INFO] Firewall active. Continuing..."\n\
\n\
# Download model if not present\n\
if [ ! -f /data/llama-2-7b-chat.ggmlv3.q8_0.bin ]; then\n\
  echo "Downloading model file..."\n\
  curl -L -o /data/llama-2-7b-chat.ggmlv3.q8_0.bin \\\n\
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --progress-bar\n\
  echo "Verifying model file..."\n\
  echo "3bfdde943555c78294626a6ccd40184162d066d39774bd2c98dae24943d32cc3  /data/llama-2-7b-chat.ggmlv3.q8_0.bin" | sha256sum -c -\n\
else\n\
  echo "Model file already exists, skipping download."\n\
fi\n\
\n\
ls -lh /data/llama-2-7b-chat.ggmlv3.q8_0.bin\n\
\n\
# Start the app\n\
export DISPLAY=:0\n\
exec python main.py' > /app/firewall_start.sh

# Make the firewall script executable
RUN chmod +x /app/firewall_start.sh

# Generate random API key and write config.json
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

# Entry point
CMD ["/app/firewall_start.sh"]
