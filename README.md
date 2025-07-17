```
docker run --rm -it \
  --cap-add=NET_ADMIN \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$HOME/humoid_data:/data" \
  -v "$HOME/humoid_data/nltk_data:/root/nltk_data" \
  -v "$HOME/humoid_data/weaviate:/root/.cache/weaviate-embedded" \
  humoid-chat
```
