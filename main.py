import tkinter as tk
import customtkinter
import threading
import os
import sqlite3 
import logging
import numpy as np
import base64
import queue
import uuid
import requests
import io
import sys
import random
import re
import uvicorn
import json
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
from os import path
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from collections import Counter
from summa import summarizer
import nltk
from textblob import TextBlob
from weaviate.util import generate_uuid5
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from datetime import datetime
from weaviate.embedded import EmbeddedOptions
import weaviate
import pennylane as qml
import psutil
import webcolors
from nltk.tokenize import word_tokenize
import colorsys
import hmac
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type
import bleach


ARGON2_TIME_COST_DEFAULT = 3          
ARGON2_MEMORY_COST_KIB    = 262144   
ARGON2_PARALLELISM        = max(1, min(4, os.cpu_count() or 1))
ARGON2_HASH_LEN           = 32

VAULT_PASSPHRASE_ENV      = "VAULT_PASSPHRASE"
VAULT_VERSION             = 1        
DATA_KEY_VERSION          = 1         
VAULT_NONCE_SIZE          = 12    
DATA_NONCE_SIZE           = 12

def _aad_str(*parts: str) -> bytes:
    return ("|".join(parts)).encode("utf-8")

customtkinter.set_appearance_mode("Dark")

nltk.data.path.append("/root/nltk_data")


def download_nltk_data():
    try:

        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger'
        }

        for path_, package in resources.items():
            try:
                nltk.data.find(path_)
                print(f"'{package}' already downloaded.")
            except LookupError:
                nltk.download(package)
                print(f"'{package}' downloaded successfully.")

    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


download_nltk_data()
        
client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
executor = ThreadPoolExecutor(max_workers=5)
bundle_dir = path.abspath(path.dirname(__file__))
path_to_config = path.join(bundle_dir, 'config.json')
model_path = "/data/llama-2-7b-chat.ggmlv3.q8_0.bin"
logo_path = path.join(bundle_dir, 'logo.png')

API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def load_config(file_path=path_to_config):
    with open(file_path, 'r') as file:
        return json.load(file)


q = queue.Queue()
logger = logging.getLogger(__name__)
config = load_config()

SAFE_ALLOWED_TAGS: list[str] = []
SAFE_ALLOWED_ATTRS: dict[str, list[str]] = {}
SAFE_ALLOWED_PROTOCOLS: list[str] = []


_CONTROL_WHITELIST = {'\n', '\r', '\t'}

def _strip_control_chars(s: str) -> str:

    return ''.join(ch for ch in s if ch.isprintable() or ch in _CONTROL_WHITELIST)

def sanitize_text(
    text: str,
    *,
    max_len: int = 4000,
    strip: bool = True,
) -> str:

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text[:max_len]
    text = _strip_control_chars(text)
    cleaned = bleach.clean(
        text,
        tags=SAFE_ALLOWED_TAGS,
        attributes=SAFE_ALLOWED_ATTRS,
        protocols=SAFE_ALLOWED_PROTOCOLS,
        strip=strip,
        strip_comments=True,
    )
    return cleaned

_PROMPT_INJECTION_PAT = re.compile(
    r'(?is)(?:^|\n)\s*(system:|assistant:|ignore\s+previous|do\s+anything|jailbreak\b).*'
)

def sanitize_for_prompt(text: str, *, max_len: int = 2000) -> str:

    cleaned = sanitize_text(text, max_len=max_len)
    cleaned = _PROMPT_INJECTION_PAT.sub('', cleaned)
    return cleaned.strip()

def sanitize_for_graphql_string(s: str, *, max_len: int = 512) -> str:

    s = sanitize_text(s, max_len=max_len)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return s

DB_NAME = config['DB_NAME']
API_KEY = config['API_KEY']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_QUERY_PATH = config['WEAVIATE_QUERY_PATH']
app = FastAPI()
class SecureKeyManager:


    def __init__(
        self,
        method="argon2id",
        vault_path="secure/key_vault.json",
        time_cost: int = ARGON2_TIME_COST_DEFAULT,
        memory_cost: int = ARGON2_MEMORY_COST_KIB,
        parallelism: int = ARGON2_PARALLELISM,
        hash_len: int = ARGON2_HASH_LEN,
    ):
        self.method       = method
        self.vault_path   = vault_path
        self.time_cost    = time_cost
        self.memory_cost  = memory_cost
        self.parallelism  = parallelism
        self.hash_len     = hash_len


        self._ensure_vault()


        vault_meta = self._load_vault()


        self.active_version = vault_meta["active_version"]


        self._keys = {
            int(kv["version"]): base64.b64decode(kv["master_secret"])
            for kv in vault_meta["keys"]
        }


        self._derived_keys = {}
        vault_salt = base64.b64decode(vault_meta["salt"])
        for ver, master_secret in self._keys.items():
            self._derived_keys[ver] = self._derive_key(master_secret, vault_salt)


    def _get_passphrase(self) -> bytes:

        pw = os.getenv(VAULT_PASSPHRASE_ENV)
        if pw is None or pw == "":
            pw = base64.b64encode(os.urandom(32)).decode()
            logging.warning(
                "[SecureKeyManager] VAULT_PASSPHRASE not set; generated ephemeral key. "
                "Vault will not be readable across restarts!"
            )
        return pw.encode("utf-8")

    def _derive_vault_key(self, passphrase: bytes, salt: bytes) -> bytes:

        return hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=max(self.time_cost, 3),   
            memory_cost=max(self.memory_cost, 262144),
            parallelism=max(self.parallelism, 1),
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _derive_key(self, master_secret: bytes, salt: bytes) -> bytes:

        return hash_secret_raw(
            secret=master_secret,
            salt=salt,
            time_cost=self.time_cost,
            memory_cost=self.memory_cost,
            parallelism=self.parallelism,
            hash_len=self.hash_len,
            type=Type.ID,
        )


    def _ensure_vault(self):

        if not os.path.exists("secure"):
            os.makedirs("secure", exist_ok=True)
        if os.path.exists(self.vault_path):
            return


        salt          = os.urandom(16)
        master_secret = os.urandom(32) 

        vault_body = {
            "version": VAULT_VERSION,
            "active_version": DATA_KEY_VERSION,
            "keys": [
                {
                    "version": DATA_KEY_VERSION,
                    "master_secret": base64.b64encode(master_secret).decode(),
                    "created": datetime.utcnow().isoformat() + "Z",
                }
            ],
            "salt": base64.b64encode(salt).decode(),
        }

        self._write_encrypted_vault(vault_body)

    def _write_encrypted_vault(self, vault_body: dict):

        plaintext = json.dumps(vault_body, indent=2).encode("utf-8")
        salt      = base64.b64decode(vault_body["salt"])

        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        nonce      = os.urandom(VAULT_NONCE_SIZE)
        ct         = aesgcm.encrypt(nonce, plaintext, _aad_str("vault", str(vault_body["version"])))

        on_disk = {
            "vault_format": VAULT_VERSION,
            "salt": vault_body["salt"],  
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ct).decode(),
        }
        with open(self.vault_path, "w") as f:
            json.dump(on_disk, f, indent=2)

    def _load_vault(self) -> dict:

        with open(self.vault_path, "r") as f:
            data = json.load(f)

        if "ciphertext" not in data:
            salt          = base64.b64decode(data["salt"])
            master_secret = base64.b64decode(data["master_secret"])
            vault_body = {
                "version": VAULT_VERSION,
                "active_version": DATA_KEY_VERSION,
                "keys": [
                    {
                        "version": DATA_KEY_VERSION,
                        "master_secret": base64.b64encode(master_secret).decode(),
                        "created": datetime.utcnow().isoformat() + "Z",
                    }
                ],
                "salt": base64.b64encode(salt).decode(),
            }
            self._write_encrypted_vault(vault_body)
            return vault_body

 
        salt      = base64.b64decode(data["salt"])
        nonce     = base64.b64decode(data["nonce"])
        ct        = base64.b64decode(data["ciphertext"])

        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        plaintext  = aesgcm.decrypt(nonce, ct, _aad_str("vault", str(VAULT_VERSION)))
        return json.loads(plaintext.decode("utf-8"))

    def encrypt(
        self,
        plaintext: str,
        *,
        aad: bytes = None,
        key_version: int = None,
    ) -> str:

        if plaintext is None:
            plaintext = ""
        if key_version is None:
            key_version = self.active_version
        if aad is None:
            aad = _aad_str("global", f"k{key_version}")

        key    = self._derived_keys[key_version]
        aesgcm = AESGCM(key)
        nonce  = os.urandom(DATA_NONCE_SIZE)
        ct     = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)

        token = {
            "v": VAULT_VERSION,        
            "k": key_version,          
            "aad": aad.decode("utf-8"),
            "n": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
        }
        return json.dumps(token, separators=(",", ":"))

    def decrypt(self, token: str) -> str:

        if not token:
            return ""

        if token.startswith("{"):
            try:
                meta = json.loads(token)
            except Exception:
                logging.warning("[SecureKeyManager] Invalid JSON token; returning raw.")
                return token

            v   = int(meta.get("v", 1))
            ver = int(meta.get("k", self.active_version))
            aad = meta.get("aad", "global").encode()
            n   = base64.b64decode(meta["n"])
            ct  = base64.b64decode(meta["ct"])

            key = self._derived_keys.get(ver)
            if key is None:
                raise ValueError(f"No key for version {ver}; cannot decrypt.")
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(n, ct, aad)
            return pt.decode("utf-8")


        try:
            raw   = base64.b64decode(token.encode())
            nonce = raw[:DATA_NONCE_SIZE]
            ct    = raw[DATA_NONCE_SIZE:]
            key   = self._derived_keys[self.active_version]
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(nonce, ct, None)
            return pt.decode("utf-8")
        except Exception as e:
            logging.warning(f"[SecureKeyManager] Legacy decrypt failed: {e}")
            return token


    def add_new_key_version(self) -> int:

        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1

        master_secret = os.urandom(32)
        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(master_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })

        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)


        self._keys[new_version] = master_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(master_secret, salt)
        self.active_version = new_version
        logging.info(f"[SecureKeyManager] Installed new key version {new_version}.")
        return new_version

    def rotate_and_migrate_storage(self, migrate_func):

        new_ver = self.add_new_key_version()
        try:
            migrate_func(self)
        except Exception as e:
            logging.error(f"[SecureKeyManager] Migration failed after key rotation: {e}")
            raise
        logging.info(f"[SecureKeyManager] Migration to key v{new_ver} complete.")


crypto = SecureKeyManager()  

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)


api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

def build_record_aad(user_id: str, *, source: str, table: str = "", cls: str = "") -> bytes:

    context_parts = [source]
    if table:
        context_parts.append(table)
    if cls:
        context_parts.append(cls)
    context_parts.append(user_id)
    return _aad_str(*context_parts)


def compute_text_embedding(text: str) -> list[float]:

    return [0.0] * 384

def generate_uuid_for_weaviate(identifier, namespace=''):
    if not identifier:
        raise ValueError("Identifier for UUID generation is empty or None")


    if not namespace:
        namespace = str(uuid.uuid4())

    try:
        return generate_uuid5(namespace, identifier)
    except Exception as e:
        logger.error(f"Error generating UUID: {e}")
        raise


def is_valid_uuid(uuid_to_test, version=5):
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
        return str(uuid_obj) == uuid_to_test
    except ValueError:
        return False
    
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def rgb_quantum_gate(r, g, b, cpu_usage):
    qml.RX(r * 3.1415 * cpu_usage, wires=0)
    qml.RY(g * 3.1415 * cpu_usage, wires=1)
    qml.RZ(b * 3.1415 * cpu_usage, wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return (
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2)),
    )
  
def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")

def generate_quantum_state(rgb=None):
    if rgb is None:
        rgb = (128, 128, 128)

    try:
        cpu = psutil.cpu_percent(interval=0.3) / 100.0
        r, g, b = [c / 255 for c in rgb]
        z0, z1, z2 = rgb_quantum_gate(r, g, b, cpu)
        return (
            f"[QuantumGate] CPU={cpu*100:.0f}% │ "
            f"Colors={rgb} → Z=({z0:.3f},{z1:.3f},{z2:.3f})"
        )
    except Exception as e:
        logger.error(f"Error in generate_quantum_state: {e}")
        return "[QuantumGate] error"

def get_current_multiversal_time():
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    x, y, z, t = 34, 76, 12, 5633
    return f"X:{x}, Y:{y}, Z:{z}, T:{t}, Time:{current_time}"

def extract_rgb_from_text(text):

    if not text or not isinstance(text, str):
        return (128, 128, 128) 

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity 
    subjectivity = blob.sentiment.subjectivity 

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    word_count = len(tokens)
    sentence_count = len(blob.sentences) or 1
    avg_sentence_length = word_count / sentence_count

    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))

    punctuation_density = sum(1 for ch in text if ch in ',;:!?') / max(1, word_count)

    valence = polarity 

    arousal = (verb_count + adv_count) / max(1, word_count)

    dominance = (adj_count + 1) / (noun_count + 1) 


    hue_raw = ((1 - valence) * 120 + dominance * 20) % 360
    hue = hue_raw / 360.0

    saturation = min(1.0, max(0.2, 0.25 + 0.4 * arousal + 0.2 * subjectivity + 0.15 * (dominance - 1)))

    brightness = max(0.2, min(1.0,
        0.9 - 0.03 * avg_sentence_length + 0.2 * punctuation_density
    ))

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (int(r * 255), int(g * 255), int(b * 255))
  
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                response TEXT,
                response_time TEXT
            )
        """)
        conn.commit()
        conn.close()

        interaction_history_class = {
            "class": "InteractionHistory",
            "properties": [
                {"name": "user_id", "dataType": ["string"]},
                {"name": "response", "dataType": ["string"]},
                {"name": "response_time", "dataType": ["string"]}
            ]
        }

        existing_classes = client.schema.get()['classes']
        if not any(cls['class'] == 'InteractionHistory' for cls in existing_classes):
            client.schema.create_class(interaction_history_class)

    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

def save_user_message(user_id, user_input):
    logger.info(f"[save_user_message] user_id={user_id}")
    if not user_input:
        logger.warning("User input is empty.")
        return

    try:

        user_input = sanitize_text(user_input, max_len=4000)

        response_time = get_current_multiversal_time()

 
        aad_sql  = build_record_aad(user_id=user_id, source="sqlite", table="local_responses")
        aad_weav = build_record_aad(user_id=user_id, source="weaviate", cls="InteractionHistory")

        encrypted_input_sql  = crypto.encrypt(user_input, aad=aad_sql)
        encrypted_input_weav = crypto.encrypt(user_input, aad=aad_weav)


        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (user_id, encrypted_input_sql, response_time)
            )
            conn.commit()


        embedding = compute_text_embedding(user_input)


        encrypted_weaviate_object = {
            "user_id": user_id,
            "user_message": encrypted_input_weav,
            "response_time": response_time,
        }
        generated_uuid = generate_uuid5(user_id, user_input)

        response = requests.post(
            'http://127.0.0.1:8079/v1/objects',
            json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": encrypted_weaviate_object,
                "vector": embedding,
            }
        )
        if response.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {response.status_code} {response.text}")

    except Exception as e:
        logger.exception(f"Exception in save_user_message: {e}")



def save_bot_response(bot_id: str, bot_response: str):
    logger.info(f"[save_bot_response] bot_id={bot_id}")

    if not bot_response:
        logger.warning("Bot response is empty.")
        return

    try:

        bot_response = sanitize_text(bot_response, max_len=4000)

        response_time = get_current_multiversal_time()

  
        aad_sql  = build_record_aad(user_id=bot_id, source="sqlite", table="local_responses")
        enc_sql  = crypto.encrypt(bot_response, aad=aad_sql)
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (bot_id, enc_sql, response_time)
            )
            conn.commit()

 
        aad_weav = build_record_aad(user_id=bot_id, source="weaviate", cls="InteractionHistory")
        enc_weav = crypto.encrypt(bot_response, aad=aad_weav)

        props = {
            "user_id": bot_id,
            "ai_response": enc_weav,
            "response_time": response_time,
        }

        embedding = compute_text_embedding(bot_response)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        generated_uuid = generate_uuid5(bot_id, bot_response)

        resp = requests.post(
            "http://127.0.0.1:8079/v1/objects",
            json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": props,
                "vector": embedding,
            },
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {resp.status_code} {resp.text}")

    except Exception as e:
        logger.exception(f"Exception in save_bot_response: {e}")



class UserInput(BaseModel):
    message: str

@app.post("/process/")
def process_input(user_input: UserInput, api_key: str = Depends(get_api_key)):
    try:
        safe_msg = sanitize_for_prompt(user_input.message, max_len=4000)
        response = llama_generate(safe_msg, client)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=3900,
)


def is_code_like(chunk):
   code_patterns = r'\b(def|class|import|if|else|for|while|return|function|var|let|const|print)\b|[\{\}\(\)=><\+\-\*/]'
   return bool(re.search(code_patterns, chunk))


def determine_token(chunk, memory, max_words_to_check=500):
   combined_chunk = f"{memory} {chunk}"
   if not combined_chunk:
       return "[attention]"

   if is_code_like(combined_chunk):
       return "[code]"

   words = word_tokenize(combined_chunk)[:max_words_to_check]
   tagged_words = pos_tag(words)

   pos_counts = Counter(tag[:2] for _, tag in tagged_words)
   most_common_pos, _ = pos_counts.most_common(1)[0]

   if most_common_pos == 'VB':
       return "[action]"
   elif most_common_pos == 'NN':
       return "[subject]"
   elif most_common_pos in ['JJ', 'RB']:
       return "[description]"
   else:
       return "[general]"


def find_max_overlap(chunk, next_chunk):
   max_overlap = min(len(chunk), 240)
   return next((overlap for overlap in range(max_overlap, 0, -1) if chunk.endswith(next_chunk[:overlap])), 0)


def truncate_text(text, max_words=100):
   return ' '.join(text.split()[:max_words])


def fetch_relevant_info(chunk, client, user_input):
    if not user_input:
        logger.error("User input is None or empty.")
        return ""


    chunk_clean = sanitize_text(chunk, max_len=1024)
    user_input_clean = sanitize_text(user_input, max_len=1024)

    summarized_chunk = summarizer.summarize(chunk_clean)
    query_chunk = summarized_chunk if summarized_chunk else chunk_clean

    if not query_chunk:
        logger.error("Query chunk is empty.")
        return ""

    query = {
        "query": {
            "nearText": {
                "concepts": [user_input_clean],
                "certainty": 0.7
            }
        }
    }

    try:
        response = client.query.raw(json.dumps(query))
        logger.debug(f"Query sent: {json.dumps(query)}")
        logger.debug(f"Response received: {response}")

        if (
            response and
            'data' in response and
            'Get' in response['data'] and
            'InteractionHistory' in response['data']['Get'] and
            response['data']['Get']['InteractionHistory']
        ):
            interaction = response['data']['Get']['InteractionHistory'][0]
            user_msg = interaction.get('user_message', '')
            ai_resp  = interaction.get('ai_response', '')

            return f"{user_msg} {ai_resp}"
        else:
            logger.warning("No relevant data found in Weaviate response.")
            return ""

    except Exception as e:
        logger.error(f"Weaviate query failed: {e}")
        return ""


def llama_generate(prompt, weaviate_client=None, user_input=None):
   config = load_config()
   max_tokens = config.get('MAX_TOKENS', 2500)
   chunk_size = config.get('CHUNK_SIZE', 358)
   try:
       prompt_chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]
       responses = []
       last_output = ""
       memory = ""

       for i, current_chunk in enumerate(prompt_chunks):
           relevant_info = fetch_relevant_info(current_chunk, weaviate_client, user_input)
           combined_chunk = f"{relevant_info} {current_chunk}"
           token = determine_token(combined_chunk, memory)
           output = tokenize_and_generate(combined_chunk, token, max_tokens, chunk_size)

           if output is None:
               logger.error(f"Failed to generate output for chunk: {combined_chunk}")
               continue

           if i > 0 and last_output:
               overlap = find_max_overlap(last_output, output)
               output = output[overlap:]

           memory += output
           responses.append(output)
           last_output = output

       final_response = ''.join(responses)
       return final_response if final_response else None
   except Exception as e:
       logger.error(f"Error in llama_generate: {e}")
       return None


def tokenize_and_generate(chunk, token, max_tokens, chunk_size):
   try:
       inputs = llm(f"[{token}] {chunk}", max_tokens=min(max_tokens, chunk_size))
       if inputs is None or not isinstance(inputs, dict):
           logger.error(f"Llama model returned invalid output for input: {chunk}")
           return None

       choices = inputs.get('choices', [])
       if not choices or not isinstance(choices[0], dict):
           logger.error("No valid choices in Llama output")
           return None

       return choices[0].get('text', '')
   except Exception as e:
       logger.error(f"Error in tokenize_and_generate: {e}")
       return None

def extract_verbs_and_nouns(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        verbs_and_nouns = [word for word, tag in tagged_words if tag.startswith('VB') or tag.startswith('NN')]
        return verbs_and_nouns

    except Exception as e:
        print(f"Error in extract_verbs_and_nouns: {e}")
        return []

def try_decrypt(value):
    try:
        return crypto.decrypt(value)
    except Exception as e:
        logger.warning(f"[decryption] Could not decrypt value: {e}")
        return value

class App(customtkinter.CTk):

    @staticmethod
    def _encrypt_field(value: str) -> str:
        try:
            return crypto.encrypt(value if value is not None else "")
        except Exception as e:
            logger.error(f"[encrypt] Failed to encrypt value: {e}")
            return value if value is not None else ""

    @staticmethod
    def _decrypt_field(value: str) -> str:
        if value is None:
            return ""
        try:
            return crypto.decrypt(value)
        except Exception as e:
            logger.warning(f"[decrypt] Could not decrypt value (returning raw): {e}")
            return value

    def __init__(self, user_identifier):
        super().__init__()
        self.user_id = user_identifier
        self.bot_id = "bot"
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)

    def retrieve_past_interactions(self, user_input, result_queue):

        try:
            keywords = extract_verbs_and_nouns(user_input)
            concepts_query = ' '.join(keywords)

            user_message, ai_response = self.fetch_relevant_info_internal(concepts_query)

            if user_message and ai_response:
                combo = f"{user_message} {ai_response}"
                summarized_interaction = summarizer.summarize(combo) or combo
                sentiment = TextBlob(summarized_interaction).sentiment.polarity
                processed_interaction = {
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "summarized_interaction": summarized_interaction,
                    "sentiment": sentiment
                }
                result_queue.put([processed_interaction])
            else:
                logger.info("No relevant interactions found for the given user input.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])

    def fetch_relevant_info_internal(self, chunk):
        if self.client:
            safe_chunk = sanitize_for_graphql_string(chunk, max_len=256)

            query = f"""
            {{
                Get {{
                    InteractionHistory(
                        nearText: {{
                            concepts: ["{safe_chunk}"],
                            certainty: 0.7
                        }}
                        limit: 1
                    ) {{
                        user_message
                        ai_response
                        response_time
                    }}
                }}
            }}
            """
            try:
                response = self.client.query.raw(query)
                results = (
                    response.get('data', {})
                            .get('Get', {})
                            .get('InteractionHistory', [])
                )
                if not results:
                    return "", ""
                interaction = results[0]

                user_msg_raw = self._decrypt_field(interaction.get('user_message', ''))
                ai_resp_raw  = self._decrypt_field(interaction.get('ai_response', ''))


                user_msg = sanitize_text(user_msg_raw, max_len=4000)
                ai_resp  = sanitize_text(ai_resp_raw, max_len=4000)

                return user_msg, ai_resp

            except Exception as e:
                logger.error(f"Weaviate query failed: {e}")
                return "", ""
        return "", ""

    def fetch_interactions(self):
        try:
            gql = """
            {
                Get {
                    InteractionHistory(
                        sort: [{path: "response_time", order: desc}],
                        limit: 15
                    ) {
                        user_message
                        ai_response
                        response_time
                    }
                }
            }
            """
            response = self.client.query.raw(gql)
            results = (
                response.get('data', {})
                        .get('Get', {})
                        .get('InteractionHistory', [])
            )
            decrypted = []
            for interaction in results:
                u_raw = self._decrypt_field(interaction.get('user_message', ''))
                a_raw = self._decrypt_field(interaction.get('ai_response', ''))
                decrypted.append({
                    'user_message' : sanitize_text(u_raw, max_len=4000),
                    'ai_response'  : sanitize_text(a_raw, max_len=4000),
                    'response_time': interaction.get('response_time', '')
                })
            return decrypted
        except Exception as e:
            logger.error(f"Error fetching interactions from Weaviate: {e}")
            return []


    def process_response_and_store_in_weaviate(self, user_message, ai_response):

        try:
            response_blob = TextBlob(ai_response)
            keywords = response_blob.noun_phrases
            sentiment = response_blob.sentiment.polarity
            enhanced_keywords = set()
            for phrase in keywords:
                enhanced_keywords.update(phrase.split())

            interaction_object = {
                "user_message": self._encrypt_field(user_message),
                "ai_response":  self._encrypt_field(ai_response),
                "keywords":     list(enhanced_keywords),
                "sentiment":    sentiment
            }

            interaction_uuid = str(uuid.uuid4())

            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )
            logger.info(f"Interaction stored in Weaviate with UUID: {interaction_uuid}")

        except Exception as e:            
            logger.error(f"Error storing interaction in Weaviate: {e}")

    def create_interaction_history_object(self, user_message, ai_response):

        interaction_object = {
            "user_message": self._encrypt_field(user_message),
            "ai_response":  self._encrypt_field(ai_response)
        }

        try:
            object_uuid = str(uuid.uuid4())
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            logger.info(f"Interaction history object created with UUID: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating interaction history object in Weaviate: {e}")


    def map_keywords_to_weaviate_classes(self, keywords, context):
        try:
            summarized_context = summarizer.summarize(context) or context
        except Exception as e:
            logger.error(f"Error in summarizing context: {e}")
            summarized_context = context

        try:
            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            sentiment = TextBlob("").sentiment

        positive_class_mappings = {
            "keyword1": "PositiveClassA",
            "keyword2": "PositiveClassB",
        }
        negative_class_mappings = {
            "keyword1": "NegativeClassA",
            "keyword2": "NegativeClassB",
        }
        default_mapping = {
            "keyword1": "NeutralClassA",
            "keyword2": "NeutralClassB",
        }

        if sentiment.polarity > 0:
            mapping = positive_class_mappings
        elif sentiment.polarity < 0:
            mapping = negative_class_mappings
        else:
            mapping = default_mapping
            
        mapped_classes = {}
        for keyword in keywords:
            try:
                if keyword in mapping:
                    mapped_classes[keyword] = mapping[keyword]
            except KeyError as e:
                logger.error(f"Error in mapping keyword '{keyword}': {e}")

        return mapped_classes
    def generate_response(self, user_input):
        try:
            if not user_input:
                logger.error("User input is None or empty.")
                return

            user_id = self.user_id
            bot_id  = self.bot_id


            safe_user_input = sanitize_for_prompt(user_input, max_len=4000)

            save_user_message(user_id, safe_user_input)

            include_past_context = "[pastcontext]" in safe_user_input
            cleaned_user_input = (
                safe_user_input
                .replace("[pastcontext]", "")
                .replace("[/pastcontext]", "")
            )
            past_context = ""

            if include_past_context:
                result_queue = queue.Queue()
                self.retrieve_past_interactions(cleaned_user_input, result_queue)
                past_interactions = result_queue.get()
                if past_interactions:
                    past_context_combined = "\n".join(
                        [f"User: {interaction['user_message']}\nAI: {interaction['ai_response']}"
                         for interaction in past_interactions]
                    )

                    past_context = sanitize_for_prompt(past_context_combined, max_len=1500)

            rgb = extract_rgb_from_text(cleaned_user_input)
            quantum_state = generate_quantum_state(rgb=rgb)
            complete_prompt = f"{quantum_state}\n{past_context}\nUser: {cleaned_user_input}"
            logger.info(f"Generating response for prompt: {complete_prompt}")

            response = llama_generate(complete_prompt, self.client)
            if response:
                logger.info(f"Generated response: {response}")
                save_bot_response(bot_id, response)
                self.process_generated_response(response)
            else:
                logger.error("No response generated by llama_generate")

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")

    def process_generated_response(self, response_text):

        try:
            self.response_queue.put({'type': 'text', 'data': response_text})
        except Exception as e:
            logger.error(f"Error in process_generated_response: {e}")

    def run_async_in_thread(self, coro_func, user_input, result_queue):
        try:
            coro_func(user_input, result_queue)
        except Exception as e:
            logger.error(f"Error running function in thread: {e}")

    def on_submit(self, event=None):
        raw_input = self.input_textbox.get("1.0", tk.END)
        user_input = sanitize_text(raw_input, max_len=4000).strip()
        if user_input:
            self.text_box.insert(tk.END, f"{self.user_id}: {user_input}\n")
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)
            self.text_box.see(tk.END)

            self.executor.submit(self.generate_response, user_input)
            self.executor.submit(self.generate_images, user_input)
            self.after(100, self.process_queue)
        return "break"


    def process_queue(self):
        try:
            while True:
                msg = self.response_queue.get_nowait()
                if msg['type'] == 'text':
                    self.text_box.insert(tk.END, f"AI: {msg['data']}\n")
                elif msg['type'] == 'image':
                    self.image_label.configure(image=msg['data'])
                    self.image_label.image = msg['data']
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)


    def create_object(self, class_name, object_data):

        object_data = {
            k: self._encrypt_field(v) if k in {"user_message", "ai_response"} else v
            for k, v in object_data.items()
        }

        unique_string = f"{object_data.get('time', '')}-{object_data.get('user_message', '')}-{object_data.get('ai_response', '')}"
        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex

        try:
            self.client.data_object.create(object_data, object_uuid, class_name)
            logger.info(f"Object created with UUID: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating object in Weaviate: {e}")

        return object_uuid


    def extract_keywords(self, message):
        try:
            blob = TextBlob(message)
            nouns = blob.noun_phrases
            return list(nouns)
        except Exception as e:
            print(f"Error in extract_keywords: {e}")
            return []

    def generate_images(self, message):
        try:
            url = config['IMAGE_GENERATION_URL']
            payload = self.prepare_image_generation_payload(message)
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                self.process_image_response(response)
            else:
                logger.error(f"Error generating image: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Error in generate_images: {e}")

    def prepare_image_generation_payload(self, message):
        safe_prompt = sanitize_text(message, max_len=1000)
        return {
            "prompt": safe_prompt,
            "steps": 51,
            "seed": random.randrange(sys.maxsize),
            "enable_hr": "false",
            "denoising_strength": "0.7",
            "cfg_scale": "7",
            "width": 526,
            "height": 756,
            "restore_faces": "true",
        }

    def process_image_response(self, response):
        try:
            image_data = response.json()['images']
            self.loaded_images = []  # ✅ Store strong references to images

            for img_data in image_data:
                img_tk = self.convert_base64_to_tk(img_data)
                if img_tk:
                    self.response_queue.put({'type': 'image', 'data': img_tk})
                    self.loaded_images.append(img_tk)  # ✅ Prevent garbage collection
                    self.save_generated_image(img_data)
                else:
                    logger.warning("Failed to convert base64 image to tkinter image.")
        except ValueError as e:
            logger.error(f"Error processing image data: {e}")

    def convert_base64_to_tk(self, base64_data):
        if ',' in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        image_data = base64.b64decode(base64_data)
        try:
            photo = tk.PhotoImage(data=base64_data)
            return photo
        except tk.TclError as e:
            logger.error(f"Error converting base64 to PhotoImage: {e}")
            return None

    def save_generated_image(self, base64_data):
        try:
            if ',' in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(base64_data)
            file_name = f"generated_image_{uuid.uuid4()}.png"
            image_path = os.path.join("saved_images", file_name)
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            print(f"Image saved to {image_path}")
        except Exception as e:
            logger.error(f"Error saving generated image: {e}")

    def update_username(self):
        """Update the username based on the input field."""
        new_username = self.username_entry.get()
        if new_username:
            self.user_id = new_username
            print(f"Username updated to: {self.user_id}")
        else:
            print("Please enter a valid username.")

    def setup_gui(self):
        customtkinter.set_appearance_mode("Dark") 
        self.title("OneLoveIPFS AI")
        window_width = 1920
        window_height = 1080
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        

        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        
        try:
            logo_photo = tk.PhotoImage(file=logo_path) 
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_photo)
            self.logo_label.image = logo_photo
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            logger.error(f"Error loading logo image: {e}")
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Logo")
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))


        self.image_label = customtkinter.CTkLabel(self.sidebar_frame)
        self.image_label.grid(row=1, column=0, padx=20, pady=10)
        try:
            placeholder_photo = tk.PhotoImage(width=140, height=140)

            placeholder_photo.put(("gray",), to=(0, 0, 140, 140))
            self.image_label.configure(image=placeholder_photo)
            self.image_label.image = placeholder_photo
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")


        self.text_box = customtkinter.CTkTextbox(self, bg_color="black", text_color="white", border_width=0, height=360, width=50, font=customtkinter.CTkFont(size=23))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")


        self.input_textbox_frame = customtkinter.CTkFrame(self)
        self.input_textbox_frame.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame.grid_columnconfigure(0, weight=1)
        self.input_textbox_frame.grid_rowconfigure(0, weight=1)
        self.input_textbox = tk.Text(self.input_textbox_frame, font=("Roboto Medium", 12),
                                     bg=customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
                                     fg=customtkinter.ThemeManager.theme["CTkLabel"]["text_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0], relief="flat", height=1)
        self.input_textbox.grid(padx=20, pady=20, sticky="nsew")
        self.input_textbox_scrollbar = customtkinter.CTkScrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.input_textbox.configure(yscrollcommand=self.input_textbox_scrollbar.set)


        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox.bind('<Return>', self.on_submit)


        self.settings_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.username_label = customtkinter.CTkLabel(self.settings_frame, text="Username:")
        self.username_label.grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = customtkinter.CTkEntry(self.settings_frame, width=120, placeholder_text="Enter username")
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        self.username_entry.insert(0, "gray00")
        self.update_username_button = customtkinter.CTkButton(self.settings_frame, text="Update", command=self.update_username)
        self.update_username_button.grid(row=0, column=2, padx=5, pady=5)


if __name__ == "__main__":
    try:
        user_id = "gray00"
        app_gui = App(user_id)
        init_db()
        app_gui.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
