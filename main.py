import tkinter as tk
import customtkinter
import threading
import os
import sqlite3 
import weaviate
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
import logging
import pennylane as qml
import psutil
import webcolors
from nltk.tokenize import word_tokenize


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

DB_NAME = config['DB_NAME']
API_KEY = config['API_KEY']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_QUERY_PATH = config['WEAVIATE_QUERY_PATH']
app = FastAPI()


def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)


api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

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
    logger.info(f"save_user_message called with user_id: {user_id}, user_input: {user_input}")

    if not user_input:
        logger.error("User input is None or empty.")
        return

    try:
        response_time = get_current_multiversal_time()
        data_object = {
            "user_id": user_id,
            "response": user_input,
            "response_time": response_time
        }
        generated_uuid = generate_uuid5(user_id, user_input)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                       (user_id, user_input, response_time))
        conn.commit()
        conn.close()

        response = requests.post('http://127.0.0.1:8079/v1/objects', json={
            "class": "InteractionHistory",
            "id": generated_uuid,
            "properties": data_object
        })

        if response.status_code != 200:
            logger.error(f"Failed to post to Weaviate: {response.status_code} {response.text}")

    except Exception as e:
        logger.error(f"Error saving user message: {e}")

def save_bot_response(bot_id, bot_response):
    logger.info(f"save_bot_response called with bot_id: {bot_id}, bot_response: {bot_response}")

    if not bot_response:
        logger.error("Bot response is None or empty.")
        return

    try:
        response_time = get_current_multiversal_time()
        data_object = {
            "user_id": bot_id,
            "response": bot_response,
            "response_time": response_time
        }
        generated_uuid = generate_uuid5(bot_id, bot_response)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                       (bot_id, bot_response, response_time))
        conn.commit()
        conn.close()

        response = requests.post('http://127.0.0.1:8079/v1/objects', json={
            "class": "InteractionHistory",
            "id": generated_uuid,
            "properties": data_object
        })

        if response.status_code != 200:
            logger.error(f"Failed to post to Weaviate: {response.status_code} {response.text}")

    except Exception as e:
        logger.error(f"Error saving bot response: {e}")


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


class UserInput(BaseModel):
    message: str

@app.post("/process/")
def process_input(user_input: UserInput, api_key: str = Depends(get_api_key)):
    try:
        response = llama_generate(user_input.message, client)
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

   summarized_chunk = summarizer.summarize(chunk)
   query_chunk = summarized_chunk if summarized_chunk else chunk

   if not query_chunk:
       logger.error("Query chunk is empty.")
       return ""

   query = {
       "query": {
           "nearText": {
               "concepts": [user_input],
               "certainty": 0.7
           }
       }
   }

   try:
       response = client.query.raw(json.dumps(query))
       logger.debug(f"Query sent: {json.dumps(query)}")
       logger.debug(f"Response received: {response}")

       if response and 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
           interaction = response['data']['Get']['InteractionHistory'][0]
           return f"{interaction['user_message']} {interaction['ai_response']}"
       else:
           logger.error("Weaviate client returned no relevant data for query: " + json.dumps(query))
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


def truncate_text(text, max_length=55):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        return text if len(text) <= max_length else text[:max_length] + '...'

    except Exception as e:
        print(f"Error in truncate_text: {e}")
        return ""


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


class App(customtkinter.CTk):
    def __init__(self, user_identifier):
        super().__init__()
        self.user_id = user_identifier
        self.bot_id = "bot"
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        self.executor = ThreadPoolExecutor(max_workers=4)

    def retrieve_past_interactions(self, user_input, result_queue):
        try:
            keywords = extract_verbs_and_nouns(user_input)
            concepts_query = ' '.join(keywords)

            user_message, ai_response = self.fetch_relevant_info_internal(concepts_query)

            if user_message and ai_response:
                summarized_interaction = summarizer.summarize(f"{user_message} {ai_response}")
                sentiment = TextBlob(summarized_interaction).sentiment.polarity
                processed_interaction = {
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "summarized_interaction": summarized_interaction,
                    "sentiment": sentiment
                }
                result_queue.put([processed_interaction])
            else:
                logger.error("No relevant interactions found for the given user input.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])

    def fetch_relevant_info_internal(self, chunk):
        if self.client:
            query = f"""
            {{
                Get {{
                    InteractionHistory(nearText: {{
                        concepts: ["{chunk}"],
                        certainty: 0.7
                    }}) {{
                        user_message
                        ai_response
                        .with_limit(1)
                    }}
                }}
            }}
            """
            try:
                response = self.client.query.raw(query)
                if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                    interaction = response['data']['Get']['InteractionHistory'][0]
                    return interaction['user_message'], interaction['ai_response']
                else:
                    return "", ""
            except Exception as e:
                logger.error(f"Weaviate query failed: {e}")
                return "", ""
        return "", ""

    def process_response_and_store_in_weaviate(self, user_message, ai_response):
        response_blob = TextBlob(ai_response)
        keywords = response_blob.noun_phrases
        sentiment = response_blob.sentiment.polarity
        enhanced_keywords = set()
        for phrase in keywords:
            enhanced_keywords.update(phrase.split())

        interaction_object = {
            "userMessage": user_message,
            "aiResponse": ai_response,
            "keywords": list(enhanced_keywords),
            "sentiment": sentiment
        }

        interaction_uuid = str(uuid.uuid4())

        try:
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )

            print(f"Interaction stored in Weaviate with UUID: {interaction_uuid}")

        except Exception as e:            
            print(f"Error storing interaction in Weaviate: {e}")

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)

    def create_interaction_history_object(self, user_message, ai_response):
        interaction_object = {
            "user_message": user_message,
            "ai_response": ai_response
        }

        try:
            object_uuid = uuid.uuid4()
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            print(f"Interaction history object created with UUID: {object_uuid}")
        except Exception as e:
            print(f"Error creating interaction history object in Weaviate: {e}")

    def map_keywords_to_weaviate_classes(self, keywords, context):
        try:
            summarized_context = summarizer.summarize(context)
        except Exception as e:
            print(f"Error in summarizing context: {e}")
            summarized_context = context

        try:
            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
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
                print(f"Error in mapping keyword '{keyword}': {e}")

        return mapped_classes

    def generate_response(self, user_input):
        try:
            if not user_input:
                logger.error("User input is None or empty.")
                return

            user_id = self.user_id
            bot_id = self.bot_id

            save_user_message(user_id, user_input)

            include_past_context = "[pastcontext]" in user_input
            user_input = user_input.replace("[pastcontext]", "").replace("[/pastcontext]", "")
            past_context = ""

            if include_past_context:
                result_queue = queue.Queue()
                self.retrieve_past_interactions(user_input, result_queue)
                past_interactions = result_queue.get()
                if past_interactions:
                    past_context_combined = "\n".join(
                        [f"User: {interaction['user_message']}\nAI: {interaction['ai_response']}" 
                         for interaction in past_interactions])
                    past_context = past_context_combined[-1500:]
            rgb = extract_rgb_from_text(user_input)
            quantum_state = generate_quantum_state(rgb=rgb)
            complete_prompt = f"{quantum_state}\n{past_context}\nUser: {user_input}"
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

    def fetch_interactions(self):
        try:
            query = {
                "query": """
                {
                    Get {
                        InteractionHistory(sort: [{path: "response_time", order: desc}], limit: 15) {
                            user_message
                            ai_response
                            response_time
                        }
                    }
                }
                """
            }
            response = self.client.query.raw(json.dumps(query))
            if 'data' in response and 'Get' in response['data'] and 'InteractionHistory' in response['data']['Get']:
                interactions = response['data']['Get']['InteractionHistory']
                return [{'user_message': interaction['user_message'], 'ai_response': interaction['ai_response'], 'response_time': interaction['response_time']} for interaction in interactions]
            else:
                return []
        except Exception as e:
            logger.error(f"Error fetching interactions from Weaviate: {e}")
            return []

    def on_submit(self, event=None):
        user_input = self.input_textbox.get("1.0", tk.END).strip()
        if user_input:
            self.text_box.insert(tk.END, f"{self.user_id}: {user_input}\n")
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)
            self.text_box.see(tk.END)
            self.executor.submit(self.generate_response, user_input)
            self.executor.submit(self.generate_images, user_input)
            self.after(100, self.process_queue)

        return "break"

    def create_object(self, class_name, object_data):

        unique_string = f"{object_data['time']}-{object_data['user_message']}-{object_data['ai_response']}"

        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex

        try:
            self.client.data_object.create(object_data, object_uuid, class_name)
            print(f"Object created with UUID: {object_uuid}")
        except Exception as e:
            print(f"Error creating object in Weaviate: {e}")

        return object_uuid

    def process_queue(self):
        try:
            while True:
                response = self.response_queue.get_nowait()
                if response['type'] == 'text':
                    self.text_box.insert(tk.END, f"AI: {response['data']}\n")
                elif response['type'] == 'image':
                    self.image_label.configure(image=response['data'])
                    self.image_label.image = response['data']
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)

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
        return {
            "prompt": message,
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
            for img_data in image_data:
                img_tk = self.convert_base64_to_tk(img_data)
                self.response_queue.put({'type': 'image', 'data': img_tk})
                self.save_generated_image(img_data)
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
