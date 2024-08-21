from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

import torch
from datasets import load_dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Initialize the processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Define the maximum sequence length for your model
max_seq_length = 600

# Example input text
chat_answer = "Your long text input goes here..."

# Tokenize the input text
inputs = processor(text=chat_answer, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length, forward_params={"speaker_embeddings": speaker_embedding})

# Generate speech
speech = model.generate(**inputs)

# Save or play the generated speech
# You can use libraries like simpleaudio or playsound to play the speech
