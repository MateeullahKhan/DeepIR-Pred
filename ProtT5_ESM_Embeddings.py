import torch
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, EsmModel
from Bio import SeqIO  # Essential for parsing .fasta files

# --- Configuration ---
# SELECTED_MODEL = "ProtT5"
SELECTED_MODEL = "ESM2"  
INPUT_FASTA = "IR_test_P137_N137.fasta"  # Your specific file
OUTPUT_CSV = f"{SELECTED_MODEL}_embeddings_274.csv"

# --- Device Setup (Crucial for M1) ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps") 
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"🚀 Using device: {device}")

# --- Model Configs for M1 Pro ---
MODEL_CONFIGS = {
    "ProtT5": {
        "name": "Rostlab/prot_t5_xl_uniref50",
        "type": "t5",
        "window_size": 512,
        "overlap": 128,
        "embed_dim": 1024
    },
    "ESM2": {
        "name": "facebook/esm2_t33_650M_UR50D",
        "type": "esm",
        "window_size": 1022,
        "overlap": 256,
        "embed_dim": 1280
    }
}

# --- Helper Functions ---
def load_model(model_key):
    config = MODEL_CONFIGS[model_key]
    print(f"Loading {config['name']}...")
    if config['type'] == 't5':
        tokenizer = T5Tokenizer.from_pretrained(config['name'], do_lower_case=False)
        model = T5EncoderModel.from_pretrained(config['name'])
    elif config['type'] == 'esm':
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        model = EsmModel.from_pretrained(config['name'])
    
    model = model.to(device).eval()
    return tokenizer, model, config

def get_embeddings(sequence, tokenizer, model, config):
    seq_len = len(sequence)
    window_size = config['window_size']
    overlap = config['overlap']
    model_type = config['type']
    
    # Strategy: Sliding Window
    final_embeddings = np.zeros((seq_len, config['embed_dim']), dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.int32)
    
    # Using window_size - overlap to ensure continuity
    steps = range(0, seq_len, max(1, window_size - overlap))
    
    for start in steps:
        end = min(start + window_size, seq_len)
        chunk = sequence[start:end]
        
        # Formatting for the specific model
        proc_chunk = " ".join(list(chunk)) if model_type == 't5' else chunk
        
        inputs = tokenizer(proc_chunk, return_tensors="pt", padding=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state[0]

            if model_type == 'esm':
                chunk_emb = last_hidden[1 : (end-start)+1] # Remove CLS/EOS
            else:
                chunk_emb = last_hidden[: (end-start)]

        final_embeddings[start:end] += chunk_emb.cpu().numpy()
        counts[start:end] += 1
        if end == seq_len: break

    return final_embeddings / np.maximum(counts[:, None], 1)

# --- Main Processing Loop ---
def process_fasta_file():
    tokenizer, model, config = load_model(SELECTED_MODEL)
    results = []

    # Parse FASTA one-by-one to save memory
    print(f"Reading sequences from {INPUT_FASTA}...")
    records = list(SeqIO.parse(INPUT_FASTA, "fasta"))

    for record in tqdm(records, desc=f"Embedding on {device.type.upper()}"):
        pid = record.id
        seq = str(record.seq).upper()
        
        # Cleanup
        seq = re.sub(r"[^A-Z]", "", seq)
        seq = re.sub(r"[UZOB]", "X", seq)
        
        if len(seq) == 0: continue

        # 1. Get Per-Residue Embeddings
        residue_embs = get_embeddings(seq, tokenizer, model, config)
        
        # 2. Get Global Protein Embedding (Mean Pool)
        protein_emb = residue_embs.mean(0)
        
        results.append([pid, len(seq)] + protein_emb.tolist())

    # Save to CSV
    print(f"Saving to {OUTPUT_CSV}...")
    cols = ["ID", "Length"] + [f"dim_{i}" for i in range(config['embed_dim'])]
    df = pd.DataFrame(results, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)
    print("✨ Processing Complete!")

if __name__ == "__main__":
    process_fasta_file()