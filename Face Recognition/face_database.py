import numpy as np
import pickle
import os

DB_PATH = 'face_db.pkl'

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_database(db):
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)

def add_face(name, embedding):
    db = load_database()
    if name not in db:
        db[name] = {'embeddings': [], 'mean_embedding': None}
    db[name]['embeddings'].append(embedding)
    db[name]['mean_embedding'] = np.mean(db[name]['embeddings'], axis=0)
    save_database(db)

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_face(embedding, threshold=0.5, k=5):
    db = load_database()
    if not db:
        return None, 0.0, []
    scores = {}
    for name, entry in db.items():
        if isinstance(entry, dict) and 'mean_embedding' in entry:
            mean_emb = entry['mean_embedding']
        else:
            continue
        score = cosine_similarity(embedding, mean_emb)
        scores[name] = score
    if not scores:
        return None, 0.0, []
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_scores[:k]
    best_name, best_score = top_k[0]
    if best_score >= threshold:
        return best_name, best_score, top_k
    return None, best_score, top_k
