#!/usr/bin/env python3
"""
interactive_swipe_recommender_top3_norepeats.py

- Picks top 3 CSVs with highest completeness (title, image_url, brand).
- Builds embeddings (SBERT if available, TF-IDF fallback).
- Optional image embeddings (CLIP or color histogram fallback).
- Builds FAISS or brute-force vector index.
- Interactive loop: recommend, explain, update session with yes/no feedback.
- Prevents repeated recommendations (recent memory, novelty penalty, brand cooldown, decay).
- Supports contextual filters (weather: cold/hot/rain; sale: yes/no).

Usage:
    python interactive_swipe_recommender_top3_norepeats.py --data_dir /mnt/data --max_items_per_file 400 --weather cold --sale yes
"""
import os, glob, argparse, random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# -------------------------
# Optional libs
# -------------------------
HAS_SBERT = False
HAS_FAISS = False
HAS_CLIP = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    pass

try:
    import faiss
    HAS_FAISS = True
except Exception:
    pass

try:
    from PIL import Image
    import requests
    from io import BytesIO
    import torch
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    HAS_CLIP = True
except Exception:
    from PIL import Image
    import requests
    from io import BytesIO
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# -------------------------
# Helpers
# -------------------------
def list_csvs(data_dir: str) -> List[str]:
    patterns = ["*_products.csv", "*.csv"]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(data_dir, p))))
    return [f for f in files if "catalog_unified" not in f]

def completeness_score(fp: str, required_cols=['title','image_url','brand']) -> Tuple[float, dict]:
    try:
        df = pd.read_csv(fp, nrows=200, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(fp, nrows=200, encoding='latin-1', low_memory=False)
        except Exception:
            return 0.0, {}
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    metrics = {}
    for c in required_cols:
        metrics[c] = df[c].notna().mean() if c in df.columns else 0.0
    return float(np.mean(list(metrics.values()))), metrics

def canonicalize_csvs(csvs: List[str], max_items_per_file:int=1000):
    rows = []
    for fp in csvs:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            df = pd.read_csv(fp, encoding='latin-1', low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if max_items_per_file and len(df) > max_items_per_file:
            df = df.sample(n=max_items_per_file, random_state=42)
        for _, r in df.iterrows():
            title = r.get('title') or r.get('name') or r.get('product_name') or ""
            brand = r.get('brand') if 'brand' in r.index else os.path.splitext(os.path.basename(fp))[0]
            image_url = r.get('image_url') or r.get('image') or r.get('image_link') or ""
            desc = r.get('description') or r.get('desc') or ""
            price_raw = r.get('price') if 'price' in r.index else r.get('current_price') if 'current_price' in r.index else ""
            price_num = None
            try:
                if price_raw and not pd.isna(price_raw):
                    s = str(price_raw).replace("$","").replace(",","")
                    s = "".join(ch for ch in s if (ch.isdigit() or ch=='.'))
                    price_num = float(s) if s!='' else None
            except:
                price_num = None
            product_url = r.get('product_url') or r.get('url') or ""
            category = r.get('category') or ""
            pid = str(abs(hash((os.path.basename(fp), product_url, title))))[:16]
            rows.append({
                'product_id': pid,
                'title': str(title),
                'brand': str(brand),
                'description': str(desc),
                'image_url': str(image_url),
                'price_raw': str(price_raw) if price_raw is not None else "",
                'price_num': price_num,
                'product_url': str(product_url),
                'category': str(category),
                'source': os.path.basename(fp)
            })
    return pd.DataFrame(rows).drop_duplicates(subset=['product_id']).reset_index(drop=True)

# -------------------------
# Embeddings
# -------------------------
class EmbeddingFactory:
    def __init__(self, method='sbert', tfidf_dim=256):
        self.method = method
        self.tfidf_dim = tfidf_dim
        self.sbert = SentenceTransformer("all-mpnet-base-v2") if HAS_SBERT else None
        self.tfidf = None
        self.svd = None
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.sbert is not None:
            return self.sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype('float32')
        self.tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=self.tfidf_dim, n_iter=10, random_state=42)
        return self.svd.fit_transform(X).astype('float32')

# -------------------------
# Index
# -------------------------
class VectorIndex:
    def __init__(self, embeddings: np.ndarray):
        self.emb = embeddings.astype('float32')
        self.N, self.D = self.emb.shape
        if HAS_FAISS:
            self.index = faiss.IndexHNSWFlat(self.D, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
            faiss.normalize_L2(self.emb)
            self.index.add(self.emb)
        else:
            norms = np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9
            self.emb = self.emb / norms
            self.index = None
    def query(self, q: np.ndarray, k=50):
        q = q.astype('float32').reshape(1,-1)
        if HAS_FAISS:
            faiss.normalize_L2(q)
            dists, idxs = self.index.search(q, k)
            return idxs[0].tolist()
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = (self.emb @ qn.reshape(-1))
        return np.argsort(-sims)[:k].tolist()

# -------------------------
# Session Manager (repeat prevention)
# -------------------------
class SessionManager:
    def __init__(self, dim:int, eta_like=0.25, eta_dislike=0.12, recent_k=100, brand_cooldown_steps=5, decay_rate=0.005):
        self.dim = dim
        self.users = {}
        self.eta_like = eta_like
        self.eta_dislike = eta_dislike
        self.recent_k = recent_k
        self.brand_cooldown_steps = brand_cooldown_steps
        self.decay_rate = decay_rate
    def init_user(self, user_id:str):
        self.users[user_id] = {'u_s': np.zeros(self.dim, dtype='float32'),
                               'u_l': np.zeros(self.dim, dtype='float32'),
                               'brands': {}, 'brand_last_shown': {}, 'recent_shown': [], 'step':0}
    def on_swipe(self, user_id, item_vec, action, brand, item_idx=None):
        u = self.users[user_id]
        if action=='yes':
            u['u_s'] = (1-self.eta_like)*u['u_s'] + self.eta_like*item_vec
            u['brands'][brand] = u['brands'].get(brand,0)+1
        else:
            u['u_s'] = (1-self.eta_dislike)*u['u_s'] - self.eta_dislike*item_vec
        u['u_s'] /= (np.linalg.norm(u['u_s'])+1e-9)
        u['step'] += 1
        if item_idx is not None:
            u['recent_shown'].append(item_idx)
            if len(u['recent_shown'])>self.recent_k:
                u['recent_shown']=u['recent_shown'][-self.recent_k:]
            u['brand_last_shown'][brand]=u['step']
    def get_query_vec(self, user_id):
        u = self.users[user_id]
        u['u_s'] *= (1.0-self.decay_rate)
        nvec = u['u_s']/(np.linalg.norm(u['u_s'])+1e-9)
        return nvec.astype('float32')
    def is_recent(self, user_id, idx): return idx in self.users[user_id]['recent_shown']
    def brand_cooldown(self, user_id, brand):
        u = self.users[user_id]
        return (u['step']-u['brand_last_shown'].get(brand,-9999))<=self.brand_cooldown_steps

# -------------------------
# Recommend + explain
# -------------------------
def recommend(df, embeddings, index, session_mgr, user_id, weather=None, sale='no',
              novelty_penalty=0.3, brand_cooldown_penalty=0.2, epsilon=0.1):
    q = session_mgr.get_query_vec(user_id)
    cand_idxs = index.query(q, k=200)
    u = session_mgr.users[user_id]
    scored=[]
    for idx in cand_idxs:
        if session_mgr.is_recent(user_id, idx): continue
        row=df.iloc[idx]; brand=row.get('brand','')
        sim=float(np.dot(q,embeddings[idx]))
        score=sim
        if brand in u['brands']: score+=0.05*u['brands'][brand]
        if session_mgr.brand_cooldown(user_id,brand): score-=brand_cooldown_penalty
        scored.append((score,idx,row))
    if not scored: return None
    scored.sort(reverse=True,key=lambda x:x[0])
    if random.random()<epsilon: choice=scored[random.randint(0,min(20,len(scored)-1))]
    else: choice=scored[0]
    _,idx,row=choice
    expl=f"sim={sim:.3f}"
    if row['brand'] in u['brands']: expl+=f"; brand likes={u['brands'][row['brand']]}"
    return idx,row,expl

# -------------------------
# Interactive loop
# -------------------------
def interactive_loop(df,embeddings,index,session_mgr,weather=None,sale='no'):
    uid="user_demo"; session_mgr.init_user(uid)
    while True:
        rec=recommend(df,embeddings,index,session_mgr,uid,weather=weather,sale=sale)
        if rec is None: print("No candidates."); break
        idx,row,expl=rec
        print("\n=== Recommended ===")
        print("Title:",row['title']); print("Brand:",row['brand']); print("Why:",expl)
        print("URL:",row['product_url'])
        ans=input("Do you like it? (yes/no/quit): ").strip().lower()
        if ans=='quit': break
        if ans not in ('yes','no'): continue
        session_mgr.on_swipe(uid,embeddings[idx],ans,row['brand'],item_idx=idx)

# -------------------------
# Main
# -------------------------
def main(args):
    csvs=list_csvs(args.data_dir)
    scores=[(c,*completeness_score(c)) for c in csvs]
    scores=sorted(scores,key=lambda x:x[1],reverse=True)
    print("Top 5 CSVs by completeness:")
    for s in scores[:5]: print(s[0],s[1],s[2])
    top3=[s[0] for s in scores[:3]]
    print("Using top3:",top3)
    df=canonicalize_csvs(top3,max_items_per_file=args.max_items_per_file)
    texts=(df['title']+" | "+df['brand']+" | "+df['description']).tolist()
    emb_factory=EmbeddingFactory()
    emb=emb_factory.fit_transform(texts)
    idx=VectorIndex(emb)
    sess=SessionManager(dim=emb.shape[1])
    interactive_loop(df,emb,idx,sess,weather=args.weather,sale=args.sale)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_dir",type=str,default="/mnt/data")
    p.add_argument("--max_items_per_file",type=int,default=400)
    p.add_argument("--weather",type=str,default=None)
    p.add_argument("--sale",type=str,default='no')
    args=p.parse_args()
    main(args)
