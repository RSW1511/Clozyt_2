# app.py
"""
Closet — Accurate & Efficient Swipe Recommender (patcheD)
- One-time heavy work cached; swipes are instant.
- Default embeddings: TF-IDF + TruncatedSVD (fast & reliable).
- Optional SBERT (attempted only on request; robust fallback to TF-IDF).
- Image caching to avoid repeated downloads.
- Prebuffer of next items for buttery UI.
- Use "Load/Rebuild" in sidebar to refresh catalog/embeddings/index.
"""

import os
import glob
import random
from io import BytesIO
from typing import List, Optional

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Optional heavy libs flags
HAS_SBERT = False
HAS_FAISS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    # SBERT not installed or import failed; we'll handle it when user requests SBERT
    HAS_SBERT = False

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# -------------------------
# Utility functions: CSV discovery + canonicalization
# -------------------------
def list_csvs(data_dir: str) -> List[str]:
    """Return list of CSV files in directory."""
    patterns = ["*_products.csv", "*.csv"]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(os.path.join(data_dir, p))))
    return [f for f in files if os.path.isfile(f) and "catalog_unified" not in f]

def canonicalize_csvs(csvs: List[str], max_items_per_file: int = 300) -> pd.DataFrame:
    """
    Read up to max_items_per_file from each CSV and extract canonical fields.
    This is a simple canonicalizer — it tolerates different column names.
    """
    rows = []
    for fp in csvs:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            # skip unreadable files
            continue
        # normalize column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # sample for speed if large
        if len(df) > max_items_per_file:
            df = df.sample(n=max_items_per_file, random_state=42)
        for _, r in df.iterrows():
            title = r.get("title") or r.get("name") or r.get("product_name") or ""
            brand = r.get("brand") if pd.notna(r.get("brand")) else os.path.basename(fp)
            image_url = r.get("image_url") or r.get("image") or r.get("image_link") or ""
            description = r.get("description") or r.get("desc") or ""
            price_raw = r.get("price") or r.get("current_price") or ""
            price_num = None
            try:
                if price_raw and not pd.isna(price_raw):
                    s = str(price_raw).replace("$", "").replace(",", "")
                    s = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
                    price_num = float(s) if s != "" else None
            except Exception:
                price_num = None
            pid = str(abs(hash((os.path.basename(fp), str(r.get('product_url','')), title))))[:16]
            rows.append({
                "product_id": pid,
                "title": str(title),
                "brand": str(brand),
                "description": str(description),
                "image_url": str(image_url),
                "price_num": price_num,
                "source": os.path.basename(fp)
            })
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        return df_all
    df_all = df_all.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    return df_all

# -------------------------
# Embedding utilities (TF-IDF+SVD default)
# -------------------------
@st.cache_resource
def get_tfidf_svd_components(tfidf_dim: int = 128):
    """Return TF-IDF vectorizer and SVD transformer objects cached in memory."""
    vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    svd = TruncatedSVD(n_components=tfidf_dim, n_iter=7, random_state=42)
    return vec, svd

def build_text_embeddings_tfidf(texts: List[str], tfidf_dim: int = 128) -> np.ndarray:
    """Build TF-IDF + truncated SVD embeddings and L2-normalize them."""
    vec, svd = get_tfidf_svd_components(tfidf_dim)
    X = vec.fit_transform(texts)
    Xred = svd.fit_transform(X)
    Xred = Xred / (np.linalg.norm(Xred, axis=1, keepdims=True) + 1e-9)
    return Xred.astype("float32")

def try_build_sbert_embeddings(texts: List[str], model_name: str = "all-mpnet-base-v2") -> Optional[np.ndarray]:
    """
    Attempt to build SBERT embeddings. This may fail (missing package, download failure, OOM).
    We catch exceptions and return None if anything goes wrong.
    """
    try:
        # local import attempt (so missing package doesn't crash import-time)
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return emb.astype("float32")
    except Exception as e:
        # Log to Streamlit and fall back outside this function
        st.warning(f"SBERT embedding failed ({type(e).__name__}): {str(e)[:300]}")
        return None

# -------------------------
# Vector index helpers (FAISS optional)
# -------------------------
def build_index(embeddings: np.ndarray, use_faiss: bool = False):
    """
    Build index object. If FAISS is available and requested, build HNSW index.
    Otherwise return pre-normalized brute-force matrix for dot-product search.
    """
    if use_faiss and HAS_FAISS:
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return ("faiss", index)
    # brute-force pre-normalized matrix
    embn = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    return ("brute", embn)

def query_index(index_obj, qvec: np.ndarray, k: int = 100) -> List[int]:
    """Query index: FAISS search or brute-force dot-product search."""
    typ = index_obj[0]
    if typ == "faiss":
        idx = index_obj[1]
        q = qvec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        _, ids = idx.search(q, k)
        return ids[0].tolist()
    embn = index_obj[1]
    qn = qvec / (np.linalg.norm(qvec) + 1e-9)
    sims = embn.dot(qn)
    return np.argsort(-sims)[:k].tolist()

# -------------------------
# Image fetcher (cached)
# -------------------------
@st.cache_data(ttl=3600)
def cached_image_fetch(url: str) -> Optional[Image.Image]:
    """Fetch image from local path or HTTP and cache result for some time."""
    if not url or not isinstance(url, str) or url.strip() == "":
        return None
    # local path support
    if os.path.exists(url):
        try:
            return Image.open(url).convert("RGB")
        except Exception:
            return None
    # http(s)
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None

# -------------------------
# Session manager (short-term + long-term)
# -------------------------
class SessionManager:
    """
    Keeps per-user short-term and long-term vectors plus brand counts and recent shown list.
    on_swipe updates u_s quickly; u_l updated slowly every N steps.
    """
    def __init__(self, dim: int, eta_like: float = 0.25, eta_dislike: float = 0.12, recent_k: int = 120):
        self.dim = dim
        self.users = {}
        self.eta_like = eta_like
        self.eta_dislike = eta_dislike
        self.recent_k = recent_k

    def init_user(self, uid: str):
        self.users[uid] = {
            "u_s": np.zeros(self.dim, dtype="float32"),
            "u_l": np.zeros(self.dim, dtype="float32"),
            "brands": {},
            "recent": [],
            "step": 0
        }

    def on_swipe(self, uid: str, item_vec: np.ndarray, action: str, brand: str, item_idx: int):
        if uid not in self.users:
            self.init_user(uid)
        u = self.users[uid]
        if action == "yes":
            u["u_s"] = (1 - self.eta_like) * u["u_s"] + self.eta_like * item_vec
            u["brands"][brand] = u["brands"].get(brand, 0) + 1
        else:
            # soft negative update (do not permanently erase preferences)
            u["u_s"] = (1 - self.eta_dislike) * u["u_s"] - self.eta_dislike * item_vec
        # normalize
        u["u_s"] = u["u_s"] / (np.linalg.norm(u["u_s"]) + 1e-9)
        # recent items
        u["recent"].append(item_idx)
        if len(u["recent"]) > self.recent_k:
            u["recent"] = u["recent"][-self.recent_k :]
        u["step"] += 1
        # slow long-term update
        if u["step"] % 20 == 0:
            u["u_l"] = 0.98 * u["u_l"] + 0.02 * u["u_s"]
            u["u_l"] = u["u_l"] / (np.linalg.norm(u["u_l"]) + 1e-9)

    def get_query_vector(self, uid: str) -> np.ndarray:
        if uid not in self.users:
            self.init_user(uid)
        u = self.users[uid]
        alpha = 1.0 - 1.0 / (1 + np.exp(-0.4 * (u["step"] - 3)))  # ramping alpha
        q = alpha * u["u_s"] + (1 - alpha) * u["u_l"]
        return q / (np.linalg.norm(q) + 1e-9)

    def is_recent(self, uid: str, idx: int) -> bool:
        return idx in self.users[uid]["recent"]

# -------------------------
# Recommendation function (candidate generation + prebuffer)
# -------------------------
def generate_recommendations(df: pd.DataFrame, emb: np.ndarray, index_obj, session_mgr: SessionManager, uid: str, prebuffer: int = 5):
    """
    Query index with current user vector, filter recent items, return a small prebuffer list of (idx, reason).
    Reason is a short string explaining the score (similarity and brand hints).
    """
    q = session_mgr.get_query_vector(uid)
    cand_idxs = query_index(index_obj, q, k=300)
    recs = []
    for idx in cand_idxs:
        if session_mgr.is_recent(uid, idx):
            continue
        sim = float(np.dot(q, emb[idx]))
        # simple reason; you can extend with brand boosts, price, filters etc.
        reason = f"sim={sim:.3f}"
        row = df.iloc[idx]
        if session_mgr.users[uid]["brands"].get(row.get("brand", ""), 0) > 0:
            reason += f"; brand_like={session_mgr.users[uid]['brands'][row.get('brand','')]}"
        recs.append((idx, reason))
        if len(recs) >= prebuffer:
            break
    return recs

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Closet — Fast & Accurate Swipe Demo", layout="centered")
st.title("Closet — Fast & Accurate Swipe Demo")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    DATA_DIR = st.text_input("CSV folder path", r"C:\Users\rohan\Clozyt_2\data")
    max_items_per_file = st.number_input("Max items per CSV (sample)", min_value=50, max_value=2000, value=300, step=50)
    tfidf_dim = st.slider("TF-IDF SVD dim (speed vs quality)", 64, 384, 128, 32)
    use_sbert = st.checkbox("Use SBERT embeddings (may be slow / use RAM)", value=False)
    # recommend smaller SBERT model on low RAM machines
    if use_sbert:
        st.caption("If you have limited RAM, prefer 'paraphrase-MiniLM-L6-v2' instead of 'all-mpnet-base-v2'.")
    use_faiss = st.checkbox("Use FAISS (if installed) for faster ANN", value=False)
    prebuffer_n = st.number_input("Prebuffer count (next items)", min_value=1, max_value=10, value=5, step=1)
    if st.button("Load / Rebuild (one-time)"):
        # indicate rebuild to code below
        st.session_state["rebuild"] = True
        st.rerun()

# Validate path
if not os.path.exists(DATA_DIR):
    st.error(f"CSV folder not found: {DATA_DIR}")
    st.stop()

# Discover CSVs (top 3 by simple completeness metric)
csvs = list_csvs(DATA_DIR)
if not csvs:
    st.error("No CSV files found in the given folder.")
    st.stop()

# Build or load canonical catalog, embeddings, index into session_state
if "catalog_built" not in st.session_state or st.session_state.get("rebuild", False):
    st.info("Building catalog and embeddings — this runs once. Please wait...")
    # pick top 3 CSVs by quick completeness score (title/image_url/brand presence)
    scores = []
    for fp in csvs:
        try:
            tmp = pd.read_csv(fp, nrows=200, low_memory=False)
            tmp.columns = [c.strip().lower().replace(" ", "_") for c in tmp.columns]
            metrics = {
                "title": float(tmp['title'].notna().mean()) if 'title' in tmp.columns else 0.0,
                "image_url": float(tmp['image_url'].notna().mean()) if 'image_url' in tmp.columns else 0.0,
                "brand": float(tmp['brand'].notna().mean()) if 'brand' in tmp.columns else 0.0
            }
            scores.append((fp, sum(metrics.values()) / len(metrics), metrics))
        except Exception:
            scores.append((fp, 0.0, {}))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_n = [s[0] for s in scores[:3]] if len(scores) >= 3 else [s[0] for s in scores]
    if not top_n:
        st.error("No suitable CSVs found for canonicalization.")
        st.stop()
    # canonicalize (sample per file for speed)
    catalog_df = canonicalize_csvs(top_n, max_items_per_file=max_items_per_file)
    if catalog_df.empty:
        st.error("No rows produced by canonicalization. Check CSV formats.")
        st.stop()
    # Optionally filter to rows that have image_url (we want images to display)
    img_only = catalog_df[catalog_df["image_url"].notna() & (catalog_df["image_url"].str.strip() != "")].reset_index(drop=True)
    if not img_only.empty:
        catalog_df = img_only  # prefer image-only rows for UI reliability

    # Build text input list for embeddings
    texts = (catalog_df["title"].fillna("") + " | " + catalog_df["brand"].fillna("") + " | " + catalog_df["description"].fillna("")).tolist()

    # Embedding selection: try SBERT only if user requested, otherwise TF-IDF fallback
    embeddings = None
    if use_sbert:
        st.info("Attempting SBERT embeddings. This can be slow and memory heavy.")
        # allow user-chosen small model via an environment variable or fallback names — default to all-mpnet
        # If you have limited RAM, consider "paraphrase-MiniLM-L6-v2" (smaller & faster).
        sbert_choice = os.getenv("CLOSET_SBERT_MODEL", "all-mpnet-base-v2")
        embeddings = try_build_sbert_embeddings(texts, model_name=sbert_choice)
        if embeddings is None:
            st.warning("SBERT attempt failed — falling back to TF-IDF+SVD.")
    if embeddings is None:
        with st.spinner("Building TF-IDF + SVD embeddings (fast and reliable)..."):
            embeddings = build_text_embeddings_tfidf(texts, tfidf_dim)
    # Build index (FAISS optional)
    index_obj = build_index(embeddings, use_faiss and HAS_FAISS)

    # Cache into session_state for instant interactions
    st.session_state["catalog_df"] = catalog_df
    st.session_state["embeddings"] = embeddings
    st.session_state["index_obj"] = index_obj
    st.session_state["session_mgr"] = SessionManager(dim=embeddings.shape[1])
    st.session_state["catalog_built"] = True
    st.session_state["rebuild"] = False
    st.success(f"Catalog built with {len(catalog_df)} items. Ready to swipe!")

# load from session_state for fast access
catalog_df = st.session_state["catalog_df"]
embeddings = st.session_state["embeddings"]
index_obj = st.session_state["index_obj"]
session_mgr = st.session_state["session_mgr"]

# demo user id (simple single-user demo). For multi-user, key by login id.
user_id = "demo_user"
if user_id not in session_mgr.users:
    session_mgr.init_user(user_id)

# Generate prebuffered recommendations (small number)
prebuffer = int(st.sidebar.number_input("Prebuffer (next items to prepare)", min_value=1, max_value=10, value=5))
recs = generate_recommendations(catalog_df, embeddings, index_obj, session_mgr, user_id, prebuffer=prebuffer)
if not recs:
    st.warning("No recommendations available after filtering. Try rebuilding with larger sample or different CSVs.")
    st.stop()

# Show current top recommendation (first in prebuffer)
current_idx, current_reason = recs[0]
current_row = catalog_df.iloc[current_idx]
st.subheader(current_row["title"])
st.write(f"Brand: **{current_row.get('brand','')}**  |  Price: `{current_row.get('price_num','')}`")
# fetch image from cache (local path or http)
img = cached_image_fetch(current_row.get("image_url", ""))
if img is not None:
    st.image(img, use_column_width=True)
else:
    st.write("_Image unavailable for this item_")
st.write("Why:", current_reason)

# Buttons: instant updates (only update session vectors and rerun)
col1, col2, col3 = st.columns(3)
if col1.button("👍 Yes"):
    session_mgr.on_swipe(user_id, embeddings[current_idx], "yes", current_row.get("brand", ""), current_idx)
    # instant rerun to show next item (embeddings/index remain cached)
    st.rerun()
if col2.button("👎 No"):
    session_mgr.on_swipe(user_id, embeddings[current_idx], "no", current_row.get("brand", ""), current_idx)
    st.rerun()
if col3.button("⏭ Skip"):
    # small negative push for skip to avoid being stuck
    session_mgr.on_swipe(user_id, embeddings[current_idx] * 0.05, "no", current_row.get("brand", ""), current_idx)
    st.rerun()

# Show prebuffered next items (UI-only preview)
st.markdown("---")
st.caption("Next in queue (prebuffer):")
for i, (ix, reason) in enumerate(recs[1:], start=1):
    r = catalog_df.iloc[ix]
    st.write(f"{i}. {r.get('title','')} — {r.get('brand','')} ({reason})")

# Session diagnostics in sidebar
st.sidebar.markdown("---")
st.sidebar.write("Session diagnostics")
u = session_mgr.users[user_id]
st.sidebar.write(f"Steps: {u['step']}")
st.sidebar.write(f"Brands liked (counts): {u['brands']}")
st.sidebar.write(f"Recent seen (len): {len(u['recent'])}")

# End of file
