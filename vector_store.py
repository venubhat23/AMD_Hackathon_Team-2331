
import os
import json
import uuid
import difflib
import hashlib
import sqlite3
import threading
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
 
CHROMA_DIR       = "chroma_db"
SQLITE_PATH      = "chroma_db/sessions.db"
COLLECTION_NAME  = "retail_products"          # one doc per PRODUCT (not per image)
IMAGE_COLLECTION = "retail_images"            # one doc per IMAGE (for history/dedup)
 
_client           = None
_prod_collection  = None
_img_collection   = None
_embedder         = None
 
# ── Global write lock — prevents concurrent SQLite writes from parallel threads ──
_db_write_lock = threading.Lock()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────
 
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB — two collections
# ─────────────────────────────────────────────────────────────────────────────
 
def _get_client():
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client
 
def get_product_collection():
    """One vector per individual product — enables precise RAG."""
    global _prod_collection
    if _prod_collection is None:
        _prod_collection = _get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _prod_collection
 
def get_image_collection():
    """One vector per image scan — used for history page."""
    global _img_collection
    if _img_collection is None:
        _img_collection = _get_client().get_or_create_collection(
            name=IMAGE_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _img_collection
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SQLite — WAL mode + busy timeout to handle concurrent access gracefully
# ─────────────────────────────────────────────────────────────────────────────
 
def _get_sqlite():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, timeout=30)   # wait up to 30s before raising
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent readers + one writer without blocking reads
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")          # retry writes for up to 10s
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            created_at   TEXT,
            label        TEXT,
            image_paths  TEXT,
            analysis_ids TEXT
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role       TEXT,
            content    TEXT,
            timestamp  TEXT
        );
        CREATE TABLE IF NOT EXISTS inventory_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id          TEXT,
            product_name     TEXT,
            brand            TEXT,
            category         TEXT,
            quantity         INTEGER,
            inventory_status TEXT,
            price            TEXT,
            price_numeric    REAL,
            discount_pct     REAL,
            image_path       TEXT,
            image_hash       TEXT,
            timestamp        TEXT
        );
        CREATE TABLE IF NOT EXISTS image_hashes (
            image_hash  TEXT PRIMARY KEY,
            image_path  TEXT,
            scan_id     TEXT,
            analyzed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS price_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product_key  TEXT,
            product_name TEXT,
            brand        TEXT,
            price        TEXT,
            price_numeric REAL,
            discount_pct REAL,
            image_path   TEXT,
            image_hash   TEXT,
            timestamp    TEXT
        );
    """)
    conn.commit()
    return conn
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Image Deduplication (MD5)
# ─────────────────────────────────────────────────────────────────────────────
 
def compute_image_hash(image_path: str) -> str:
    """Return MD5 hex digest of image file bytes."""
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
 
 
def is_duplicate_image(image_hash: str) -> dict | None:
    """
    Returns existing record if this image was already analyzed, else None.
    dict keys: image_path, scan_id, analyzed_at
    READ-ONLY — no lock needed.
    """
    conn = _get_sqlite()
    row  = conn.execute(
        "SELECT * FROM image_hashes WHERE image_hash=?", (image_hash,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None
 
 
def _register_image_hash(conn: sqlite3.Connection, image_hash: str, image_path: str, scan_id: str):
    """
    Insert image hash record using a SHARED connection.
    Caller must hold _db_write_lock and call conn.commit() themselves.
    """
    conn.execute(
        "INSERT OR IGNORE INTO image_hashes (image_hash, image_path, scan_id, analyzed_at) VALUES (?,?,?,?)",
        (image_hash, image_path, scan_id, datetime.now().isoformat())
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Price parsing helper
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_price(price_str) -> float | None:
    """Extract numeric value from price strings like '₹49', '$3.99', '49.00'."""
    if not price_str:
        return None
    import re
    m = re.search(r"[\d]+\.?[\d]*", str(price_str).replace(",", ""))
    return float(m.group()) if m else None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Price Change Detection
# ─────────────────────────────────────────────────────────────────────────────
 
def _product_key(name: str, brand: str) -> str:
    """Normalised key for matching same product across scans."""
    return f"{name.strip().lower()}|{brand.strip().lower()}"
 
 
def _detect_price_changes(
    conn: sqlite3.Connection,
    products: list[dict],
    image_path: str,
    image_hash: str
) -> list[dict]:
    """
    Compare current scan prices against most-recent price in price_history.
    Uses a SHARED connection — caller holds _db_write_lock.
    Returns list of change dicts for products whose price changed.
    """
    changes = []
    now     = datetime.now().isoformat()
 
    for p in products:
        key          = _product_key(p.get("name",""), p.get("brand",""))
        price_now    = _parse_price(p.get("price"))
        discount_now = p.get("discount_percentage")
 
        # Fetch most recent historical price for this product
        prev = conn.execute("""
            SELECT price, price_numeric, discount_pct, timestamp, image_path
            FROM price_history
            WHERE product_key=?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (key,)).fetchone()
 
        if prev:
            prev_numeric = prev["price_numeric"]
            if price_now is not None and prev_numeric is not None:
                delta     = price_now - prev_numeric
                delta_pct = round((delta / prev_numeric) * 100, 1) if prev_numeric else 0
 
                if abs(delta) > 0.01:          # price actually changed
                    changes.append({
                        "product_name":   p.get("name",""),
                        "brand":          p.get("brand",""),
                        "category":       p.get("category",""),
                        "prev_price":     prev["price"],
                        "curr_price":     p.get("price",""),
                        "delta":          round(delta, 2),
                        "delta_pct":      round(delta_pct, 1),
                        "direction":      "increased" if delta > 0 else "decreased",
                        "prev_scan_date": prev["timestamp"][:10],
                        "curr_scan_date": now[:10],
                        "prev_image":     prev["image_path"],
                        "curr_image":     image_path,
                    })
 
        # Always upsert into price_history
        conn.execute("""
            INSERT INTO price_history
            (product_key, product_name, brand, price, price_numeric, discount_pct, image_path, image_hash, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            key, p.get("name",""), p.get("brand",""),
            p.get("price",""), price_now, discount_now,
            image_path, image_hash, now
        ))
 
    return changes
 
 
def get_price_change_history() -> list[dict]:
    """Return all detected price changes across all scans, newest first. READ-ONLY."""
    conn = _get_sqlite()
    rows = conn.execute("""
        SELECT ph1.product_name, ph1.brand, ph1.price as curr_price,
               ph2.price as prev_price,
               ph1.price_numeric - ph2.price_numeric as delta,
               ph1.timestamp as curr_date, ph2.timestamp as prev_date,
               ph1.image_path as curr_image, ph2.image_path as prev_image
        FROM price_history ph1
        JOIN price_history ph2
          ON ph1.product_key = ph2.product_key
         AND ph1.timestamp > ph2.timestamp
        GROUP BY ph1.product_key
        ORDER BY ph1.timestamp DESC
        LIMIT 50
    """).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["direction"] = "increased" if (d["delta"] or 0) > 0 else "decreased"
        #d["delta_pct"] = round((d["delta"] / (_parse_price(d["prev_price"]) or 1)) * 100, 1) if d["prev_price"] else 0
        d["delta_pct"] = round(((d["delta"] or 0) / (_parse_price(d["prev_price"]) or 1)) * 100, 1) if d["prev_price"] else 0
        result.append(d)
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Per-Product ChromaDB Embedding
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_product_text(p: dict, promotions: list[dict], image_path: str) -> str:
    """
    Rich natural-language sentence for a single product.
    Includes category synonyms so 'snacks', 'chips', 'crisps' all match.
    """
    name     = p.get("name", "")
    brand    = p.get("brand", "")
    cat      = p.get("category", "")
    price    = p.get("price", "")
    orig     = p.get("original_price", "")
    disc     = p.get("discount_percentage", "")
    unit     = p.get("unit", "")
    qty      = p.get("quantity_available", "")
    status   = p.get("inventory_status", "unknown")
    inv_note = p.get("inventory_notes", "")
 
    # Find any promotions that mention this product
    my_promos = [
        pr.get("details","") for pr in promotions
        if name.lower() in pr.get("product","").lower()
    ]
 
    text = f"{name} by {brand}, category: {cat}, price: {price}"
    if orig:  text += f", original price: {orig}"
    if disc:  text += f", {disc}% discount"
    if unit:  text += f", pack size: {unit}"
    if qty:   text += f", quantity on shelf: {qty}"
    text += f", inventory status: {status}"
    if inv_note: text += f", notes: {inv_note}"
    if my_promos: text += f", promotions: {'; '.join(my_promos)}"
    text += f", seen in image: {os.path.basename(image_path)}"
    return text
 
 
def store_analysis(image_path: str, analysis: dict, session_id: str = None) -> dict:
    """
    Store analysis into ChromaDB (per-product) + SQLite.
 
    KEY FIX: All SQLite writes happen on ONE shared connection inside a single
    _db_write_lock block, eliminating the 'database is locked' error that
    occurred when _register_image_hash and _detect_price_changes each opened
    their own connection concurrently during parallel image analysis.
 
    Returns dict with scan_id and price_changes list.
    """
    prod_col   = get_product_collection()
    img_col    = get_image_collection()
    embedder   = get_embedder()
 
    products   = analysis.get("products", [])
    promotions = analysis.get("promotions", [])
    best_value = analysis.get("best_value_product", "")
    timestamp  = datetime.now().isoformat()
    scan_id    = str(uuid.uuid4())
 
    # ── Pre-compute embeddings OUTSIDE the lock (CPU/network work, not DB) ──
    product_texts      = []
    product_embeddings = []
    product_meta       = []
    status_counts      = {"present": 0, "low": 0, "absent": 0, "unknown": 0}
 
    image_hash = compute_image_hash(image_path)   # pure file I/O, no DB
 
    for p in products:
        status = p.get("inventory_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
 
        product_text = _build_product_text(p, promotions, image_path)
        embedding    = embedder.encode(product_text).tolist()
 
        try:
            qty = int(str(p.get("quantity_available") or 0).split()[0])
        except (ValueError, TypeError):
            qty = None
 
        product_texts.append(product_text)
        product_embeddings.append(embedding)
        product_meta.append({
            "p": p, "qty": qty, "status": status,
            "product_text": product_text, "embedding": embedding
        })
 
    img_text = (
        f"Scan of {os.path.basename(image_path)} with {len(products)} products. "
        f"Categories: {', '.join(set(p.get('category','') for p in products))}. "
        f"Best value: {best_value}."
    )
    img_embedding = embedder.encode(img_text).tolist()
 
    # ── All SQLite writes in ONE connection under ONE lock ───────────────────
    price_changes = []
    with _db_write_lock:
        conn = _get_sqlite()
        try:
            # 1. Register image hash
            _register_image_hash(conn, image_hash, image_path, scan_id)
 
            # 2. Detect price changes (reads + writes price_history)
            price_changes = _detect_price_changes(conn, products, image_path, image_hash)
 
            # 3. Write inventory log rows for each product
            for meta in product_meta:
                p   = meta["p"]
                qty = meta["qty"]
                conn.execute("""
                    INSERT INTO inventory_log
                    (scan_id, product_name, brand, category, quantity, inventory_status,
                     price, price_numeric, discount_pct, image_path, image_hash, timestamp)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    scan_id, p.get("name",""), p.get("brand",""), p.get("category",""),
                    qty, meta["status"], p.get("price",""), _parse_price(p.get("price")),
                    p.get("discount_percentage"), image_path, image_hash, timestamp
                ))
 
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
 
    # ── ChromaDB writes (thread-safe internally, no lock needed) ─────────────
    for meta in product_meta:
        p   = meta["p"]
        qty = meta["qty"]
        prod_col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[meta["embedding"]],
            documents=[meta["product_text"]],
            metadatas=[{
                "scan_id":           scan_id,
                "image_path":        image_path,
                "image_hash":        image_hash,
                "timestamp":         timestamp,
                "session_id":        session_id or "",
                "product_name":      p.get("name", ""),
                "brand":             p.get("brand", ""),
                "category":          p.get("category", ""),
                "price":             str(p.get("price", "")),
                "price_numeric":     str(_parse_price(p.get("price")) or ""),
                "discount_pct":      str(p.get("discount_percentage") or ""),
                "unit":              str(p.get("unit", "")),
                "quantity":          str(qty or ""),
                "inventory_status":  meta["status"],
            }]
        )
 
    img_col.add(
        ids=[scan_id],
        embeddings=[img_embedding],
        documents=[img_text],
        metadatas=[{
            "image_path":      image_path,
            "image_hash":      image_hash,
            "timestamp":       timestamp,
            "session_id":      session_id or "",
            "product_count":   len(products),
            "promotion_count": len(promotions),
            "best_value":      best_value,
            "present_count":   status_counts["present"],
            "low_count":       status_counts["low"],
            "absent_count":    status_counts["absent"],
            "analysis_json":   json.dumps(analysis),
            "price_changes":   json.dumps(price_changes),
        }]
    )
 
    return {"scan_id": scan_id, "price_changes": price_changes}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RAG Query  — now hits per-product collection
# ─────────────────────────────────────────────────────────────────────────────
 
def query_products(user_query: str, n_results: int = 8) -> list[dict]:
    """
    Semantic search across individual product embeddings.
    Returns deduplicated list sorted by relevance.
    """
    prod_col = get_product_collection()
    embedder = get_embedder()
 
    if prod_col.count() == 0:
        return []
 
    query_embedding = embedder.encode(user_query).tolist()
    results = prod_col.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, prod_col.count()),
        include=["documents", "metadatas", "distances"]
    )
 
    output = []
    seen   = set()  # deduplicate by product_name+brand
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        key = f"{meta.get('product_name','')}|{meta.get('brand','')}"
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "text_chunk":       doc,
            "image_path":       meta.get("image_path"),
            "timestamp":        meta.get("timestamp"),
            "product_name":     meta.get("product_name"),
            "brand":            meta.get("brand"),
            "category":         meta.get("category"),
            "price":            meta.get("price"),
            "discount_pct":     meta.get("discount_pct"),
            "quantity":         meta.get("quantity"),
            "inventory_status": meta.get("inventory_status"),
            "similarity_score": round(1 - dist, 3),
        })
 
    return output
 
 
# ─────────────────────────────────────────────────────────────────────────────
# History (uses image collection)
# ─────────────────────────────────────────────────────────────────────────────
 
def get_all_history() -> list[dict]:
    img_col = get_image_collection()
    if img_col.count() == 0:
        return []
 
    results = img_col.get(include=["metadatas", "documents"])
    items   = []
    for meta in results["metadatas"]:
        analysis = json.loads(meta.get("analysis_json", "{}"))
        items.append({
            "image_path":      meta.get("image_path"),
            "image_hash":      meta.get("image_hash",""),
            "timestamp":       meta.get("timestamp"),
            "product_count":   meta.get("product_count", 0),
            "promotion_count": meta.get("promotion_count", 0),
            "best_value":      meta.get("best_value", ""),
            "present_count":   meta.get("present_count", 0),
            "low_count":       meta.get("low_count", 0),
            "absent_count":    meta.get("absent_count", 0),
            "price_changes":   json.loads(meta.get("price_changes","[]")),
            "analysis":        analysis,
        })
 
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return items
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Inventory helpers (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────
 
def get_inventory_summary() -> dict:
    conn = _get_sqlite()
    rows = conn.execute("""
        SELECT inventory_status, COUNT(*) as cnt
        FROM inventory_log GROUP BY inventory_status
    """).fetchall()
 
    low_items = conn.execute("""
        SELECT product_name, brand, category, quantity, inventory_status, price, image_path, timestamp
        FROM inventory_log
        WHERE inventory_status IN ('low','absent')
        ORDER BY quantity ASC LIMIT 20
    """).fetchall()
 
    conn.close()
    return {
        "status_counts":   {r["inventory_status"]: r["cnt"] for r in rows},
        "low_stock_items": [dict(r) for r in low_items]
    }
 
 
def get_inventory_table() -> list[dict]:
    conn = _get_sqlite()
    rows = conn.execute("""
        SELECT product_name, brand, category, quantity, inventory_status,
               price, discount_pct, MAX(timestamp) as last_seen
        FROM inventory_log
        GROUP BY product_name, brand
        ORDER BY inventory_status ASC, product_name ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Session helpers (unchanged interface)
# ─────────────────────────────────────────────────────────────────────────────
 
def create_session(label: str = None) -> str:
    with _db_write_lock:
        conn = _get_sqlite()
        sid  = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, label, image_paths, analysis_ids) VALUES (?,?,?,?,?)",
            (sid, datetime.now().isoformat(),
             label or f"Session {datetime.now().strftime('%d %b %H:%M')}", "[]", "[]")
        )
        conn.commit()
        conn.close()
    return sid
 
 
def update_session(session_id: str, image_path: str, doc_id: str):
    with _db_write_lock:
        conn = _get_sqlite()
        row  = conn.execute(
            "SELECT image_paths, analysis_ids FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row:
            paths = json.loads(row["image_paths"]); paths.append(image_path)
            ids   = json.loads(row["analysis_ids"]); ids.append(doc_id)
            conn.execute(
                "UPDATE sessions SET image_paths=?, analysis_ids=? WHERE session_id=?",
                (json.dumps(paths), json.dumps(ids), session_id)
            )
            conn.commit()
        conn.close()
 
 
def get_all_sessions() -> list[dict]:
    conn = _get_sqlite()
    rows = conn.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]
 
 
def save_chat_message(session_id: str, role: str, content: str):
    with _db_write_lock:
        conn = _get_sqlite()
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            (session_id, role, content, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
 
 
def load_chat_history(session_id: str) -> list[dict]:
    conn = _get_sqlite()
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows]
 
 
def get_db_stats() -> dict:
    prod_col = get_product_collection()
    img_col  = get_image_collection()
    conn     = _get_sqlite()
    sessions = conn.execute("SELECT COUNT(*) as c FROM sessions").fetchone()["c"]
    conn.close()
    return {
        "total_analyses":    img_col.count(),
        "total_products_db": prod_col.count(),
        "total_sessions":    sessions,
        "db_path":           CHROMA_DIR,
    }
 
 
 
 
def _name_similarity(a: str, b: str) -> float:
    """Fuzzy string similarity between two product names (0.0 – 1.0)."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
 
 
def get_source_type_from_history() -> list[dict]:
    """
    Pull all history items with their source_type tag.
    source_type is stored in shelf_summary.image_type → 'shelf' or 'flyer'.
    """
    history = get_all_history()
    tagged = []
    for item in history:
        image_type = item["analysis"].get("shelf_summary", {}).get("image_type", "shelf")
        item["source_type"] = "flyer" if image_type == "flyer" else "shelf"
        tagged.append(item)
    return tagged
 
 
def check_planogram_compliance(name_threshold: float = 0.75) -> list[dict]:
    """
    For every product promoted in a flyer, check whether it's actually present
    on the shelf. Flags promoted-but-missing products (compliance gap).
    """
    history = get_source_type_from_history()
    shelf_names, flyer_promo_products = [], []
 
    for item in history:
        src = item.get("source_type", "shelf")
        if src == "shelf":
            shelf_names.extend(
                p.get("name", "") for p in item["analysis"].get("products", [])
            )
        else:
            for promo in item["analysis"].get("promotions", []):
                flyer_promo_products.append({
                    "product":      promo.get("product", ""),
                    "promo_details": promo.get("details", ""),
                    "savings":      promo.get("savings_amount", ""),
                    "validity":     promo.get("validity", ""),
                    "timestamp":    item["timestamp"],
                })
 
    report = []
    for fp in flyer_promo_products:
        if not fp["product"]:
            continue
        best_score = max(
            (_name_similarity(fp["product"], s) for s in shelf_names),
            default=0.0
        )
        report.append({
            "promoted_product":  fp["product"],
            "promotion":         fp["promo_details"],
            "savings":           fp["savings"],
            "validity":          fp["validity"],
            "found_on_shelf":    best_score >= name_threshold,
            "match_confidence":  round(best_score, 2),
            "flyer_date":        fp["timestamp"][:10],
        })
 
    return sorted(report, key=lambda x: x["found_on_shelf"])  # missing first