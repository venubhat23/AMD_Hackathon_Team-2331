import os
import requests
import json
from vector_store import query_products

VLLM_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"
DEMO_MODE  = os.environ.get("DEMO_MODE", "false").lower() in ("true", "1", "yes")

SYSTEM_PROMPT = """You are a smart retail store assistant AI powered by computer vision and a vector product database.
You help store managers and customers find products, check stock levels, compare prices, and discover promotions.

When answering:
- Be concise and specific
- Always mention inventory status: ✅ Present / ⚠️ Low Stock / ❌ Out of Stock
- Highlight promotions and best deals prominently
- Format prices clearly (₹49)
- Group similar products when listing
- If no data found in database, say so and suggest uploading shelf images
"""

DEMO_ANSWERS = {
    "snack": "From our scanned shelves:\n• ✅ **Lays Classic Salted** (₹20, 20% off) — 8 units in stock\n• ⚠️ **Kurkure Masala Munch** (₹10) — only 3 units left, needs restocking\n• ⚠️ **Maggi 2-Minute Noodles** (₹14) — only 2 units remaining\n\n**Best deal:** Lays at 20% off (save ₹5 per pack).",
    "hair":  "Hair care products from scanned flyers:\n• ✅ **Pantene Anti-Dandruff Shampoo** — ₹199 (was ₹299, **33% off**, save ₹100)\n• ✅ **Head & Shoulders Cool Menthol** — ₹249 (was ₹349, 28% off)\n• ✅ **Clinic Plus Strong & Long** — ₹85 (was ₹110, 22% off)\n• ✅ **Dove Intense Repair Conditioner** — ₹179 (was ₹229, 21% off)\n\n🏷️ **Active promo:** Buy any 2 shampoos, get 1 conditioner FREE (save ₹179!)",
    "beverage": "Beverages found in shelf scans:\n• ✅ **Pepsi Cola 600ml** — ₹40 (was ₹45) | 12 units | 🏷️ Buy 2 get 1 FREE\n• ❌ **Coca-Cola 750ml** — ₹45 | **OUT OF STOCK** — needs urgent restocking",
    "stock":  "⚠️ **Low/Out of Stock Alert:**\n• ❌ Coca-Cola 750ml — OUT OF STOCK\n• ⚠️ Kurkure Masala Munch — only 3 units\n• ⚠️ Maggi 2-Minute Noodles — only 2 units\n\nRecommendation: Restock these 3 items before next audit.",
    "deal":   "🏆 **Best Deals Right Now:**\n1. Pepsi 600ml — Buy 2 Get 1 FREE (save ₹40)\n2. Pantene Shampoo — 33% off (save ₹100)\n3. Lays Classic — 20% off (save ₹5)\n4. Dove Conditioner — 21% off (save ₹50)",
    "default": "Based on our scanned shelf data, I found products across categories: snacks, beverages, hair care, and bakery. Ask me about specific categories like 'show snacks', 'hair products', 'beverages', or 'what is low on stock'."
}


def _demo_answer(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["snack","chip","kurkure","lays","maggi","biscuit","cookie"]):
        return DEMO_ANSWERS["snack"]
    if any(w in q for w in ["hair","shampoo","conditioner","pantene","dove","dandruff"]):
        return DEMO_ANSWERS["hair"]
    if any(w in q for w in ["beverage","drink","pepsi","cola","water","juice"]):
        return DEMO_ANSWERS["beverage"]
    if any(w in q for w in ["stock","low","absent","empty","restock","missing"]):
        return DEMO_ANSWERS["stock"]
    if any(w in q for w in ["deal","discount","offer","best","cheap","save","promo"]):
        return DEMO_ANSWERS["deal"]
    return DEMO_ANSWERS["default"]


def build_rag_context(user_query: str) -> str:
    """
    Retrieve per-product embeddings from ChromaDB and build a structured context string.
    Each result is now a single product (not a whole image), giving much higher precision.
    """
    results = query_products(user_query, n_results=8)
    if not results:
        return ""

    lines = []
    for r in results:
        status = r.get("inventory_status", "unknown")
        emoji  = {"present": "✅", "low": "⚠️", "absent": "❌"}.get(status, "❓")
        line   = (
            f"{emoji} {r.get('product_name','')} ({r.get('brand','')}) "
            f"[{r.get('category','')}] "
            f"Price: {r.get('price','')} "
            f"Qty: {r.get('quantity','')} "
            f"Status: {status}"
        )
        if r.get("discount_pct"):
            line += f" | {r['discount_pct']}% off"
        line += f" | Relevance: {r.get('similarity_score', '')}"
        lines.append(line)

    return "Products from database (per-product vector search):\n" + "\n".join(lines)


def chat_with_retail_ai(user_query: str, chat_history: list[dict]) -> str:
    """RAG-powered chat. Auto-falls back to demo answers if vLLM unreachable."""

    rag_context = build_rag_context(user_query)

    # Demo mode or empty DB — use canned answers
    if DEMO_MODE:
        return _demo_answer(user_query)

    if rag_context:
        augmented = (
            f"{rag_context}\n\n"
            f"User question: {user_query}\n\n"
            "Answer using the product data above. Include inventory status and prices."
        )
    else:
        augmented = (
            f"{user_query}\n\n"
            "[No products found in database. Tell user to upload shelf images first.]"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": augmented})

    try:
        response = requests.post(
            VLLM_URL,
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0.3, "max_tokens": 800},
            timeout=30
        ) 
        # Raise exception for HTTP 4xx/5xx
        response.raise_for_status()

        data = response.json()

    # Debug: print actual vLLM response
        print("vLLM Response:", data)

    # Check for expected OpenAI-compatible format
        if "choices" not in data:
            return f"⚠️ vLLM returned unexpected response:\n{data}"
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        # Auto-fallback to demo answers when vLLM is down
        return _demo_answer(user_query)
    except Exception as e:
        return f"⚠️ Error: {e}"


def get_suggested_questions() -> list[str]:
    return [
        "What snack products are available?",
        "Show me all hair care items",
        "Which products are low on stock?",
        "What's the best deal today?",
        "Are there any BOGO promotions?",
        "List all beverages and their prices",
        "What products are out of stock?",
        "Which brand appears most often?",
    ]