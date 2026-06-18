import os
import requests
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

VLLM_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"

# ── Demo mode: set DEMO_MODE=true in environment or .env to bypass vLLM ──────
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() in ("true", "1", "yes")


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _infer_inventory_status(qty) -> str:
    if qty is None:
        return "unknown"
    try:
        q = int(str(qty).split()[0])
    except (ValueError, TypeError):
        return "unknown"
    if q == 0:   return "absent"
    if q <= 3:   return "low"
    return "present"


PROMPT = """You are a retail shelf intelligence system with expert vision. Analyze this retail image carefully.

Return ONLY a valid JSON object — NO markdown, NO backticks, NO extra text outside the JSON:

{
  "products": [
    {
      "name": "Exact product name from label",
      "brand": "Brand name",
      "category": "snacks|beverages|dairy|personal_care|hair_care|household|frozen|bakery|produce|other",
      "price": "₹49 (exactly as shown, null if not visible)",
      "original_price": "₹99 (only if struck-through / MRP shown, else null)",
      "discount_percentage": 50,
      "unit": "500g or 1L or 6-pack (pack size if visible, else null)",
      "quantity_available": 5,
      "inventory_notes": "Shelf looks well-stocked / nearly empty / out of stock / single unit remaining"
    }
  ],
  "promotions": [
    {
      "type": "percentage_off|buy_one_get_one|combo_offer|flat_discount|cashback|seasonal_sale|free_gift",
      "details": "Complete human-readable promotion description",
      "product": "Product name this applies to",
      "savings_amount": "₹50 or null",
      "validity": "Offer valid till DD/MM/YYYY or null if not shown"
    }
  ],
  "best_value_product": "Product name offering the best price-to-value or highest discount",
  "shelf_summary": {
    "total_products_visible": 12,
    "categories_present": ["snacks", "beverages"],
    "price_range": {"min": "₹10", "max": "₹500"},
    "has_active_promotions": true,
    "overall_stock_level": "well_stocked|partially_stocked|low_stock|out_of_stock",
    "image_type": "shelf|flyer|display|mixed"
  }
}

Rules:
- quantity_available: count visible product units/facings as integer. Use null for flyer images.
- Extract ALL visible products, even partially visible
- Read prices EXACTLY as shown on labels/price tags
- Identify promotions from banners, shelf talkers, stickers, labels
- Return ONLY the JSON object, nothing else"""


def _call_vllm(image_path: str) -> dict:
    """Make the actual vLLM API call."""
    image_b64 = encode_image(image_path)
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }],
        "temperature": 0.1,
        "max_tokens": 2500
    }
    response = requests.post(VLLM_URL, json=payload, timeout=90)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    content = content.replace("```json", "").replace("```", "").strip()
    start, end = content.find("{"), content.rfind("}") + 1
    if start != -1 and end > start:
        content = content[start:end]
    return json.loads(content)


def analyze_retail_image(image_path: str, demo_index: int = 0) -> dict:
    """
    Analyze a single retail image.
    Falls back to demo data if DEMO_MODE=true or if vLLM is unreachable.
    """
    if DEMO_MODE:
        from demo_data import get_demo_result
        return get_demo_result(image_path, demo_index)

    try:
        parsed = _call_vllm(image_path)
    except requests.exceptions.ConnectionError:
        # vLLM server not running — auto-fallback to demo data
        from demo_data import get_demo_result
        result = get_demo_result(image_path, demo_index)
        result["_demo_mode"] = True
        result["_fallback_reason"] = "vLLM server unreachable — showing demo data"
        return result
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}", "products": [], "promotions": [], "best_value_product": "", "shelf_summary": {}, "source_image": image_path}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "products": [], "promotions": [], "best_value_product": "", "shelf_summary": {}, "source_image": image_path}
    except Exception as e:
        return {"error": f"Unexpected error: {e}", "products": [], "promotions": [], "best_value_product": "", "shelf_summary": {}, "source_image": image_path}

    # Enrich with computed inventory_status
    parsed.setdefault("products", [])
    parsed.setdefault("promotions", [])
    parsed.setdefault("best_value_product", "")
    parsed.setdefault("shelf_summary", {})
    for p in parsed["products"]:
        p["inventory_status"] = _infer_inventory_status(p.get("quantity_available"))
    parsed["source_image"] = image_path
    return parsed


def analyze_multiple_images(image_paths: list[str], max_workers: int = 3) -> list[dict]:
    """Analyze multiple images in parallel. Returns results in same order as input."""
    results = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(analyze_retail_image, path, i): i
            for i, path in enumerate(image_paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "error": str(e), "products": [], "promotions": [],
                    "best_value_product": "", "shelf_summary": {},
                    "source_image": image_paths[idx]
                }
    return results