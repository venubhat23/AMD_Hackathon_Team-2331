"""
Demo mode fallback data — used when DEMO_MODE=True in .env or vLLM is unreachable.
Simulates realistic retail shelf analysis results so the app runs fully without a GPU.
"""

DEMO_RESULTS = [
    {
        "source_image": "demo_shelf_1.jpg",
        "products": [
            {"name": "Lays Classic Salted", "brand": "Lays", "category": "snacks",
             "price": "₹20", "original_price": "₹25", "discount_percentage": 20,
             "unit": "26g", "quantity_available": 8, "inventory_status": "present",
             "inventory_notes": "Well stocked, front row full"},
            {"name": "Kurkure Masala Munch", "brand": "Kurkure", "category": "snacks",
             "price": "₹10", "original_price": None, "discount_percentage": None,
             "unit": "22g", "quantity_available": 3, "inventory_status": "low",
             "inventory_notes": "Only 3 packets remaining"},
            {"name": "Pepsi Cola 600ml", "brand": "Pepsi", "category": "beverages",
             "price": "₹40", "original_price": "₹45", "discount_percentage": 11,
             "unit": "600ml", "quantity_available": 12, "inventory_status": "present",
             "inventory_notes": "Full shelf, multiple rows deep"},
            {"name": "Coca-Cola 750ml", "brand": "Coca-Cola", "category": "beverages",
             "price": "₹45", "original_price": None, "discount_percentage": None,
             "unit": "750ml", "quantity_available": 0, "inventory_status": "absent",
             "inventory_notes": "Empty shelf space, price tag present but no stock"},
            {"name": "Britannia Marie Gold", "brand": "Britannia", "category": "bakery",
             "price": "₹35", "original_price": "₹40", "discount_percentage": 12,
             "unit": "250g", "quantity_available": 6, "inventory_status": "present",
             "inventory_notes": "Good stock level"},
            {"name": "Maggi 2-Minute Noodles", "brand": "Nestle", "category": "snacks",
             "price": "₹14", "original_price": None, "discount_percentage": None,
             "unit": "70g", "quantity_available": 2, "inventory_status": "low",
             "inventory_notes": "Nearly out, needs restocking"},
        ],
        "promotions": [
            {"type": "percentage_off", "details": "20% off on Lays Classic", "product": "Lays Classic Salted", "savings_amount": "₹5", "validity": None},
            {"type": "combo_offer",    "details": "Buy 2 Pepsi get 1 free",   "product": "Pepsi Cola 600ml",   "savings_amount": "₹40", "validity": "31 Dec 2025"},
            {"type": "flat_discount",  "details": "₹5 off on Britannia Marie Gold", "product": "Britannia Marie Gold", "savings_amount": "₹5", "validity": None},
        ],
        "best_value_product": "Pepsi Cola 600ml (Buy 2 Get 1 Free)",
        "shelf_summary": {
            "total_products_visible": 6,
            "categories_present": ["snacks", "beverages", "bakery"],
            "price_range": {"min": "₹10", "max": "₹45"},
            "has_active_promotions": True,
            "overall_stock_level": "partially_stocked",
            "image_type": "shelf"
        }
    },
    {
        "source_image": "demo_flyer_1.jpg",
        "products": [
            {"name": "Pantene Shampoo Anti-Dandruff", "brand": "Pantene", "category": "hair_care",
             "price": "₹199", "original_price": "₹299", "discount_percentage": 33,
             "unit": "340ml", "quantity_available": None, "inventory_status": "unknown",
             "inventory_notes": "Flyer image — stock count not applicable"},
            {"name": "Head & Shoulders Cool Menthol", "brand": "Head & Shoulders", "category": "hair_care",
             "price": "₹249", "original_price": "₹349", "discount_percentage": 28,
             "unit": "400ml", "quantity_available": None, "inventory_status": "unknown",
             "inventory_notes": "Flyer image"},
            {"name": "Dove Intense Repair Conditioner", "brand": "Dove", "category": "personal_care",
             "price": "₹179", "original_price": "₹229", "discount_percentage": 21,
             "unit": "180ml", "quantity_available": None, "inventory_status": "unknown",
             "inventory_notes": "Flyer image"},
            {"name": "Clinic Plus Strong & Long", "brand": "Clinic Plus", "category": "hair_care",
             "price": "₹85", "original_price": "₹110", "discount_percentage": 22,
             "unit": "175ml", "quantity_available": None, "inventory_status": "unknown",
             "inventory_notes": "Flyer image"},
        ],
        "promotions": [
            {"type": "percentage_off", "details": "Up to 33% off on hair care range", "product": "Pantene Shampoo Anti-Dandruff", "savings_amount": "₹100", "validity": "15 Jan 2026"},
            {"type": "buy_one_get_one", "details": "Buy any 2 shampoos get 1 conditioner free", "product": "Head & Shoulders Cool Menthol", "savings_amount": "₹179", "validity": "31 Dec 2025"},
        ],
        "best_value_product": "Pantene Shampoo Anti-Dandruff (33% off, save ₹100)",
        "shelf_summary": {
            "total_products_visible": 4,
            "categories_present": ["hair_care", "personal_care"],
            "price_range": {"min": "₹85", "max": "₹249"},
            "has_active_promotions": True,
            "overall_stock_level": "well_stocked",
            "image_type": "flyer"
        }
    }
]


def get_demo_result(image_path: str, index: int = 0) -> dict:
    """
    Return a demo analysis result for the given image path.
    Cycles through DEMO_RESULTS if multiple images uploaded.
    """
    result = DEMO_RESULTS[index % len(DEMO_RESULTS)].copy()
    result["source_image"] = image_path
    result["_demo_mode"] = True
    return result