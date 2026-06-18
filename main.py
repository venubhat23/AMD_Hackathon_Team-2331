import os
import json
from datetime import datetime
import streamlit as st
from PIL import Image
import pandas as pd

from llm_service import analyze_retail_image, analyze_multiple_images, DEMO_MODE
from vector_store import (
    store_analysis, get_all_history, get_db_stats,
    create_session, update_session, get_all_sessions,
    save_chat_message, load_chat_history,
    get_inventory_summary, get_inventory_table,
    compute_image_hash, is_duplicate_image,
    get_price_change_history,
    check_planogram_compliance,          # ← add this
)
from chatbot_service import chat_with_retail_ai, get_suggested_questions

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Retail Intelligence AI", layout="wide", page_icon="🛒")

st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0d1117;
}

/* Radio option text */
[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #00E5FF !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Hover */
[data-testid="stSidebar"] [role="radiogroup"] label:hover p {
    color: #FFD700 !important;
}

/* Selected item */
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] p {
    color: #22C55E !important;
    font-weight: 700 !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Caption */
[data-testid="stSidebar"] .stCaption {
    color: #9ca3af !important;
}

</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("active_session_id", None),
    ("active_session_label", None),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ───────────────────────────────────────────────────────────────────
INVENTORY_EMOJI = {
    "present": "✅ Present", "low": "⚠️ Low",
    "absent":  "❌ Absent",  "unknown": "❓ Unknown"
}

def status_badge(s):
    return INVENTORY_EMOJI.get(s, "❓ Unknown")


def render_products_table(products):
    if not products:
        st.info("No products detected.")
        return
    rows = [{
        "Product":   p.get("name",""),
        "Brand":     p.get("brand",""),
        "Category":  p.get("category",""),
        "Price":     p.get("price",""),
        "Orig":      p.get("original_price",""),
        "Disc%":     p.get("discount_percentage",""),
        "Unit":      p.get("unit",""),
        "Inventory": status_badge(p.get("inventory_status","unknown")),
    } for p in products]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_price_changes(changes: list[dict]):
    """Render a price-change alert table."""
    if not changes:
        return
    st.warning(f"🔔 **{len(changes)} price change(s) detected vs previous scan!**")
    rows = []
    for c in changes:
        arrow = "🔺" if c["direction"] == "increased" else "🔻"
        rows.append({
            "Product":    c["product_name"],
            "Brand":      c.get("brand",""),
            "Prev Price": c["prev_price"],
            "New Price":  c["curr_price"],
            "Change":     f"{arrow} {c['delta']:+.2f} ({c['delta_pct']:+.1f}%)",
            "Prev Scan":  c.get("prev_scan_date",""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=55)
    st.title("Retail AI")
    st.caption("Vision · RAG · Inventory · Chatbot")

    if DEMO_MODE:
        st.success("🎭 DEMO MODE ON")

    st.divider()

    page = st.radio("Navigation", [
        "📸 Analyze Images",
        "💬 Product Chatbot",
        "📦 Inventory Status",   
        "💰 Price Changes",
        "🕒 History",
        "📊 Analytics",
        "🗂️ Sessions",
    ], label_visibility="collapsed")

    st.divider()

    st.subheader("🗂️ Active Session")
    if st.session_state.active_session_id:
        st.success(f"**{st.session_state.active_session_label}**")
        if st.button("➕ New Session"):
            st.session_state.active_session_id = None
            st.session_state.active_session_label = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        label = st.text_input("Session name", placeholder="e.g. Morning Audit")
        if st.button("▶ Start Session", type="primary"):
            sid = create_session(label or None)
            st.session_state.active_session_id = sid
            st.session_state.active_session_label = label or f"Session {sid[:6]}"
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    stats = get_db_stats()
    st.metric("Images Analyzed",  stats["total_analyses"])
    st.metric("Products in DB",   stats["total_products_db"])
    st.metric("Sessions",         stats["total_sessions"])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE IMAGES
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📸 Analyze Images":
    st.title("📸 Retail Shelf & Flyer Analyzer")
    st.caption("Upload one or more images — parallel analysis, dedup detection, price change alerts.")

    if DEMO_MODE:
        st.markdown('<div class="demo-banner">🎭 <b>Demo Mode Active</b> — vLLM not required. Pre-loaded sample data will be used.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop shelf images or flyers here",
        type=["jpg","jpeg","png","webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Save files and check for duplicates via MD5
        saved_paths = []
        dup_info    = {}   # path → existing record if duplicate

        cols = st.columns(min(len(uploaded_files), 4))
        for i, uf in enumerate(uploaded_files):
            path = os.path.join(UPLOAD_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(path)

            img_hash  = compute_image_hash(path)
            dup_check = is_duplicate_image(img_hash)
            if dup_check:
                dup_info[path] = dup_check

            with cols[i % 4]:
                st.image(Image.open(path), caption=uf.name, use_container_width=True)
                if dup_check:
                    st.caption(f"⚠️ Already analyzed on {dup_check['analyzed_at'][:10]}")

        # Duplicate warning
        if dup_info:
            dup_names = [os.path.basename(p) for p in dup_info]
            st.markdown(
                f'<div class="dedup-banner">ℹ️ <b>Duplicate detected:</b> {", ".join(dup_names)} '
                f'were already analyzed. Re-analyzing will store another record.</div>',
                unsafe_allow_html=True
            )

        st.divider()
        c_btn, c_info = st.columns([1, 3])
        with c_btn:
            btn_label = f"🔍 Analyze {len(uploaded_files)} Image{'s' if len(uploaded_files)>1 else ''}"
            analyze_btn = st.button(btn_label, use_container_width=True, type="primary")
        with c_info:
            if dup_info:
                force = st.checkbox("Re-analyze duplicates anyway", value=False)
            else:
                force = True
            if len(uploaded_files) > 1:
                st.info(f"⚡ {len(uploaded_files)} images analyzed in parallel.")

        if analyze_btn:
            # Filter out duplicates unless forced
            paths_to_analyze = [
                p for p in saved_paths
                if force or p not in dup_info
            ]
            skipped = [p for p in saved_paths if p not in paths_to_analyze]

            if skipped:
                st.info(f"⏭️ Skipped {len(skipped)} duplicate(s). Check 'Re-analyze duplicates' to force.")

            if not paths_to_analyze:
                st.warning("All images were duplicates and skipped.")
            else:
                progress = st.progress(0, text="Sending to Qwen-VL...")

                if len(paths_to_analyze) == 1:
                    results = [analyze_retail_image(paths_to_analyze[0], demo_index=0)]
                else:
                    # ── FIX: max_workers reduced to 2 ──────────────────────────
                    # The threading lock in vector_store serializes DB writes, so
                    # more than 2 workers only adds thread overhead without speed
                    # benefit during the store phase.
                    results = analyze_multiple_images(paths_to_analyze, max_workers=2)

                progress.progress(60, text="Saving to vector database...")

                all_price_changes = []
                stored  = 0
                errors  = 0

                for path, result in zip(paths_to_analyze, results):
                    if "error" in result:
                        errors += 1
                        st.error(f"❌ Analysis failed for **{os.path.basename(path)}**: {result['error']}")
                        continue

                    # ── FIX: wrap store_analysis in try/except ─────────────────
                    # One image failing to store must not abort the whole batch.
                    try:
                        store_result = store_analysis(
                            path, result, st.session_state.active_session_id
                        )
                    except Exception as e:
                        errors += 1
                        st.error(f"❌ Failed to store **{os.path.basename(path)}**: {e}")
                        continue

                    # store_analysis returns dict: {scan_id, price_changes}
                    if isinstance(store_result, dict):
                        scan_id       = store_result.get("scan_id")
                        price_changes = store_result.get("price_changes", [])
                    else:
                        scan_id       = store_result  # backwards compat
                        price_changes = []

                    all_price_changes.extend(price_changes)
                    stored += 1

                    # ── FIX: guard update_session — only call when scan_id exists
                    if st.session_state.active_session_id and scan_id:
                        try:
                            update_session(
                                st.session_state.active_session_id, path, scan_id
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Session update failed for {os.path.basename(path)}: {e}")

                progress.progress(100, text="Done!")

                # Result summary banner
                if any(r.get("_demo_mode") for r in results if "error" not in r):
                    st.info("🎭 Showing demo data (vLLM not connected).")
                elif errors == 0:
                    st.success(f"✅ {stored}/{len(paths_to_analyze)} image(s) analyzed and stored.")
                else:
                    st.warning(f"⚠️ {stored} stored, {errors} failed out of {len(paths_to_analyze)} image(s).")

                # ── Price change alert (across ALL images in this batch) ──
                if all_price_changes:
                    st.divider()
                    render_price_changes(all_price_changes)

                # ── Per-image results ─────────────────────────────────────
                for i, (path, result) in enumerate(zip(paths_to_analyze, results)):
                    with st.expander(f"📷 {os.path.basename(path)}", expanded=(len(results)==1)):
                        if "error" in result:
                            st.error(result["error"])
                            continue

                        products = result.get("products", [])
                        promos   = result.get("promotions", [])
                        summary  = result.get("shelf_summary", {})
                        statuses = [p.get("inventory_status","unknown") for p in products]

                        c1,c2,c3,c4 = st.columns(4)
                        c1.metric("Products",    len(products))
                        c2.metric("Promotions",  len(promos))
                        c3.metric("✅ Present",  statuses.count("present"))
                        c4.metric("⚠️Low/❌Out", statuses.count("low")+statuses.count("absent"))

                        if result.get("best_value_product"):
                            st.info(f"🏆 Best Value: **{result['best_value_product']}**")

                        cats = summary.get("categories_present", [])
                        if cats:
                            st.caption("Categories: " + " · ".join(f"`{c}`" for c in cats))

                        st.subheader("🛍️ Products & Inventory")
                        render_products_table(products)

                        if promos:
                            st.subheader("🏷️ Promotions")
                            st.dataframe(pd.DataFrame(promos), use_container_width=True, hide_index=True)

                        with st.expander("Raw JSON"):
                            st.json(result)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Product Chatbot":
    st.title("💬 Retail Product Assistant")

    sid = st.session_state.active_session_id
    if sid and not st.session_state.chat_history:
        st.session_state.chat_history = load_chat_history(sid)

    if DEMO_MODE:
        st.markdown('<div class="demo-banner">🎭 Demo Mode — chatbot uses pre-loaded retail knowledge.</div>', unsafe_allow_html=True)

    if sid:
        st.info(f"🗂️ Session: **{st.session_state.active_session_label}** — chat is auto-saved.")
    else:
        st.warning("💡 Start a session from the sidebar to persist chat history.")

    st.subheader("💡 Quick Questions")
    suggestions = get_suggested_questions()
    cols = st.columns(4)
    for i, q in enumerate(suggestions[:8]):
        with cols[i % 4]:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                if sid: save_chat_message(sid, "user", q)
                with st.spinner("🔍 Searching..."):
                    answer = chat_with_retail_ai(q, st.session_state.chat_history[:-1])
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                if sid: save_chat_message(sid, "assistant", answer)
                st.rerun()

    st.divider()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🛒" if msg["role"]=="assistant" else None):
            st.write(msg["content"])

    if not st.session_state.chat_history:
        st.info("👋 Ask anything about products, stock levels, prices, or promotions!")

    user_input = st.chat_input("Ask about products, stock, prices, deals…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        if sid: save_chat_message(sid, "user", user_input)
        with st.spinner("🔍 Searching product database..."):
            answer = chat_with_retail_ai(user_input, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if sid: save_chat_message(sid, "assistant", answer)
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INVENTORY STATUS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Inventory Status":
    st.title("📦 Inventory Status Dashboard")
    st.caption("Stock levels extracted from shelf images via computer vision.")

    inv_summary = get_inventory_summary()
    inv_table   = get_inventory_table()

    counts = inv_summary.get("status_counts", {})
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("✅ In Stock",     counts.get("present", 0))
    c2.metric("⚠️ Low Stock",   counts.get("low", 0))
    c3.metric("❌ Out of Stock", counts.get("absent", 0))
    c4.metric("❓ Unknown",      counts.get("unknown", 0))

    low_items = inv_summary.get("low_stock_items", [])
    if low_items:
        st.divider()
        with st.expander(f"🚨 {len(low_items)} Items Need Attention", expanded=True):
            df_low = pd.DataFrame(low_items)
            df_low["Status"] = df_low["inventory_status"].map(status_badge)
            st.dataframe(df_low[["product_name","brand","category","quantity","Status","price","timestamp"]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📋 Full Inventory Table")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        status_filter = st.multiselect("Filter by status", ["present","low","absent","unknown"],
            default=["present","low","absent","unknown"], format_func=status_badge)
    with col_f2:
        search_term = st.text_input("🔍 Search product/brand", "")

    if inv_table:
        df_inv = pd.DataFrame(inv_table)
        df_inv["Status"] = df_inv["inventory_status"].map(status_badge)
        if status_filter:
            df_inv = df_inv[df_inv["inventory_status"].isin(status_filter)]
        if search_term:
            mask = (df_inv["product_name"].str.contains(search_term,case=False,na=False) |
                    df_inv["brand"].str.contains(search_term,case=False,na=False))
            df_inv = df_inv[mask]
        cols_ = [c for c in ["product_name","brand","category","quantity","Status","price","discount_pct","last_seen"] if c in df_inv.columns]
        st.dataframe(df_inv[cols_], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export CSV", df_inv.to_csv(index=False), "inventory.csv", "text/csv")
    else:
        st.info("No inventory data yet. Analyze some shelf images first!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRICE CHANGES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Price Changes":
    st.title("💰 Price Change Tracker")
    st.caption("Automatically detected price movements across all your shelf scans.")

    changes = get_price_change_history()

    if not changes:
        st.info("No price changes detected yet. You need at least **2 scans of the same shelf** to detect changes.")
        st.markdown("""
        **How it works:**
        1. Scan a shelf → prices stored in DB
        2. Scan the same shelf again later
        3. System auto-detects which products changed price
        4. Changes appear here instantly
        """)
    else:
        increases = [c for c in changes if c.get("direction")=="increased"]
        decreases = [c for c in changes if c.get("direction")=="decreased"]

        c1,c2,c3 = st.columns(3)
        c1.metric("Total Changes",  len(changes))
        c2.metric("🔺 Increases",   len(increases))
        c3.metric("🔻 Decreases",   len(decreases))

        st.divider()

        if increases:
            st.subheader("🔺 Price Increases")
            rows = [{"Product": c["product_name"], "Brand": c.get("brand",""),
                     "Prev": c["prev_price"], "Now": c["curr_price"],
                     #"Change": f"+{c['delta']:.2f} (+{c.get('delta_pct',0):.1f}%)",
                     "Change": f"+{c['delta']:.2f} (+{c.get('delta_pct',0):.1f}%)",
                     "Prev Scan": str(c.get("prev_date",""))[:10],
                     "Curr Scan": str(c.get("curr_date",""))[:10]} for c in increases]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if decreases:
            st.subheader("🔻 Price Decreases / Discounts Applied")
            rows = [{"Product": c["product_name"], "Brand": c.get("brand",""),
                     "Prev": c["prev_price"], "Now": c["curr_price"],
                     #"Change": f"{c['delta']:.2f} ({c.get('delta_pct',0):.1f}%)",
                     "Change": f"+{(c.get('delta') or 0):.2f} (+{(c.get('delta_pct') or 0):.1f}%)",
                  
                     "Prev Scan": str(c.get("prev_date",""))[:10],
                     "Curr Scan": str(c.get("curr_date",""))[:10]} for c in decreases]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.download_button("⬇️ Export Price Change Log", pd.DataFrame(changes).to_csv(index=False), "price_changes.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🕒 History":
    st.title("🕒 Scan History")
    history = get_all_history()

    if not history:
        st.info("No scans yet. Go to **Analyze Images** to get started!")
    else:
        st.metric("Total Scans", len(history))
        st.divider()
        for item in history:
            ts    = item["timestamp"][:19].replace("T"," ")
            label = os.path.basename(item["image_path"])
            with st.expander(f"📷 {label}  —  {ts}"):
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Products",   item["product_count"])
                c2.metric("Promotions", item["promotion_count"])
                c3.metric("✅ Present", item.get("present_count",0))
                c4.metric("⚠️ Low",    item.get("low_count",0))
                c5.metric("❌ Absent", item.get("absent_count",0))

                pchanges = item.get("price_changes",[])
                if pchanges:
                    render_price_changes(pchanges)

                if os.path.exists(item["image_path"]):
                    st.image(item["image_path"], width=280)

                analysis = item["analysis"]
                if analysis.get("products"):
                    st.subheader("Products")
                    render_products_table(analysis["products"])
                if analysis.get("promotions"):
                    st.subheader("Promotions")
                    st.dataframe(pd.DataFrame(analysis["promotions"]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("📊 Retail Analytics Dashboard")
    history = get_all_history()

    if not history:
        st.info("No data yet. Analyze some images first!")
    else:
        all_products   = [p for item in history for p in item["analysis"].get("products",[])]
        all_promotions = [p for item in history for p in item["analysis"].get("promotions",[])]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Scans",      len(history))
        c2.metric("Total Products",   len(all_products))
        c3.metric("Total Promotions", len(all_promotions))
        c4.metric("Avg Prods/Scan",   round(len(all_products)/len(history),1) if history else 0)

        st.divider()

        if all_products:
            df = pd.DataFrame(all_products)
            col_a, col_b = st.columns(2)

            with col_a:
                if "category" in df.columns:
                    st.subheader("📦 By Category")
                    cc = df["category"].value_counts().reset_index()
                    cc.columns = ["Category","Count"]
                    st.bar_chart(cc.set_index("Category"))

            with col_b:
                if "inventory_status" in df.columns:
                    st.subheader("📊 Inventory Distribution")
                    ic = df["inventory_status"].value_counts().reset_index()
                    ic.columns = ["Status","Count"]
                    ic["Status"] = ic["Status"].map(status_badge)
                    st.bar_chart(ic.set_index("Status"))

            if "brand" in df.columns:
                st.subheader("🏷️ Top Brands")
                bc = df["brand"].dropna().value_counts().head(10).reset_index()
                bc.columns = ["Brand","Count"]
                st.bar_chart(bc.set_index("Brand"))

            if all_promotions:
                st.subheader("🎯 Promotion Types")
                df_pr = pd.DataFrame(all_promotions)
                if "type" in df_pr.columns:
                    pt = df_pr["type"].value_counts().reset_index()
                    pt.columns = ["Type","Count"]
                    st.bar_chart(pt.set_index("Type"))

            st.subheader("📋 All Products")
            render_products_table(all_products)


elif page == "📋 Planogram Compliance":
    st.title("📋 Planogram Compliance Checker")
    st.caption("Checks whether products promoted in flyers are actually stocked on the shelf.")

    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown("""
        1. Upload and analyze a **flyer image** — promotions are extracted
        2. Upload and analyze a **shelf image** — products are extracted  
        3. This page cross-references them: every promoted product is checked
           against the shelf using fuzzy name matching
        4. Missing products are flagged as **compliance gaps**
        """)

    threshold = st.slider(
        "Match confidence threshold", 0.5, 1.0, 0.75, 0.05,
        help="Lower = more lenient matching. 0.75 recommended."
    )

    if st.button("🔍 Run Compliance Check", type="primary"):
        with st.spinner("Cross-referencing flyers vs shelf scans..."):
            report = check_planogram_compliance(name_threshold=threshold)

        if not report:
            st.info("No data yet. You need at least one **flyer scan** and one **shelf scan** to run this check.")
        else:
            missing  = [r for r in report if not r["found_on_shelf"]]
            present  = [r for r in report if r["found_on_shelf"]]

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Promoted Products", len(report))
            c2.metric("✅ Found on Shelf",        len(present))
            c3.metric("❌ Missing from Shelf",    len(missing))

            compliance_pct = round(len(present) / len(report) * 100) if report else 0
            color = "green" if compliance_pct >= 80 else "orange" if compliance_pct >= 50 else "red"
            st.markdown(
                f"### Compliance Score: "
                f"<span style='color:{color};font-size:28px'>{compliance_pct}%</span>",
                unsafe_allow_html=True
            )

            if missing:
                st.divider()
                st.subheader("❌ Compliance Gaps — Promoted but NOT on Shelf")
                st.caption("These products are advertised in your flyer but not found in any shelf scan.")
                df_miss = pd.DataFrame(missing)
                df_miss["match_confidence"] = df_miss["match_confidence"].apply(lambda x: f"{x:.0%}")
                st.dataframe(
                    df_miss[["promoted_product","promotion","savings","validity","match_confidence","flyer_date"]],
                    use_container_width=True, hide_index=True
                )
                st.error(f"⚠️ Action needed: Stock {len(missing)} missing product(s) before the promotion period ends.")

            if present:
                st.divider()
                st.subheader("✅ Compliant — Promoted and Found on Shelf")
                df_pres = pd.DataFrame(present)
                df_pres["match_confidence"] = df_pres["match_confidence"].apply(lambda x: f"{x:.0%}")
                st.dataframe(
                    df_pres[["promoted_product","promotion","savings","match_confidence","flyer_date"]],
                    use_container_width=True, hide_index=True
                )

            # Export
            df_full = pd.DataFrame(report)
            st.download_button(
                "⬇️ Export Compliance Report CSV",
                df_full.to_csv(index=False),
                "planogram_compliance.csv",
                "text/csv"
            )
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SESSIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗂️ Sessions":
    st.title("🗂️ Session Manager")
    sessions = get_all_sessions()

    if not sessions:
        st.info("No sessions yet. Start one from the sidebar!")
    else:
        for s in sessions:
            image_paths = json.loads(s.get("image_paths","[]"))
            with st.expander(f"🗂️ {s['label']}  —  {s['created_at'][:19].replace('T',' ')}"):
                st.caption(f"ID: `{s['session_id']}`")
                st.metric("Images in session", len(image_paths))
                for p in image_paths:
                    st.caption(f"• {os.path.basename(p)}")
                chat = load_chat_history(s["session_id"])
                if chat:
                    st.write(f"**{len(chat)} chat messages**")
                    with st.expander("View transcript"):
                        for msg in chat:
                            icon = "🧑" if msg["role"]=="user" else "🛒"
                            st.write(f"{icon} **{msg['role'].title()}:** {msg['content']}")
                if st.button("▶ Restore", key=f"restore_{s['session_id']}"):
                    st.session_state.active_session_id    = s["session_id"]
                    st.session_state.active_session_label = s["label"]
                    st.session_state.chat_history = load_chat_history(s["session_id"])
                    st.rerun()