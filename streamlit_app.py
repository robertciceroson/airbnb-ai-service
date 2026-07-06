"""
Airbnb AI Service — Streamlit front-end
Two tabs:
  Tab 1 — Price Predictor (existing XGBoost app, now calling FastAPI /predict)
  Tab 2 — Support Chat   (LangGraph agent via FastAPI /chat)

Run the FastAPI backend first:
    uvicorn app.main:app --reload --port 8000

Then launch this app:
    streamlit run streamlit_app.py
"""
import os
import streamlit as st
import datetime
import uuid
import httpx

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb AI Service",
    page_icon="🏙️",
    layout="wide",
)

# API_BASE: reads from env var on Streamlit Cloud, falls back to localhost for local dev.
# Set this in Streamlit Cloud → App Settings → Secrets as:
#   API_BASE_URL = "https://your-service.onrender.com"
API_BASE = os.getenv("API_BASE_URL", st.secrets.get("API_BASE_URL", "http://localhost:8000")
                     if hasattr(st, "secrets") else "http://localhost:8000")

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; max-width: 1200px; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
    div[data-testid="stSlider"] { padding-top: 0; padding-bottom: 0.1rem; }
    h3 { margin-top: 0.3rem !important; margin-bottom: 0.2rem !important; }
    hr { margin: 0.5rem 0 !important; }
    div[data-testid="stMetric"] { padding: 0.3rem 0; }

    /* Green date input */
    div[data-testid="stDateInput"] input {
        background-color: #D4EDDA !important; border-color: #28A745 !important;
        color: #155724 !important; font-weight: 600 !important; }
    div[data-testid="stDateInput"] label {
        color: #155724 !important; font-weight: 600 !important; }

    /* Green Predict button */
    div[data-testid="stButton"] button {
        background-color: #D4EDDA !important; border: 1px solid #28A745 !important;
        color: #155724 !important; font-weight: 600 !important; }
    div[data-testid="stButton"] button:hover {
        background-color: #28A745 !important; color: #FFFFFF !important; }

    /* Chat bubbles */
    .chat-user {
        background: #D4EDDA; border: 1px solid #28A745; border-radius: 12px 12px 2px 12px;
        padding: 10px 14px; margin: 6px 0; max-width: 80%; margin-left: auto;
        color: #155724; font-size: 0.9rem; }
    .chat-agent {
        background: #EBF3FB; border: 1px solid #2E75B6; border-radius: 12px 12px 12px 2px;
        padding: 10px 14px; margin: 6px 0; max-width: 85%;
        color: #1A1A1A; font-size: 0.9rem; }
    .tool-badge {
        font-size: 0.72rem; color: #555; margin-top: 4px; font-style: italic; }
    .chat-container {
        height: 440px; overflow-y: auto; padding: 8px 4px;
        border: 1px solid #ddd; border-radius: 8px; background: #FAFAFA;
        margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🏙️ Airbnb AI Service</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#555;'>NYC Price Prediction + AI-Powered Customer Support</p>",
    unsafe_allow_html=True
)
st.divider()

# ── Seasonal constants ────────────────────────────────────────────────────────
SEASONAL = {
    1: 0.88, 2: 0.90, 3: 0.97,  4: 1.05,
    5: 1.08, 6: 1.18, 7: 1.22,  8: 1.20,
    9: 1.07, 10: 1.05, 11: 0.95, 12: 1.02,
}
SEASON_LABEL = {
    1: "❄️ Off-season",    2: "❄️ Off-season",
    3: "🌸 Shoulder",      4: "🌸 Spring",        5: "🌸 Spring peak",
    6: "☀️ Summer peak",   7: "☀️ Summer peak",   8: "☀️ Summer peak",
    9: "🍂 Fall shoulder", 10: "🍂 Fall shoulder",
    11: "❄️ Off-season",  12: "🎄 Holiday",
}

# ── API helpers ───────────────────────────────────────────────────────────────

def api_predict(payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}/predict", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot reach the FastAPI backend. Run: `uvicorn app.main:app --reload --port 8000`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_chat(message: str, conversation_id: str, history: list) -> dict | None:
    payload = {
        "message": message,
        "conversation_id": conversation_id,
        "history": history,
    }
    try:
        r = httpx.post(f"{API_BASE}/chat", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        st.error("⚠️ Cannot reach the FastAPI backend. Run: `uvicorn app.main:app --reload --port 8000`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💰 Price Predictor", "💬 Support Chat"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Price Predictor
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    # Borough/neighbourhood options (representative — full list served by API)
    BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    NEIGHBOURHOODS = {
        "Manhattan": ["Battery Park City", "Chelsea", "Chinatown", "East Harlem",
                      "East Village", "Financial District", "Harlem", "Hell's Kitchen",
                      "Inwood", "Kips Bay", "Little Italy", "Lower East Side",
                      "Midtown", "Morningside Heights", "Murray Hill", "NoHo",
                      "Nolita", "Roosevelt Island", "SoHo", "Theater District",
                      "Tribeca", "Two Bridges", "Upper East Side", "Upper West Side",
                      "Washington Heights", "West Village"],
        "Brooklyn": ["Bedford-Stuyvesant", "Boerum Hill", "Borough Park", "Brighton Beach",
                     "Brooklyn Heights", "Bushwick", "Carroll Gardens", "Clinton Hill",
                     "Cobble Hill", "Crown Heights", "Downtown Brooklyn", "DUMBO",
                     "East Flatbush", "Flatbush", "Fort Greene", "Gowanus",
                     "Greenpoint", "Park Slope", "Prospect Heights", "Red Hook",
                     "Sunset Park", "Williamsburg", "Windsor Terrace"],
        "Queens": ["Astoria", "Elmhurst", "Flushing", "Forest Hills", "Jackson Heights",
                   "Jamaica", "Long Island City", "Ridgewood", "Rockaway Beach",
                   "Sunnyside", "Woodside"],
        "Bronx": ["Baychester", "Belmont", "Fordham", "Kingsbridge", "Mott Haven",
                  "Riverdale", "South Bronx", "Tremont"],
        "Staten Island": ["Arden Heights", "Charleston", "Dongan Hills", "Great Kills",
                          "New Brighton", "Port Richmond", "St. George", "Tompkinsville"],
    }
    ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]

    col1, col2, col3 = st.columns([1.1, 1.2, 1.0])

    with col1:
        st.markdown("**📋 Listing Details**")
        borough = st.selectbox("Borough", BOROUGHS, index=2)
        neighbourhood = st.selectbox("Neighbourhood", NEIGHBOURHOODS.get(borough, ["—"]))
        room_type = st.selectbox("Room Type", ROOM_TYPES)

    with col2:
        st.markdown("**⚙️ Listing Parameters**")
        minimum_nights    = st.slider("Minimum nights required", 1, 30, 2)
        availability      = st.slider("Availability (days / year)", 0, 365, 200)
        number_of_reviews = st.slider("Number of reviews", 0, 300, 20)
        reviews_per_month = st.slider("Reviews per month", 0.0, 10.0, 1.0, step=0.1)
        host_listings     = st.slider("Host total listings", 1, 50, 1)

    with col3:
        st.markdown("**📅 Stay Dates**")
        today      = datetime.date.today()
        date_range = st.date_input(
            "Check-in → Check-out",
            value=(today, today + datetime.timedelta(days=minimum_nights)),
            min_value=datetime.date(2024, 1, 1),
            max_value=datetime.date(2027, 12, 31),
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            checkin_date, checkout_date = date_range
            num_nights = max((checkout_date - checkin_date).days, 1)
        else:
            checkin_date  = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]
            checkout_date = checkin_date + datetime.timedelta(days=minimum_nights)
            num_nights    = minimum_nights

        month         = checkin_date.month
        seasonal_mult = SEASONAL[month]
        season_label  = SEASON_LABEL[month]
        adj_pct       = (seasonal_mult - 1) * 100
        adj_str       = f"+{adj_pct:.0f}%" if adj_pct >= 0 else f"{adj_pct:.0f}%"

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Check-in",  checkin_date.strftime("%b %d, %Y"))
        dc2.metric("Check-out", checkout_date.strftime("%b %d, %Y"))
        dc3.metric("Nights",    num_nights)
        st.caption(f"{season_label} · Seasonal adjustment: **{adj_str}** ({checkin_date.strftime('%B')} in NYC)")
        st.markdown("")
        predict_clicked = st.button("🔍 Predict Price", use_container_width=True)

    st.divider()

    if predict_clicked:
        payload = {
            "borough": borough,
            "neighbourhood": neighbourhood,
            "room_type": room_type,
            "minimum_nights": minimum_nights,
            "availability_365": availability,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "calculated_host_listings_count": host_listings,
            "checkin_month": month,
        }
        result = api_predict(payload)

        if result:
            adj_price  = result["adjusted_price"]
            base_price = result["base_price"]
            total_cost = adj_price * num_nights

            st.markdown(
                f"""<div style="background:#D4EDDA; border:1px solid #28A745; border-radius:6px;
                padding:4px 14px; display:inline-block;">
                <span style="font-size:0.85rem; font-weight:600; color:#155724;">
                Estimated nightly price: ${adj_price:.0f} &nbsp;·&nbsp;
                Total for {num_nights} night{'s' if num_nights != 1 else ''}: ${total_cost:,.0f}
                </span></div>""",
                unsafe_allow_html=True,
            )

            ra, rb, rc = st.columns(3)
            ra.metric("Base nightly price", f"${base_price:.0f}")
            rb.metric(f"Adjusted ({checkin_date.strftime('%B')})", f"${adj_price:.0f}",
                      delta=f"{adj_str} seasonal")
            rc.metric(f"Total ({num_nights} nights)", f"${total_cost:,.0f}")

            if result.get("neighbourhood_median"):
                median = result["neighbourhood_median"]
                diff   = adj_price - median
                direction = "above" if diff > 0 else "below"
                st.markdown(
                    f"📍 Median price in **{neighbourhood}**: **${median:.0f}/night** — "
                    f"your estimate is **${abs(diff):.0f} {direction}** the neighbourhood median."
                )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Support Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "<p style='color:#555; font-size:0.9rem;'>Ask anything about Airbnb — "
        "pricing, cancellations, refunds, check-in, disputes, or policies. "
        "The agent will pick the right tool automatically.</p>",
        unsafe_allow_html=True,
    )

    # ── Session state ─────────────────────────────────────────────────────────
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # [{"role": "user"|"assistant", "content": ..., "tool": ...}]

    TOOL_ICONS = {
        "price_lookup":   "💰 Price tool",
        "policy_search":  "📄 Policy search",
        "human_handoff":  "👤 Human handoff",
        "":               "",
    }

    # ── Chat display ──────────────────────────────────────────────────────────
    chat_html = '<div class="chat-container" id="chat-box">'
    if not st.session_state.chat_history:
        chat_html += (
            '<div style="color:#999; font-size:0.85rem; text-align:center; padding-top:160px;">'
            'Start a conversation — try asking about cancellation policies or NYC prices.'
            '</div>'
        )
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_html += f'<div class="chat-user">{msg["content"]}</div>'
        else:
            tool_label = TOOL_ICONS.get(msg.get("tool", ""), "")
            tool_html  = f'<div class="tool-badge">{tool_label}</div>' if tool_label else ""
            chat_html += f'<div class="chat-agent">{msg["content"]}{tool_html}</div>'
    chat_html += '</div>'

    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Input row ─────────────────────────────────────────────────────────────
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_input = st.text_input(
            "Your message",
            placeholder="e.g. What's the cancellation policy if I cancel 2 days before check-in?",
            label_visibility="collapsed",
            key="chat_input",
        )
    with btn_col:
        send = st.button("Send ➤", use_container_width=True)

    # Suggested prompts
    st.markdown(
        "<div style='font-size:0.78rem; color:#777; margin-top:2px;'>"
        "💡 Try: &nbsp;"
        "<em>How much for a private room in Williamsburg in August?</em> &nbsp;·&nbsp; "
        "<em>What's the refund if my host cancels?</em> &nbsp;·&nbsp; "
        "<em>How do I dispute a damage charge?</em>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Send logic ────────────────────────────────────────────────────────────
    if send and user_input.strip():
        # Build history for API (exclude tool metadata)
        api_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history
        ]

        # Append user message to display history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
        })

        # Call API
        with st.spinner("Agent thinking…"):
            response = api_chat(
                message=user_input.strip(),
                conversation_id=st.session_state.conversation_id,
                history=api_history,
            )

        if response:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["reply"],
                "tool": response.get("tool_used", ""),
            })

        st.rerun()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, _ = st.columns([1, 1, 4])
    with ctrl_col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
    with ctrl_col2:
        conv_id_short = st.session_state.conversation_id[:8]
        st.caption(f"Session: `{conv_id_short}…`")

    st.divider()
    st.caption(
        "Powered by LangGraph · Llama 3.3 70B (Groq) · FAISS RAG · XGBoost · FastAPI  |  "
        "[GitHub](https://github.com/robertciceroson/airbnb-ai-service)"
    )
