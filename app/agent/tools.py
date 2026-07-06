"""
LangChain tools available to the Airbnb AI agent:
  1. price_lookup   — calls the XGBoost predictor
  2. policy_search  — RAG over Airbnb Help Center docs (FAISS)
  3. human_handoff  — escalates when confidence is low
"""
import json
from langchain_core.tools import tool


# ── Tool 1: Price Lookup ──────────────────────────────────────────────────────

def make_price_tool(predictor):
    """Factory — injects the loaded Predictor instance into the tool closure."""

    @tool
    def price_lookup(
        borough: str,
        neighbourhood: str,
        room_type: str,
        minimum_nights: int = 2,
        checkin_month: int = 6,
        availability_365: int = 200,
        number_of_reviews: int = 20,
        reviews_per_month: float = 1.0,
        calculated_host_listings_count: int = 1,
    ) -> str:
        """
        Predict the nightly Airbnb price for a listing in New York City.
        Use this when the user asks about price, cost, nightly rate, or how much a
        listing would cost for a specific borough, neighbourhood, or room type.

        Args:
            borough: NYC borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island)
            neighbourhood: Specific neighbourhood within the borough
            room_type: One of 'Entire home/apt', 'Private room', 'Shared room'
            minimum_nights: Minimum stay required (default 2)
            checkin_month: Month of check-in as integer 1–12 (default 6 = June)
            availability_365: Days per year the listing is available (default 200)
            number_of_reviews: Number of reviews the listing has (default 20)
            reviews_per_month: Average reviews per month (default 1.0)
            calculated_host_listings_count: How many listings the host has (default 1)
        """
        try:
            result = predictor.predict(
                borough=borough,
                neighbourhood=neighbourhood,
                room_type=room_type,
                minimum_nights=minimum_nights,
                availability_365=availability_365,
                number_of_reviews=number_of_reviews,
                reviews_per_month=reviews_per_month,
                calculated_host_listings_count=calculated_host_listings_count,
                checkin_month=checkin_month,
            )
            return (
                f"Estimated nightly price: ${result['adjusted_price']:.0f} "
                f"(base: ${result['base_price']:.0f}, {result['season_label']}, "
                f"seasonal multiplier: {result['seasonal_multiplier']:.2f}x). "
                + (f"Neighbourhood median: ${result['neighbourhood_median']:.0f}/night."
                   if result['neighbourhood_median'] else "")
            )
        except Exception as e:
            return f"Could not compute price: {str(e)}. Please check borough/neighbourhood spelling."

    return price_lookup


# ── Tool 2: Policy Search (RAG) ───────────────────────────────────────────────

def make_policy_tool(retriever):
    """Factory — injects the loaded FAISS retriever into the tool closure."""

    @tool
    def policy_search(query: str) -> str:
        """
        Search Airbnb's help documentation to answer questions about:
        cancellation policies, refunds, guest/host rules, check-in/check-out
        procedures, disputes, safety, payments, reviews, and account issues.
        Use this for any policy or support-related question.

        Args:
            query: The user's question or topic to search for in Airbnb's help docs.
        """
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant policy information found for that query."

            # Compile retrieved chunks with source metadata
            results = []
            sources = set()
            for doc in docs:
                results.append(doc.page_content.strip())
                src = doc.metadata.get("source", "Airbnb Help Center")
                sources.add(src)

            combined = "\n\n---\n\n".join(results)
            source_list = ", ".join(sources)
            return f"{combined}\n\n[Sources: {source_list}]"
        except Exception as e:
            return f"Policy search error: {str(e)}"

    return policy_search


# ── Tool 3: Human Handoff ─────────────────────────────────────────────────────

@tool
def human_handoff(reason: str) -> str:
    """
    Escalate to a human support agent when:
    - The user is clearly frustrated or upset
    - The question involves account security, payment disputes, or legal issues
    - The answer cannot be found in available documentation
    - The user explicitly asks to speak to a human

    Args:
        reason: Brief explanation of why escalation is needed.
    """
    return (
        f"I want to make sure you get the best help possible. "
        f"For this issue ({reason}), I recommend connecting with Airbnb's support team directly:\n\n"
        f"📞 **Airbnb Support:** airbnb.com/help\n"
        f"💬 **Live Chat:** Available 24/7 via the Help Center\n"
        f"📱 **App:** Tap 'Profile' → 'Help' → 'Contact Us'\n\n"
        f"Is there anything else I can help you with in the meantime?"
    )
