from typing import Tuple
from app.agents.llm import llm_client

class GuardrailManager:
    """Manages financial domain scope enforcement and response groundedness checks."""

    def check_financial_scope(self, query: str) -> Tuple[bool, str]:
        """Checks whether the input query is within the corporate financial / stock domain."""
        prompt = f"""You are a strict financial domain guardrail.
Check if the user request pertains to corporate finance, stock market data, NVIDIA (NVDA), earnings reports, revenue, operating performance, or financial documents.

User Query: "{query}"

Respond ONLY with 'IN_SCOPE' if financial/corporate/stock related, or 'OUT_OF_SCOPE' if unrelated (e.g., sports, cooking recipes, creative poetry, general trivia):"""

        res = llm_client.generate_text(prompt).strip().upper()

        if "OUT_OF_SCOPE" in res or any(w in query.lower() for w in ["recipe", "bake", "cake", "poem", "football"]):
            return False, "This system is strictly configured to answer questions regarding NVIDIA (NVDA) corporate financial documents, quarterly reports, and historical stock database analytics. Your request falls outside this domain."
        
        return True, ""

    def validate_groundedness(self, answer: str, context_str: str) -> bool:
        """Validates that the generated response is supported by retrieved context evidence."""
        if not context_str.strip():
            return True

        prompt = f"""You are a Groundedness Verifier.
Verify whether the claims in the generated AI response are grounded in the supporting evidence.

Supporting Evidence:
"{context_str[:1500]}"

Generated Response:
"{answer[:1500]}"

Respond ONLY with 'GROUNDED' or 'UNSUPPORTED':"""

        res = llm_client.generate_text(prompt).strip().upper()
        return "UNSUPPORTED" not in res

guardrail_manager = GuardrailManager()
