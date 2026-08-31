import time
from typing import Dict, Any
from app.config import settings

class ObservabilityTracer:
    """Langfuse Observability & Execution Logger."""
    
    def __init__(self):
        self.enabled = bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)
        self.langfuse = None
        if self.enabled:
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                print("[Langfuse] Observability tracing enabled.")
            except Exception as e:
                print(f"[Langfuse Warning] Failed to initialize Langfuse client: {e}")
                self.enabled = False

    def log_execution(self, query: str, state: Dict[str, Any], start_time: float):
        latency_ms = round((time.time() - start_time) * 1000, 2)
        route = state.get("route", "unknown")
        steps = state.get("execution_steps", [])
        answer_snippet = (state.get("final_answer") or "")[:100]

        log_summary = {
            "query": query,
            "route_selected": route,
            "latency_ms": latency_ms,
            "execution_steps_count": len(steps),
            "citations_count": len(state.get("citations", [])),
            "answer_preview": answer_snippet
        }

        print(f"[Observability Trace] Query Execution Summary: {log_summary}")

        if self.enabled and self.langfuse:
            try:
                trace = self.langfuse.trace(
                    name="OmniBrain Query Execution",
                    input=query,
                    output=state.get("final_answer"),
                    metadata=log_summary
                )
                trace.event(name="Agent Steps Completed", metadata={"steps": steps})
            except Exception as e:
                print(f"[Observability Error] Failed to log trace to Langfuse: {e}")

tracer = ObservabilityTracer()
