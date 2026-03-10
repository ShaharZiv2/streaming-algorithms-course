"""
llm_evaluator.py – TinyLlama-based relevance judge for RAG evaluation.

Uses TinyLlama-1.1B-Chat (GGUF Q4_K_M) via llama-cpp-python to evaluate
whether a retrieved passage actually answers a query.

Two evaluation modes:
  1. passage_is_relevant(query, passage) → bool
     Asks TinyLlama: "Does this passage answer the question?"
     Used to measure LLM-as-judge recall vs qrels-based recall.

  2. answer_from_context(query, passages) → str
     Asks TinyLlama to produce an answer grounded in retrieved passages.
     Used for end-to-end RAG quality assessment.

  3. evaluate_rag_results(query, passages, ground_truth_answer) → dict
     Runs both judge + answer generation and scores faithfulness/grounding.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_DIR = Path("models")
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_URL = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/"
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

# TinyLlama chat template tokens
SYS_START = "<|system|>"
SYS_END = "</s>"
USER_START = "<|user|>"
ASST_START = "<|assistant|>"


def _download_model() -> None:
    """Download TinyLlama GGUF if not already present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"[LLMEvaluator] Model already at {MODEL_PATH}")
        return
    print(f"[LLMEvaluator] Downloading TinyLlama GGUF (~670 MB) → {MODEL_PATH}")
    print("  (this only happens once)")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        print(f"\r  {min(pct, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
    print(f"\n[LLMEvaluator] Download complete.")


def _build_prompt(system: str, user: str) -> str:
    """Format a TinyLlama chat prompt."""
    return (
        f"{SYS_START}\n{system}{SYS_END}\n"
        f"{USER_START}\n{user}{SYS_END}\n"
        f"{ASST_START}\n"
    )


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class TinyLlamaEvaluator:
    """Thin wrapper around llama-cpp-python for RAG evaluation.

    Parameters
    ----------
    n_ctx       : context window size (default 2048 – enough for query+passage)
    n_gpu_layers: layers to offload to Metal/GPU on Apple Silicon (default 1)
    verbose     : whether llama.cpp prints loading info
    """

    def __init__(
        self,
        n_ctx: int = 2048,
        n_gpu_layers: int = 1,
        verbose: bool = False,
    ):
        from llama_cpp import Llama

        _download_model()
        print("[LLMEvaluator] Loading TinyLlama…")
        self._llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        print("[LLMEvaluator] Ready.")

    # ------------------------------------------------------------------
    # 1. Binary relevance judge
    # ------------------------------------------------------------------

    def passage_is_relevant(self, query: str, passage: str, max_passage_chars: int = 600) -> bool:
        """Ask TinyLlama whether the passage answers the query.

        Returns True if the model says YES, False otherwise.
        """
        passage_snippet = passage[:max_passage_chars]
        system = (
            "You are a strict relevance judge. "
            "Answer only YES or NO – nothing else."
        )
        user = (
            f"Question: {query}\n\n"
            f"Passage: {passage_snippet}\n\n"
            "Does this passage directly answer or address the question? "
            "Reply with a single word: YES or NO."
        )
        prompt = _build_prompt(system, user)
        output = self._llm(
            prompt,
            max_tokens=4,
            temperature=0.0,
            stop=["\n", ".", " "],
        )
        answer = output["choices"][0]["text"].strip().upper()
        return answer.startswith("YES")

    # ------------------------------------------------------------------
    # 2. Answer generation
    # ------------------------------------------------------------------

    def answer_from_context(
        self,
        query: str,
        passages: list[str],
        max_passage_chars: int = 400,
        max_new_tokens: int = 150,
    ) -> str:
        """Generate an answer grounded in the retrieved passages."""
        context = "\n\n".join(
            f"[{i+1}] {p[:max_passage_chars]}" for i, p in enumerate(passages)
        )
        system = (
            "You are a helpful assistant. "
            "Answer the question using ONLY the provided passages. "
            "If the passages don't contain the answer, say 'I don't know'."
        )
        user = f"Passages:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        prompt = _build_prompt(system, user)
        output = self._llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.1,
            stop=["</s>", SYS_START, USER_START],
        )
        return output["choices"][0]["text"].strip()

    # ------------------------------------------------------------------
    # 3. Full RAG evaluation for one query
    # ------------------------------------------------------------------

    def evaluate_rag(
        self,
        query: str,
        retrieved_passages: list[tuple[str, str]],  # list of (doc_id, text)
        top_k_for_answer: int = 3,
    ) -> dict:
        """Run both relevance judging and answer generation for one query.

        Parameters
        ----------
        query               : the query string
        retrieved_passages  : list of (doc_id, passage_text) ordered by rank
        top_k_for_answer    : how many top passages to pass to the answer LLM

        Returns
        -------
        dict with keys:
          llm_relevant_count  : # passages judged relevant by TinyLlama
          llm_relevant_frac   : fraction of passages judged relevant
          llm_relevant_ids    : list of doc_ids TinyLlama found relevant
          generated_answer    : the LLM-generated answer
          grounded            : True if answer contains ≥1 passage snippet
        """
        llm_relevant_ids = []
        for doc_id, text in retrieved_passages:
            if self.passage_is_relevant(query, text):
                llm_relevant_ids.append(doc_id)

        top_texts = [text for _, text in retrieved_passages[:top_k_for_answer]]
        generated_answer = self.answer_from_context(query, top_texts) if top_texts else ""

        # Rough grounding check: does answer overlap with any passage?
        grounded = False
        answer_lower = generated_answer.lower()
        for _, text in retrieved_passages[:top_k_for_answer]:
            # Check for 5-word n-gram overlap
            words = text.lower().split()
            for j in range(len(words) - 4):
                ngram = " ".join(words[j:j+5])
                if ngram in answer_lower:
                    grounded = True
                    break
            if grounded:
                break

        return {
            "llm_relevant_count": len(llm_relevant_ids),
            "llm_relevant_frac": len(llm_relevant_ids) / max(len(retrieved_passages), 1),
            "llm_relevant_ids": llm_relevant_ids,
            "generated_answer": generated_answer,
            "grounded": grounded,
        }


