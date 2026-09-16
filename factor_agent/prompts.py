EXTRACT_SYSTEM = """You extract a stock-selection factor from a Chinese research note.
Output ONLY a JSON object with keys:
  name, universe, frequency, neutralization, rebalance, expr, windows, evidence
frequency/rebalance: daily | weekly | monthly
neutralization: none | industry | size | industry_size
expr uses $close/$volume/... and RANK, TS_MEAN, TS_STD, TS_PCTCHANGE, INDUSTRY_NEUTRALIZE, COUNT.
evidence is a list of source citations with chunk_id and exact quote.
Do not invent fields that are not in the note.
"""


def extract_user(report: str) -> str:
    return f"Report:\n{report}\n\nOutput FactorSpec JSON only."


EVOLVE_SYSTEM = """You propose a small mutation of a factor that already passed extraction.
Keep universe, frequency, and neutralization. You may change windows or operators.
Use one tool per assistant turn. Validate the expression before calling evaluate_factor.
The visible tool result uses the search window. A hidden score window determines reward.
"""
