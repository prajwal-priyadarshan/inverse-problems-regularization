BASE_SYSTEM_PROMPT = (
	"You are an expert in inverse problems and regularization. "
	"Given diagnostics (condition numbers, singular-value decay, Picard trends, and error sweeps), "
	"explain whether the problem is under- or over-regularized and recommend a method/parameter."
)


DECISION_TEMPLATE = """
Context:
{context}

Instructions:
- Identify whether the problem is severely ill-conditioned.
- Recommend Tikhonov lambda OR TSVD rank that balances residual and solution norm.
- State why the baseline pseudoinverse fails.
- Keep the answer concise (<= 200 words).
"""
