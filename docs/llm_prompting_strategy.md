# LLM Prompting Strategy

- Provide diagnostics: condition number, singular-value decay, Picard trends, L-curve samples, and error sweeps for Tikhonov/TSVD.
- Ask for: suitable method, lambda or rank range, and rationale (under/over regularization cues).
- Environment: set `GROQ_API_KEY` before running `python run_pipeline.py --enable-llm`.
- Model: defaults to `llama-3.3-70b-versatile` (Groq); adjust in `src/llm_reasoning/llm_interface.py` if needed.
- Get free API key at: https://console.groq.com/
