import os
from openai import OpenAI
from dotenv import load_dotenv

from .prompt_templates import BASE_SYSTEM_PROMPT, DECISION_TEMPLATE

# Load environment variables
load_dotenv()


def has_api_key() -> bool:
	return bool(os.getenv("GROQ_API_KEY"))


def generate_decision(context: str, model: str = "llama-3.3-70b-versatile") -> str:
	api_key = os.getenv("GROQ_API_KEY")
	if not api_key:
		raise RuntimeError("GROQ_API_KEY not set. Provide an API key to call the LLM.")

	client = OpenAI(
		api_key=api_key,
		base_url="https://api.groq.com/openai/v1"
	)
	prompt = DECISION_TEMPLATE.format(context=context)
	resp = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": BASE_SYSTEM_PROMPT},
			{"role": "user", "content": prompt},
		],
		temperature=0.2,
		max_tokens=500,
	)
	return resp.choices[0].message.content.strip()
