import os

import ollama
from dotenv import load_dotenv

load_dotenv()


class LLMGenerator:

    def __init__(self):

        self.model = os.getenv(
            "GENERATOR_MODEL",
            "qwen3:1.7b"
        )

    def generate(self, question, contexts):

        if not contexts:
            return {
                "answer": (
                    "I don't have enough relevant "
                    "information in the provided "
                    "documentation to answer this question."
                ),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }

        context_parts = []

        for i, item in enumerate(contexts, start=1):

            context_parts.append(
                f"[{i}] Source: {item['source']} "
                f"(chunk {item['chunk_index']})\n"
                f"{item['text']}"
            )

        context = "\n\n".join(context_parts)

        system_prompt = """
You are a grounded question-answering assistant.

Answer the question ONLY using the provided documentation.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the documentation does not contain enough
   information, say so clearly.
4. Keep the answer concise.
5. Cite factual statements using [1], [2], etc.
6. Only cite sources that support the statement.
"""

        user_prompt = f"""
Documentation:

{context}

Question:
{question}

Answer using only the documentation above.
Include citations such as [1] or [2].
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            options={
                "temperature": 0
            }
        )

        answer = response["message"]["content"]

        input_tokens = response.get(
            "prompt_eval_count",
            0
        )

        output_tokens = response.get(
            "eval_count",
            0
        )

        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                input_tokens + output_tokens
            )
        }