from openai import OpenAI


class LocalLLMClient:
    def __init__(self, base_url="http://localhost:1234/v1", model="qwen2.5-3b-instruct", timeout=60):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)

    def answer(self, question, context):
        context = context.strip() if context else ""
        user_content = (
            f"Document context:\n\n{context}\n\nQuestion: {question}"
            if context
            else f"Question: {question}\n\n(No relevant document context was retrieved.)"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful offline document assistant. Answer using the supplied context. "
                        "Do not invent information. If context is insufficient, say so clearly."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        return content if content else "The model returned an empty response. Try rephrasing your question."
