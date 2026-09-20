import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.generation.rag_generator import RAGGenerator


load_dotenv()


class LLMGenerator:

    def __init__(
        self,
        model="openai/gpt-oss-120b",
        top_k=5
    ):
        self.model = model
        self.rag = RAGGenerator(top_k=top_k)

        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError(
                "HF_TOKEN not found. "
                "Make sure it is set in your .env file."
            )

        self.client = InferenceClient(
            provider="auto",
            api_key=token
        )

        print(f"LLM ready: {self.model}")

    def generate(self, question, contract_id):

        # Retrieve relevant clauses
        rag_result = self.rag.prepare(
            question=question,
            contract_id=contract_id
        )

        # Send grounded prompt to the LLM
        response = self.client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": rag_result["prompt"]
                }
            ],
            model=self.model,
            max_tokens=300,
            temperature=0.1
        )

        answer = response.choices[0].message.content

        return {
            "question": question,
            "contract_id": contract_id,
            "answer": answer,
            "retrieved_chunks": rag_result["retrieved_chunks"],
            "context": rag_result["context"]
        }