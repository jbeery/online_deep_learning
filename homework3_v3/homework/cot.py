from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a unit conversion assistant. Be concise. "
                    "Compute the answer step by step. "
                    "Always finish with the final number inside <answer></answer>. "
                    "Do not include units inside the answer tag."
                ),
            },
            {
                "role": "user",
                "content": "How many centimeter are there in 3 meter?",
            },
            {
                "role": "assistant",
                "content": "1 meter = 100 centimeter. 3 * 100 = <answer>300</answer>",
            },
            {
                "role": "user",
                "content": "How many meter are there in 250 centimeter?",
            },
            {
                "role": "assistant",
                "content": "1 meter = 100 centimeter. 250 / 100 = <answer>2.5</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})