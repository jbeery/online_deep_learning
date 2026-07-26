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
                    "You are a precise unit conversion assistant. Be concise. "
                    "Use these conversion facts when helpful: "
                    "1 km = 1000 meter. 1 meter = 100 centimeter. "
                    "1 kg = 1000 gram. 1 hour = 60 minute. 1 minute = 60 second. "
                    "1 foot = 12 inch. 1 yard = 3 foot. 1 mile = 5280 foot. "
                    "Always end with exactly one final numeric answer inside <answer></answer>. "
                    "Do not put units inside the answer tag."
                ),
            },
            {
                "role": "user",
                "content": "How many centimeter are there in 2 meter?",
            },
            {
                "role": "assistant",
                "content": "1 meter = 100 centimeter. 2 * 100 = <answer>200</answer>",
            },
            {
                "role": "user",
                "content": "How many meter are there in 300 centimeter?",
            },
            {
                "role": "assistant",
                "content": "1 meter = 100 centimeter. 300 / 100 = <answer>3</answer>",
            },
            {
                "role": "user",
                "content": "How many minute are there in 2 hour?",
            },
            {
                "role": "assistant",
                "content": "1 hour = 60 minute. 2 * 60 = <answer>120</answer>",
            },
            {
                "role": "user",
                "content": "How many foot are there in 4 yard?",
            },
            {
                "role": "assistant",
                "content": "1 yard = 3 foot. 4 * 3 = <answer>12</answer>",
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