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
                    "You are a precise unit conversion assistant. "
                    "Answer every question with a short calculation. "
                    "You must always end with exactly one final numeric answer inside "
                    "<answer></answer>. Do not put units inside the answer tag."
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
                "content": "How many gram are there per 6 kg?",
            },
            {
                "role": "assistant",
                "content": "1 kg = 1000 gram. 6 * 1000 = <answer>6000</answer>",
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