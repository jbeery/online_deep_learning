def generate_dataset(output_json: str = "data/rft.json", oversample: int = 10, temperature: float = 0.6):
    import json
    from pathlib import Path

    from tqdm import tqdm

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    trainset = Dataset("train")

    generated_data = []
    batch_size = 1

    for start in tqdm(range(0, len(trainset), batch_size), desc="Generating RFT data"):
        batch = [trainset[i] for i in range(start, min(start + batch_size, len(trainset)))]
        questions = [item[0] for item in batch]
        correct_answers = [float(item[1]) for item in batch]
        prompts = [model.format_prompt(question) for question in questions]

        generations = model.batched_generate(
            prompts,
            num_return_sequences=oversample,
            temperature=temperature,
        )

        for question, correct_answer, candidates in zip(questions, correct_answers, generations):
            for candidate in candidates:
                parsed_answer = model.parse_answer(candidate)
                if parsed_answer == parsed_answer and is_answer_valid(parsed_answer, correct_answer):
                    generated_data.append([question, correct_answer, candidate.strip()])
                    break

    with output_path.open("w") as f:
        json.dump(generated_data, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)