from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):
    # Reuse much of the SFT code here
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Trainer, TrainingArguments

    from .data import Dataset
    from .sft import TokenizedDataset

    def format_example(prompt: str, answer: str, reasoning: str) -> dict[str, str]:
        return {
            "question": prompt,
            "answer": reasoning,
        }

    llm = BaseLLM()
    llm.tokenizer.pad_token = llm.tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.config.use_cache = False

    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("rft"), format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=5e-4,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=10,
        **kwargs,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    llm.model.save_pretrained(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})