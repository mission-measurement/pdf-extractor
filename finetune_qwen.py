import os
import json
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def load_training_data(file_path, processor, max_length=2048):
    examples = [json.loads(line) for line in open(file_path) if line.strip()]
    
    processed = []
    for ex in examples:
        messages = [{"role": t['role'], "content": t['content']} for t in ex['conversations']]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(text=text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
        inputs["labels"] = inputs["input_ids"].clone()
        processed.append({k: v.squeeze(0) for k, v in inputs.items()})
    
    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    ))
    
    dataset = load_training_data(args.training_data, processor)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=100,
            fp16=True,
            report_to="none"
        ),
        train_dataset=split['train'],
        eval_dataset=split['test']
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.save_pretrained(os.path.join(args.output_dir, "lora_weights"))


if __name__ == "__main__":
    main()
