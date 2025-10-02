from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
import numpy as np
import os

# Define model name
MODEL_NAME = "distilbert-base-uncased"

def train(tokenizer, tokenized_datasets, data_collator, model_name=MODEL_NAME):

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Compute metrics
    accuracy = evaluate.load('accuracy')

    def compute_metrics(pred):

        logits, label = pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=label)
    
    # Define output directory
    output_dir = os.path.join("..", "models", model_name.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir=os.path.join(output_dir, 'logs')
    )

    # Train and save models
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():

    # Load dataset
    data_files = {
        "train": "../datasets/train_data.csv",
        "test": "../datasets/test_data.csv"
    }
    dataset = load_dataset('csv', data_files=data_files)

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        tokenized = tokenizer(examples['v2'], truncation=True)
        tokenized['labels'] = examples['v1']
        return tokenized
        
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train(tokenizer, tokenized_datasets, data_collator)

if __name__ == "__main__" :
    main()                                           
