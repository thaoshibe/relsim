#!/usr/bin/env python3
"""
Script to push anonymous captions dataset to HuggingFace Hub
"""

from datasets import Dataset, DatasetDict
import json
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def main():
    # Load the data
    print("Loading train data...")
    train_data = load_jsonl("anonymous_captions_train.jsonl")
    print(f"Loaded {len(train_data)} training examples")
    
    print("Loading test data...")
    test_data = load_jsonl("anonymous_captions_test.jsonl")
    print(f"Loaded {len(test_data)} test examples")
    
    # Create HuggingFace datasets
    print("\nCreating HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    print("\nDataset info:")
    print(dataset_dict)
    print("\nTrain dataset features:", train_dataset.features)
    print("Sample from train:", train_dataset[0])
    
    # Push to HuggingFace Hub
    # Replace 'your-username/dataset-name' with your actual HuggingFace username and desired dataset name
    repo_id = input("\nEnter HuggingFace repo ID (e.g., 'username/dataset-name'): ").strip()
    
    if not repo_id:
        print("No repo ID provided. Skipping upload.")
        print("\nTo push later, use:")
        print(f"  dataset_dict.push_to_hub('{repo_id}')")
        return
    
    print(f"\nPushing to HuggingFace Hub: {repo_id}")
    print("Make sure you're logged in with: huggingface-cli login")
    
    confirm = input("Continue with upload? (y/n): ").strip().lower()
    if confirm == 'y':
        dataset_dict.push_to_hub(repo_id, private=False)
        print(f"\n✓ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")
    else:
        print("Upload cancelled.")

if __name__ == "__main__":
    main()

