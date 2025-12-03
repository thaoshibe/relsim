"""
Script to push your RelSim checkpoint to HuggingFace Hub
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
LOCAL_CHECKPOINT_PATH = "your_checkpoint_path"
REPO_ID = "username/your_repo_name"  # Change to your username

def push_to_hf():
    """Push checkpoint to HuggingFace Hub"""
    
    # Make sure you're logged in first
    # Run: huggingface-cli login
    
    print(f"Pushing checkpoint from: {LOCAL_CHECKPOINT_PATH}")
    print(f"To HuggingFace repo: {REPO_ID}")
    print("="*60)
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        print("Creating repository...")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print("✅ Repository created/verified")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload the entire folder
    print("\nUploading files...")
    api.upload_folder(
        folder_path=LOCAL_CHECKPOINT_PATH,
        repo_id=REPO_ID,
        repo_type="model"
    )
    
    print("="*60)
    print("✅ Upload complete!")
    print(f"\nYou can now load your model with:")
    print(f'  model, preprocess = relsim(pretrained=True, checkpoint_dir="{REPO_ID}")')
    print("="*60)

if __name__ == "__main__":
    # Check if checkpoint exists
    if not os.path.exists(LOCAL_CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {LOCAL_CHECKPOINT_PATH}")
        exit(1)
    
    # List files that will be uploaded
    print("Files to upload:")
    for root, dirs, files in os.walk(LOCAL_CHECKPOINT_PATH):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), LOCAL_CHECKPOINT_PATH)
            print(f"  - {rel_path}")
    
    print("\n" + "="*60)
    response = input("Proceed with upload? (y/n): ")
    
    if response.lower() == 'y':
        push_to_hf()
    else:
        print("Upload cancelled.")

