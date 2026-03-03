import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get('HF_TOKEN')
SPACE_REPO_ID = os.environ.get('HF_SPACE_REPO', 'YOUR_USERNAME/YOUR_SPACE_REPO')

def main():
    if not HF_TOKEN or 'YOUR_USERNAME' in SPACE_REPO_ID:
        print('Set HF_TOKEN and HF_SPACE_REPO (e.g., username/space_repo).')
        return

    login(token=HF_TOKEN)
    api = HfApi()
    api.create_repo(repo_id=SPACE_REPO_ID, repo_type='space', exist_ok=True, space_sdk='streamlit')

    api.upload_folder(folder_path='deployment', repo_id=SPACE_REPO_ID, repo_type='space', commit_message='Deploy Streamlit app')
    print('✅ Space updated:', SPACE_REPO_ID)

if __name__ == '__main__':
    main()
