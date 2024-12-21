from huggingface_hub import HfApi, HfFolder


lang_codes = ['ar','tl','id','fa']
languages = ['arabic','tagalog','indonesian','persian']
# Set the repo_id to your model's repository
api = HfApi()

for lang, code in zip(languages, lang_codes):

    repo_id = f"jdoerfler/SpaCy-{code}_dep_web_sm"  # Change this to your model repository name

    # Initialize the Hugging Face API client


    # Push model files to the Hugging Face Hub
    api.upload_folder(
        folder_path=f"./{lang}/output/{code}_dep_web_sm",  # Path to your model directory
        repo_id=repo_id,
        repo_type="model",  # You can also use 'dataset', 'space' if appropriate
)