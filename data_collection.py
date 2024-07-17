import os
import time
import requests
from datasets import load_dataset
from datasets.utils import DownloadConfig
from requests.exceptions import HTTPError

# Create a data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def save_dataset(dataset_name, subset, save_dir, retries=3, delay=5):
    print(f"Downloading {dataset_name}...")
    for attempt in range(retries):
        try:
            download_config = DownloadConfig(delete_extracted=True)
            dataset = load_dataset(dataset_name, subset, split="train", download_config=download_config, trust_remote_code=True)
            save_path = os.path.join(save_dir, f"{dataset_name.replace('/', '_')}_{subset}.csv")
            dataset.to_csv(save_path)
            print(f"Saved {dataset_name} to {save_path}")
            return
        except HTTPError as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("All retry attempts failed.")
                raise
        except ValueError as e:
            print(f"ValueError: {e}")
            raise

# List of datasets to download
datasets_to_download = [
    ("sentence-transformers/parallel-sentences-opensubtitles", "all"),
    ("Helsinki-NLP/open_subtitles", "all"),
    ("wikipedia", "20200501.en"),
    ("bookcorpus", None),
    ("common_crawl", None),
]

# Download and save each dataset
for dataset_name, subset in datasets_to_download:
    save_dataset(dataset_name, subset, data_dir)

print("All datasets have been downloaded and saved.")
