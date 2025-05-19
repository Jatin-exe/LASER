from huggingface_hub import snapshot_download
import os
import shutil

class HFDataDownloader:
    def __init__(self, repo_id, target_folder):
        self.repo_id = repo_id
        self.target_folder = target_folder
        os.makedirs(self.target_folder, exist_ok=True)

    def download_file(self, filename):
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=self.target_folder,
            allow_patterns=[filename]
        )
        print(f"Downloaded '{filename}' to '{self.target_folder}'")

    def cleanup_cache(self):
        cache_path = os.path.join(self.target_folder, ".cache")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"Deleted cache folder: {cache_path}")
        else:
            print("ℹNo .cache folder found. Skipping cleanup.")

if __name__ == "__main__":
    repo_id = "Laudando-Associates-LLC/pucks-ros2"
    target_folder = "/ros2_ws/src/alpha_perception/datasets/04-23-2024_4w"
    files_to_download = [
        "metadata.yaml",
        "test_0.db3",
    ]

    downloader = HFDataDownloader(repo_id, target_folder)

    for file in files_to_download:
        downloader.download_file(file)

    # Delete the .cache folder after all downloads
    downloader.cleanup_cache()
