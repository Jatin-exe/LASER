from huggingface_hub import snapshot_download

class HuggingFaceDatasetDownloader:
    def __init__(self, repo_id: str, local_dir: str = "pucks_dataset"):
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.repo_type = "dataset"

    def download_folder(self, folder_name: str):
        print(f"Downloading '{folder_name}/' folder from '{self.repo_id}'...")
        snapshot_download(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            allow_patterns=f"{folder_name}/*",
            local_dir=self.local_dir
        )
        print(f"Downloaded '{folder_name}/' to '{self.local_dir}/{folder_name}'")

    def download_images(self):
        self.download_folder("images")

    def download_annotations(self):
        self.download_folder("annotations")


def main():
    downloader = HuggingFaceDatasetDownloader("Laudando-Associates-LLC/pucks")
    downloader.download_images()
    downloader.download_annotations()

if __name__ == "__main__":
    main()