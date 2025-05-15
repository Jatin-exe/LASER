from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

class ModelDownloader:
    def __init__(self, repo_id: str, filename: str, target_name: str, output_subdir: str):
        self.repo_id = repo_id
        self.filename = filename
        self.target_name = target_name
        self.output_dir = Path("models") / output_subdir
        self.temp_dir = Path("hf_temp_download")

    def download(self):
        print(f"Downloading '{self.filename}' from '{self.repo_id}'...")
        downloaded_dir = snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            allow_patterns=[self.filename],
            local_dir=self.temp_dir,
            local_dir_use_symlinks=False
        )
        self._copy_and_rename(downloaded_dir)
        self._cleanup()

    def _copy_and_rename(self, downloaded_dir: Path):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        src = Path(downloaded_dir) / self.filename
        dst = self.output_dir / self.target_name
        shutil.copy(src, dst)
        print(f"Model saved to: {dst}")

    def _cleanup(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

def main():
    models = [
        ("Laudando-Associates-LLC/d-fine-nano",  "dfine_hgnetv2_n_custom"),
        ("Laudando-Associates-LLC/d-fine-small", "dfine_hgnetv2_s_custom"),
        ("Laudando-Associates-LLC/d-fine-medium","dfine_hgnetv2_m_custom"),
        ("Laudando-Associates-LLC/d-fine-large", "dfine_hgnetv2_l_custom"),
        ("Laudando-Associates-LLC/d-fine-xlarge","dfine_hgnetv2_x_custom"),
    ]

    for repo_id, output_dir in models:
        downloader = ModelDownloader(
            repo_id=repo_id,
            filename="pytorch_model.bin",
            target_name="best_stg1.pth",
            output_subdir=output_dir
        )
        downloader.download()

if __name__ == "__main__":
    main()