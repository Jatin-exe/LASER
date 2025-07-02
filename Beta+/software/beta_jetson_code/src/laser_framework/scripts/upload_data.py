import os
import shutil
import boto3
import argparse
from pathlib import Path
import subprocess
import socket
import sys
import time
import urllib.request

class S3DatasetUploader:
    def __init__(self):
        self.args = self.parse_args()
        self.bucket_name = self.args.bucket
        self.local_dataset_dir = Path(self.args.local_dir)
        self.s3_base_prefix = 'datasets'
        self.s3 = boto3.client('s3')
        self.total_files = 0
        self.uploaded_files = 0
        self.last_progress = -1

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Upload datasets to S3 and publish ROS 2 progress via CLI.')
        parser.add_argument('--bucket', required=True, help='S3 bucket name (e.g., laserweeder)')
        parser.add_argument('--local-dir', required=True, help='Local path to datasets folder')
        return parser.parse_args()

    def check_internet(self, timeout=3, retries=10, delay=1):
        for attempt in range(1, retries + 1):
            try:
                urllib.request.urlopen("https://www.google.com", timeout=timeout)
                if attempt > 1:
                    print(f"🌐 Internet restored after {attempt} attempt(s)")
                return True
            except OSError:
                print(f"⚠️ Internet check failed ({attempt}/{retries})...")
                time.sleep(delay)
        return False

    def publish_progress(self, value):
        self.last_progress = value
        print(f"📤 Progress: {value}%")
        for i in range(5):  # Publish 5 times reliably
            cmd = [
                'ros2', 'topic', 'pub', '--once', '/laser/progress', 'std_msgs/msg/Int32',
                f'{{data: {value}}}'
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #time.sleep(0.2)  # Slight delay to help DDS propagate

    def handle_disconnect_and_exit(self):
        print("❌ Lost internet connection. Publishing 100% and exiting.")
        self.publish_progress(100)
        sys.exit(1)

    def run(self):
        if not self.check_internet():
            print("❌ No internet connection. Aborting upload.")
            self.publish_progress(100)
            return

        if not self.local_dataset_dir.exists():
            print(f"❌ Directory does not exist: {self.local_dataset_dir}")
            return

        all_folders = [f for f in self.local_dataset_dir.iterdir() if f.is_dir()]
        self.total_files = sum(
            len(files)
            for folder in all_folders
            for _, _, files in os.walk(folder)
        )

        if self.total_files == 0:
            print("⚠️ No files found to upload.")
            self.publish_progress(100)
            return

        print(f"🔍 Found {self.total_files} files to upload.")

        for folder in all_folders:
            crop_type = folder.name.split('_')[0].lower()
            s3_crop_folder = crop_type
            s3_dataset_folder = folder.name

            print(f"\n🌱 Uploading: {folder.name} → s3://{self.bucket_name}/{self.s3_base_prefix}/{s3_crop_folder}/{s3_dataset_folder}/")
            self.upload_folder(folder, s3_crop_folder, s3_dataset_folder)

        self.publish_progress(100)
        print("✅ Upload complete.")

    def upload_folder(self, local_folder_path: Path, s3_crop_folder: str, s3_dataset_folder: str):
        for root, _, files in os.walk(local_folder_path):
            for file in files:
                if not self.check_internet():
                    self.handle_disconnect_and_exit()

                local_path = Path(root) / file
                relative_path = local_path.relative_to(local_folder_path)
                s3_key = f"{self.s3_base_prefix}/{s3_crop_folder}/{s3_dataset_folder}/{relative_path.as_posix()}"

                try:
                    self.s3.upload_file(str(local_path), self.bucket_name, s3_key)
                    os.remove(local_path)  # ✅ Delete the file after upload
                    print(f"🗑️ Deleted file: {local_path}")
                except Exception as e:
                    print(f"❌ Failed to upload {local_path}: {e}")
                    self.handle_disconnect_and_exit()
                finally:
                    self.uploaded_files += 1
                    progress = int((self.uploaded_files / self.total_files) * 100)
                    self.publish_progress(progress)

        # ✅ Delete the subfolder after all files uploaded
        try:
            shutil.rmtree(local_folder_path)
            print(f"🗑️ Deleted folder: {local_folder_path}")
        except Exception as e:
            print(f"⚠️ Failed to delete folder {local_folder_path}: {e}")

if __name__ == "__main__":
    uploader = S3DatasetUploader()
    uploader.run()

