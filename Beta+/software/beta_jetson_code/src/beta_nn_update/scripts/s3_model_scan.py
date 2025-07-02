import boto3
import json
from urllib.parse import urlparse

class S3ModelResolver:
    def __init__(self, bucket_name="laserweeder", prefix="neural_networks", s3_client=None):
        self.bucket = bucket_name
        self.prefix = prefix.rstrip("/")
        self.s3 = s3_client or boto3.client("s3")

    def list_crop_types(self):
        paginator = self.s3.get_paginator("list_objects_v2")
        result = paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}/", Delimiter="/")
        crops = []

        for page in result:
            for cp in page.get("CommonPrefixes", []):
                crop = cp["Prefix"].split("/")[-2]
                crops.append(crop)
        return crops

    def get_latest_versions(self):
        crop_versions = {}
        crop_types = self.list_crop_types()

        for crop in crop_types:
            latest = self._get_latest_artifact_version(crop)
            if latest:
                crop_versions[crop] = latest

        return crop_versions

    def _get_latest_artifact_version(self, crop):
        prefix = f"{self.prefix}/{crop}/"
        result = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter="/")

        folders = sorted([
            obj["Prefix"].rstrip("/").split("/")[-1]
            for obj in result.get("CommonPrefixes", [])
        ], reverse=True)

        prev_version = None

        for folder in folders:
            metadata_key = f"{prefix}{folder}/metadata.json"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=metadata_key)
                metadata = json.loads(obj["Body"].read().decode("utf-8"))
                version = metadata.get("artifact_version")

                if prev_version and version >= prev_version:
                    break  # Versions are no longer decreasing

                prev_version = version
            except Exception as e:
                print(f"⚠️ Skipping {metadata_key}: {e}")
                continue

            return version  # ✅ Return the latest valid version

        return None


# Example usage
if __name__ == "__main__":
    resolver = S3ModelResolver()
    versions = resolver.get_latest_versions()

    print("☁️ Latest S3 model versions:")
    for crop, version in versions.items():
        print(f"  • {crop}: {version}")
