import os
import json

class LocalModelScanner:
    def __init__(self, weights_path="./src/beta_perception/weights"):
        self.weights_path = weights_path
        self.ignored_folders = {
            "perception",
            "perception_torch_preprocessing",
            "perception_preprocessing",
            "perception_postprocessing",
            "perception_router",
            "perception_ensemble",
        }

    def scan(self):
        crop_versions = {}

        if not os.path.exists(self.weights_path):
            print(f"❌ Weights directory not found: {self.weights_path}")
            return crop_versions

        for folder in os.listdir(self.weights_path):
            if not folder.startswith("perception_"):
                continue
            if folder in self.ignored_folders:
                continue

            crop_type = folder.replace("perception_", "")
            metadata_path = os.path.join(self.weights_path, folder, "metadata.json")

            version = self._read_version(metadata_path)
            crop_versions[crop_type] = version

        return crop_versions

    def _read_version(self, metadata_path):
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return metadata.get("artifact_version", None)
            except Exception as e:
                print(f"⚠️ Failed to read metadata: {metadata_path} — {e}")
        return None


# Example usage
if __name__ == "__main__":
    scanner = LocalModelScanner()
    local_versions = scanner.scan()

    print("📦 Local model versions:")
    for crop, version in local_versions.items():
        print(f"  • {crop}: {version if version else '🚫 No metadata'}")
