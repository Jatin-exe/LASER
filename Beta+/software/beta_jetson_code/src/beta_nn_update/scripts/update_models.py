import os
import json
import boto3
import textwrap
import subprocess
import time

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
        crop_info = {}

        if not os.path.exists(self.weights_path):
            print(f"❌ Weights directory not found: {self.weights_path}")
            return crop_info

        for folder in os.listdir(self.weights_path):
            if not folder.startswith("perception_"):
                continue
            if folder in self.ignored_folders:
                continue

            crop_type = folder.replace("perception_", "")
            crop_dir = os.path.join(self.weights_path, folder)
            metadata_path = os.path.join(crop_dir, "metadata.json")

            version = self._read_version(metadata_path)
            crop_info[crop_type] = {
                "version": version,
                "path": crop_dir
            }

        return crop_info

    def _read_version(self, metadata_path):
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return metadata.get("artifact_version", None)
            except Exception as e:
                print(f"⚠️ Failed to read metadata: {metadata_path} — {e}")
        return None
    
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
            version_info = self._get_latest_artifact_version(crop)
            if version_info:
                crop_versions[crop] = version_info

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

                return {
                    "version": version,
                    "s3_folder": f"{prefix}{folder}"
                }
            except Exception as e:
                print(f"⚠️ Skipping {metadata_key}: {e}")
                continue

        return None
    
class ModelUpdate:
    def __init__(self, local_weights_path="./src/beta_perception/weights"):
        self.local_scanner = LocalModelScanner(local_weights_path)
        self.s3_resolver = S3ModelResolver()

    def find_updates(self):
        local_models = self.local_scanner.scan()
        s3_models = self.s3_resolver.get_latest_versions()

        updates = {}

        for crop, s3_info in s3_models.items():
            local_info = local_models.get(crop)
            local_version = local_info["version"] if local_info else None

            if local_version != s3_info["version"]:
                updates[crop] = {
                    "local_version": local_version or "🚫 Not Found",
                    "local_path": local_info["path"] if local_info else os.path.join(self.local_scanner.weights_path, f"perception_{crop}"),
                    "latest_version": s3_info["version"],
                    "s3_folder": s3_info["s3_folder"]
                }

        return updates

    def print_update_summary(self):
        updates = self.find_updates()

        if not updates:
            print("✅ All models are up to date.")
        else:
            print("🚨 Models requiring updates:")
            for crop, info in updates.items():
                print(f"  • {crop}")
                print(f"    - Local Version : {info['local_version']}")
                print(f"    - Local Path    : {info['local_path']}")
                print(f"    - Latest Version: {info['latest_version']}")
                print(f"    - S3 Folder     : {info['s3_folder']}")

    def quantize_with_trtexec(self, onnx_path, plan_path):
        if os.path.exists(plan_path):
            os.remove(plan_path)
            print(f"🧹 Removed existing plan file: {plan_path}")

        bash_command = (
            f"export CUDA_MODULE_LOADING=LAZY && "
            f"/usr/src/tensorrt/bin/trtexec "
            f"--onnx={onnx_path} "
            f"--builderOptimizationLevel=5 "
            f"--useSpinWait --useRuntime=full --useCudaGraph "
            f"--precisionConstraints=obey --allowGPUFallback "
            f"--tacticSources=+CUBLAS,+CUDNN,+JIT_CONVOLUTIONS,+CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS "
            f"--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw "
            f"--sparsity=enable --layerOutputTypes=fp16 --layerPrecisions=fp16 "
            f"--saveEngine={plan_path} --fp16 --workspace=1280"
        )

        # Just a string, no list
        process = subprocess.Popen(
            bash_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True  # Use shell=True to allow env var + full bash command
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            print(f"✅ Quantization successful: {plan_path}")
            return True
        else:
            print(f"❌ Quantization failed with return code {process.returncode}")
            return False

    def perform_updates(self):
        updates = self.find_updates()

        total_steps = len(updates) * 3  # 3 steps per crop
        completed_steps = 0

        if not updates:
            print("✅ Nothing to update.")
            return

        for crop, info in updates.items():
            local_path = info["local_path"]
            s3_folder = info["s3_folder"]

            # Check if directory already existed
            directory_existed = os.path.exists(local_path)

            # Create directory if needed
            os.makedirs(local_path, exist_ok=True)

            if not directory_existed:
                config_path = os.path.join(local_path, "config.pbtxt")
                config_text = textwrap.dedent(f'''\
                    name: "perception_{crop}"
                    platform: "tensorrt_plan"
                    max_batch_size: 0 

                    input [
                        {{
                            name: "images"  # This corresponds to your input tensor
                            data_type: TYPE_FP16  # Based on --inputIOFormats=fp16:chw
                            dims: [ 1, 3, 384, 640 ]  # Minimum shape defined as 1x3x384x640, dynamic dims can be defined
                        }}
                    ]

                    output [
                        {{
                            name: "output0"  # The first output, descriptors, corresponds to fp16 format
                            data_type: TYPE_FP16 # Based on --outputIOFormats=fp16:chw
                            dims: [ 1, -1, 5040 ]  # Set dynamic dimensions, adjust if known
                        }}
                    ]

                    instance_group [
                        {{
                            kind: KIND_GPU  # Run the model on the GPU
                        }}
                    ]

                    optimization {{
                        execution_accelerators {{
                            gpu_execution_accelerator: [
                                {{
                                    name: "tensorrt"
                                    parameters {{
                                        key: "precision_fp16"
                                        value: "true"  # Ensure FP16 precision is used, based on --fp16
                                    }}
                                    parameters {{
                                        key: "precision_constraints"
                                        value: "obey"  # Ensure precision constraints are respected, from --precisionConstraints=obey
                                    }}
                                    parameters {{
                                        key: "sparsity"
                                        value: "enable"  # Enable sparsity, based on --sparsity=enable
                                    }}
                                    parameters {{
                                        key: "tactic_sources"
                                        value: "+CUBLAS,+CUDNN,+JIT_CONVOLUTIONS,+CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS"  # Apply tactic sources
                                    }}
                                    parameters {{
                                        key: "workspace_size"
                                        value: "1280"  # Set workspace size, based on --workspace=5120
                                    }}
                                    parameters {{
                                        key: "use_cuda_graphs"
                                        value: "true"  # Enable CUDA Graphs, based on --useCudaGraph
                                    }}
                                }}
                            ]
                        }}
                    }}''')

                with open(config_path, "w") as f:
                    f.write(config_text)

                print(f"📝 Created config.pbtxt at {config_path}")

                first_subdir = os.path.join(local_path, "1")
                os.makedirs(first_subdir, exist_ok=True)

            completed_steps += 1
            self.publish_progress(int((completed_steps / total_steps) * 100))

            for filename in ["best.onnx", "metadata.json"]:
                s3_key = f"{s3_folder}/{filename}"
                local_file_path = os.path.join(local_path, filename)

                try:
                    self.s3_resolver.s3.download_file(
                        Bucket=self.s3_resolver.bucket,
                        Key=s3_key,
                        Filename=local_file_path
                    )
                    print(f"✅ Downloaded {filename} to {local_file_path}")
                except Exception as e:
                    print(f"❌ Failed to download {filename} from {s3_key}: {e}")
            
            completed_steps += 1
            self.publish_progress(int((completed_steps / total_steps) * 100))

            onnx_path = os.path.join(local_path, "best.onnx")
            subfolder = os.path.join(local_path, "1")
            plan_path = os.path.join(subfolder, "model.plan")

            if self.quantize_with_trtexec(onnx_path, plan_path):
                os.remove(onnx_path)

            completed_steps += 1
            self.publish_progress(int((completed_steps / total_steps) * 100))

        self.publish_progress(100)


    def publish_progress(self, percent: int, repeat: int = 10, delay: float = 0.01):
        """
        Publish ROS 2 progress percentage as Int16 multiple times to ensure reception.

        :param percent: Progress value (0–100)
        :param repeat: Number of times to publish the same message
        :param delay: Delay between publications in seconds
        """
        for i in range(repeat):
            try:
                cmd = f"ros2 topic pub --once /laser/model_update_progress std_msgs/msg/Int32 '{{data: {percent}}}'"
                subprocess.run(cmd, shell=True, check=True)
                print(f"📤 Published progress: {percent}% ({i+1}/{repeat})")
                #time.sleep(delay)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to publish progress on attempt {i+1}: {e}")


# Example usage
if __name__ == "__main__":
    checker = ModelUpdate()
    checker.print_update_summary()
    print("\n📥 Starting model downloads...")
    checker.perform_updates()