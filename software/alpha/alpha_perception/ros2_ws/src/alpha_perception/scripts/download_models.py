from huggingface_hub import snapshot_download
from pathlib import Path
import os
import shutil
import subprocess
import onnx
from onnx import version_converter

class ModelProcessor:
    def __init__(self, repo_id: str, output_subdir: str):
        self.repo_id = repo_id
        self.filename = "model.onnx"
        self.output_dir = Path("/ros2_ws/src/alpha_perception/models") / output_subdir
        self.temp_dir = Path("hf_temp_download")
        self.trt_engine = self.output_dir / "model.trt"
        self.opset_model = self.output_dir / "model_opset17.onnx"

    def download_upgrade_quantise(self):
        downloaded_dir = snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            allow_patterns=[self.filename],
            local_dir=self.temp_dir
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(downloaded_dir) / self.filename

        # Convert to opset 17
        model = onnx.load(str(src_path))
        if model.opset_import[0].version < 17:
            model = version_converter.convert_version(model, 17)
        onnx.save(model, str(self.opset_model))

        # Clean up temp
        shutil.rmtree(self.temp_dir)

        print(f"Quantizing ONNX model..........")
        # Quantize with trtexec
        self._build_trt_engine()

    def _build_trt_engine(self):
        env = os.environ.copy()
        env["CUDA_MODULE_LOADING"] = "LAZY"
        trtexec_cmd = [
            "/usr/src/tensorrt/bin/trtexec",
            f"--onnx={self.opset_model}",
            "--builderOptimizationLevel=3",
            "--useSpinWait",
            "--useRuntime=full",
            "--useCudaGraph",
            "--precisionConstraints=prefer",
            "--layerPrecisions=*:fp16",
            "--allowGPUFallback",
            "--tacticSources=+CUBLAS,+CUDNN,+JIT_CONVOLUTIONS,+CUBLAS_LT,+EDGE_MASK_CONVOLUTIONS",
            "--sparsity=enable",
            f"--saveEngine={self.trt_engine}",
            "--fp16",
            "--minShapes=images:1x3x640x640,orig_target_sizes:1x2",
            "--optShapes=images:1x3x640x640,orig_target_sizes:1x2",
            "--maxShapes=images:1x3x640x640,orig_target_sizes:1x2",
            "--inputIOFormats=fp16:chw,int64:chw",
            "--outputIOFormats=int64:chw,fp16:chw,fp16:chw",
            "--maxTactics=2000",
            "--allocationStrategy=static",
            "--avgTiming=32",
            "--maxAuxStreams=4"
        ]

        subprocess.run(trtexec_cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if self.opset_model.exists():
            self.opset_model.unlink()

def main():
    models = [
        ("Laudando-Associates-LLC/d-fine-nano",  "dfine_hgnetv2_n_custom"),
        ("Laudando-Associates-LLC/d-fine-small", "dfine_hgnetv2_s_custom"),
        ("Laudando-Associates-LLC/d-fine-medium","dfine_hgnetv2_m_custom"),
        ("Laudando-Associates-LLC/d-fine-large", "dfine_hgnetv2_l_custom"),
        ("Laudando-Associates-LLC/d-fine-xlarge","dfine_hgnetv2_x_custom"),
    ]

    print("Processing models with opset conversion and quatisation... This may take several minutes based on the CUDA GPU Compute Capability\n")
    for repo_id, output_dir in models:
        print(f"{output_dir}")
        processor = ModelProcessor(repo_id, output_dir)
        processor.download_upgrade_quantise()

if __name__ == "__main__":
    main()
