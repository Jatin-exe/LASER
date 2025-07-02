import triton_python_backend_utils as pb_utils
import torch

def tensor_dict(response):
    """
    Converts InferenceResponse output tensors into a dict for O(1) access by name.
    """
    return {t.name(): t for t in response.output_tensors()}

def tensor_to_triton_gpu(name, tensor):
    """
    Converts a Triton tensor to a new GPU-backed Triton tensor using DLPack.
    """
    torch_tensor = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
    return pb_utils.Tensor.from_dlpack(name, torch.utils.dlpack.to_dlpack(torch_tensor))

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = args["model_config"]
        print("[Router] Initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input image
            input_image = pb_utils.get_input_tensor_by_name(request, "input_image")

            # Get model_name (TYPE_STRING), decode to str
            model_name_tensor = pb_utils.get_input_tensor_by_name(request, "model_name")
            crop_type = model_name_tensor.as_numpy()[0].decode('utf-8')
            model_name = f"perception_{crop_type}"

            # Step 1: Preprocessing
            preproc_resp = pb_utils.InferenceRequest(
                model_name="perception_torch_preprocessing",
                requested_output_names=["preprocessed_blob", "original_height", "original_width"],
                inputs=[input_image]
            ).exec()

            if preproc_resp.has_error():
                raise RuntimeError(f"[Router] Preprocessing failed: {preproc_resp.error().message()}")

            preproc_outputs = tensor_dict(preproc_resp)
            preprocessed_blob = preproc_outputs["preprocessed_blob"]
            original_height = preproc_outputs["original_height"]
            original_width = preproc_outputs["original_width"]

            # Step 2: Main perception model (dynamic) — GPU to GPU
            model_resp = pb_utils.InferenceRequest(
                model_name=model_name,
                requested_output_names=["output0"],
                inputs=[
                    tensor_to_triton_gpu("images", preprocessed_blob)
                ]
            ).exec()

            if model_resp.has_error():
                raise RuntimeError(f"[Router] Inference failed for {model_name}: {model_resp.error().message()}")

            model_outputs = tensor_dict(model_resp)
            raw_outputs = model_outputs["output0"]

            # Step 3: Postprocessing — GPU to GPU
            postproc_resp = pb_utils.InferenceRequest(
                model_name="perception_postprocessing",
                requested_output_names=["boxes", "areas", "scores", "keypoints", "ids"],
                inputs=[
                    tensor_to_triton_gpu("raw_outputs", raw_outputs),
                    original_height,
                    original_width
                ]
            ).exec()

            if postproc_resp.has_error():
                raise RuntimeError(f"[Router] Postprocessing failed: {postproc_resp.error().message()}")

            postproc_outputs = tensor_dict(postproc_resp)

            # Step 4: Return all final outputs
            response = pb_utils.InferenceResponse(output_tensors=[
                postproc_outputs["boxes"],
                postproc_outputs["areas"],
                postproc_outputs["scores"],
                postproc_outputs["keypoints"],
                postproc_outputs["ids"],
            ])
            responses.append(response)

        return responses

