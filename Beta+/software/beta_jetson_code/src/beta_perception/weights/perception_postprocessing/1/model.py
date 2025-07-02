import numpy as np
import triton_python_backend_utils as pb_utils
import cv2
import time

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model. This function is called only once when the model is loaded."""
        pass

    def execute(self, requests):
        """This function is called for every inference request."""
        responses = []

        for request in requests:
            # Retrieve input tensors from the request
            outputs_tensor = pb_utils.get_input_tensor_by_name(request, "raw_outputs")
            original_height_tensor = pb_utils.get_input_tensor_by_name(request, "original_height")
            original_width_tensor = pb_utils.get_input_tensor_by_name(request, "original_width")

            # Convert the input tensors to numpy arrays
            raw_outputs = outputs_tensor.as_numpy()[0]
            original_height = original_height_tensor.as_numpy()[0]  # Single value
            original_width = original_width_tensor.as_numpy()[0]    # Single value
            # **Post-processing logic will go here later**
            boxes, scores, kpts, areas, class_ids = self.get_outputs(raw_outputs, 0.3, 0.3)
            
            scaled_boxes, scaled_kpts, scaled_areas = self.scale_outputs(boxes, kpts, areas, original_width, original_height)
            # Create output tensors
            output_boxes_tensor = pb_utils.Tensor("boxes", scaled_boxes.astype(np.int32))
            output_areas_tensor = pb_utils.Tensor("areas", scaled_areas.astype(np.int32))
            output_scores_tensor = pb_utils.Tensor("scores", scores.astype(np.float16))
            output_keypoints_tensor = pb_utils.Tensor("keypoints", scaled_kpts.astype(np.int32))
            output_ids_tensor = pb_utils.Tensor("ids", class_ids.astype(np.int32))

            # Build the inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                output_boxes_tensor, output_areas_tensor, output_scores_tensor, output_keypoints_tensor, output_ids_tensor
            ])

            # Append the response to the list of responses
            responses.append(inference_response)

        return responses
    
    def scale_outputs(self, boxes, kpts, areas, original_width, original_height):
        # If there are no valid detections, return empty arrays to avoid broadcasting errors
        if boxes.size == 0 or kpts.size == 0 or areas.size == 0:
            return boxes, kpts, areas

        # Calculate scaling factors
        scale_x = original_width / 640  # Assuming model input width was 640
        scale_y = original_height / 384  # Assuming model input height was 384
        scale_factors = np.array([scale_x, scale_y, scale_x, scale_y])

        # Scale boxes
        boxes = boxes * scale_factors

        # Precompute maximum valid dimensions
        max_width = original_width - boxes[:, 0]  # Maximum valid width for each box
        max_height = original_height - boxes[:, 1]  # Maximum valid height for each box

        # Clip top-left corner (x, y)
        boxes[:, 0:2] = np.clip(boxes[:, 0:2], 0, [original_width - 1, original_height - 1])

        # Clip width and height
        boxes[:, 2] = np.clip(boxes[:, 2], 0, max_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, max_height)

        # Scale keypoints
        kpts = kpts * scale_factors[:2]

        # Clip keypoints to lie within the image dimensions
        kpts[:, 0] = np.clip(kpts[:, 0], 0, original_width - 1)  # Clip x-coordinates
        kpts[:, 1] = np.clip(kpts[:, 1], 0, original_height - 1)  # Clip y-coordinates

        # Scale areas
        areas = areas * (scale_x * scale_y)

        return boxes, kpts, areas
    
    def get_outputs(self, data, conf_thresh, iou_thresh):
        """
        Dynamically handle class scores in data and process detections.

        Args:
            data: np.ndarray with shape (N, 5040), where N >= 7.
            conf_thresh: Confidence threshold for filtering detections.
            iou_thresh: IOU threshold for NMS.

        Returns:
            final_boxes, final_scores, final_kpts, final_areas, final_class_ids
        """
        # Access fixed rows
        center_x = data[0, :]
        center_y = data[1, :]
        width = data[2, :]
        height = data[3, :]
        keypoints = data[-2:, :].T  # Always the last two rows (transpose for shape (5040, 2))

        # Dynamically determine class scores
        num_classes = data.shape[0] - 6  # Total rows minus fixed (4) and keypoints (2)
        class_scores = {}
        for i in range(num_classes):
            class_name = f"class_{i}"  # You can replace this with actual class names if known
            class_scores[class_name] = data[4 + i, :]

        # Filter detections for each class based on confidence threshold
        all_boxes = []
        all_scores = []
        all_kpts = []
        all_areas = []
        all_class_ids = []

        for class_id, (class_name, scores) in enumerate(class_scores.items()):
            # Apply confidence threshold
            valid_mask = scores > conf_thresh
            if not np.any(valid_mask):
                continue

            # Apply mask to filter valid detections
            filtered_center_x = center_x[valid_mask]
            filtered_center_y = center_y[valid_mask]
            filtered_width = width[valid_mask]
            filtered_height = height[valid_mask]
            filtered_scores = scores[valid_mask]
            filtered_keypoints = keypoints[valid_mask]

            # Calculate bounding boxes and areas
            x = (filtered_center_x - 0.5 * filtered_width).astype(int)
            y = (filtered_center_y - 0.5 * filtered_height).astype(int)
            boxes = np.stack([x, y, filtered_width.astype(int), filtered_height.astype(int)], axis=1)
            areas = filtered_width * filtered_height

            # Perform Non-Maximum Suppression (NMS)
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), filtered_scores.tolist(), conf_thresh, iou_thresh
            )
            if len(indices) == 0:
                continue

            # Gather final filtered results based on NMS indices
            indices = indices.flatten()
            final_boxes = boxes[indices]
            final_scores = filtered_scores[indices]
            final_areas = areas[indices]
            final_kpts = filtered_keypoints[indices]
            final_class_ids = np.full(len(indices), class_id, dtype=int)  # Class ID for this class

            # Append results for this class
            all_boxes.append(final_boxes)
            all_scores.append(final_scores)
            all_kpts.append(final_kpts)
            all_areas.append(final_areas)
            all_class_ids.append(final_class_ids)

        # Concatenate results across all classes
        if all_boxes:
            all_boxes = np.vstack(all_boxes)
            all_scores = np.concatenate(all_scores)
            all_kpts = np.vstack(all_kpts)
            all_areas = np.concatenate(all_areas)
            all_class_ids = np.concatenate(all_class_ids)
        else:
            all_boxes, all_scores, all_kpts, all_areas, all_class_ids = (
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            )

        return all_boxes, all_scores, all_kpts, all_areas, all_class_ids

    

    def finalize(self):
        """This function is called only once when the model is unloaded."""
        pass
