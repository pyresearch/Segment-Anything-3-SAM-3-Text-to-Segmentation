from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch
import matplotlib.pyplot as plt
import numpy as np

device = Accelerator().device
model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# Load video frames
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

# Add text prompt to detect and track objects
text = "face"
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

# Process all frames in the video
outputs_per_frame = {}
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=50
):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")

# Access results for a specific frame
frame_0_outputs = outputs_per_frame[0]
print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
print(f"Scores: {frame_0_outputs['scores'].tolist()}")
print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
print(f"Masks shape: {frame_0_outputs['masks'].shape}")

# Visualize the video frames with detected masks (imshow)
for frame_idx in sorted(outputs_per_frame.keys()):
    # Assuming video_frames is a list of numpy arrays (RGB)
    frame = video_frames[frame_idx]
    
    # Get outputs for this frame
    outputs = outputs_per_frame[frame_idx]
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    plt.imshow(frame)
    plt.title(f"Frame {frame_idx} - Detected Persons")
    
    # Overlay masks if any objects detected
    if 'masks' in outputs and len(outputs['masks']) > 0:
        masks = outputs['masks'].cpu().numpy()  # Move to CPU and convert to numpy
        scores = outputs['scores'].cpu().numpy()
        
        for i, mask in enumerate(masks):
            # Assuming mask shape is (1, H, W) or (H, W); squeeze if necessary
            mask = np.squeeze(mask)
            # Threshold the mask (assuming it's logit or probability)
            mask_display = (mask > 0.0)  # Adjust threshold as needed
            
            # Overlay the mask with transparency
            plt.imshow(mask_display, cmap='jet', alpha=0.4, interpolation='none')
            
            # Optional: Add score as text
            plt.text(10, 30 + i*20, f"Object {outputs['object_ids'][i]} Score: {scores[i]:.2f}", color='white', bbox=dict(facecolor='black', alpha=0.5))
    
    plt.axis('off')
    plt.show()