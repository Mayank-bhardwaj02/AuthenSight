import gradio as gr
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import albumentations as A
from torchvision import transforms

# Initialize models
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
resnet = InceptionResnetV1(pretrained="vggface2").eval()

# --- Face Processing Functions ---

def align_face(image):
    # Ensure the image is a PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    aligned_face = mtcnn(image)
    if aligned_face is None:
        print("No face detected")
        return None
    return aligned_face

def augment_face(image):
    # 'image' should be a NumPy array (H x W x C) in uint8
    transforms_dict = {
        "rotations": A.Rotate(limit=15, p=1.0),
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        "blur": A.Blur(blur_limit=3, p=1.0),
        "shadow": A.RandomGamma(gamma_limit=(80, 120), p=1.0)
    }
    augmented_images = []
    for name, transform in transforms_dict.items():
        augmented = transform(image=image)
        augmented_images.append(augmented["image"])
    return augmented_images

def get_facenet_embedding(aligned_face):
    if aligned_face is None:
        return None
    # If it's a tensor, convert it to a PIL Image
    if isinstance(aligned_face, torch.Tensor):
        aligned_face = transforms.ToPILImage()(aligned_face)
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    face_tensor = preprocess(aligned_face).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(face_tensor)
        embedding = embedding / embedding.norm(p=2)
    return embedding.cpu().numpy()

def enroll_pipeline(captured_images):
    enrollment_embeddings = []
    for img in captured_images:
        aligned_face = align_face(img)
        if aligned_face is None:
            continue
        if isinstance(aligned_face, torch.Tensor):
            aligned_face = transforms.ToPILImage()(aligned_face.cpu())
        embedding = get_facenet_embedding(aligned_face)
        if embedding is not None:
            enrollment_embeddings.append(embedding)
        # Prepare image for augmentation
        aligned_np = np.array(aligned_face)
        if aligned_np.dtype != np.uint8:
            aligned_np = (aligned_np * 255).astype(np.uint8)
        augmented_images = augment_face(aligned_np)
        for aug_img in augmented_images:
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
            try:
                pil_aug_img = Image.fromarray(aug_img)
            except Exception as e:
                print("Error converting augmented image to PIL Image:", e)
                continue
            aug_embedding = get_facenet_embedding(pil_aug_img)
            if aug_embedding is not None:
                enrollment_embeddings.append(aug_embedding)
    return enrollment_embeddings

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def authenticate(enrollment_embeddings, live_image, threshold=0.8):
    aligned_live = align_face(live_image)
    if aligned_live is None:
        print("No face detected in live image.")
        return False
    live_embeddings = get_facenet_embedding(aligned_live)
    if live_embeddings is None:
        return False
    for emb in enrollment_embeddings:
        similarity = cosine_similarity(live_embeddings.flatten(), emb.flatten())
        if similarity >= threshold:
            return True
    return False

# Global variable to store enrollment embeddings
enrollment_embeddings = None

def enrollment_mode_webcam(image):
    global enrollment_embeddings
    if image is None:
        return "No image captured. Please try again."
    # Wrap the single captured image in a list for processing
    captured_images = [np.array(image)]
    enrollment_embeddings = enroll_pipeline(captured_images)
    return "Enrollment completed."

def authentication_mode_webcam(image):
    if image is None:
        return "No image captured. Please try again."
    if enrollment_embeddings is None:
        return "No enrollment data available. Please enroll a face first."
    match_found = authenticate(enrollment_embeddings, np.array(image), threshold=0.8)
    if match_found:
        return "Unlocked! Face authenticated."
    else:
        return "Authentication failed. Face not recognized."

def face_recognition_system(mode, image):
    if mode == "Enroll Face":
        return enrollment_mode_webcam(image)
    elif mode == "Unlock Face":
        return authentication_mode_webcam(image)
    else:
        return "Invalid mode selected."

# --- Gradio UI with Custom CSS ---

custom_css = """
.gradio-container {
  background-color: #000000 !important;
  color: #ffffff !important;
}
.gr-button, button {
  background: linear-gradient(45deg, #FFA500, #FFFF00) !important;
  color: #000000 !important;
  border: none !important;
  font-weight: 500 !important;
  cursor: pointer !important;
}
h1, h2, .title, .description, label {
  color: #FFA500 !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# Face Recognition System")
    gr.Markdown("Select 'Enroll Face' to capture an image for enrollment. Select 'Unlock Face' to authenticate.")
    mode = gr.Radio(["Enroll Face", "Unlock Face"], label="Select Mode")
    # Use gr.Image with source="webcam" for browser-based capture.
    camera_input = gr.Image(source="webcam", type="pil", label="Capture Your Face")
    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")
    output = gr.Textbox(label="Output")
    submit_btn.click(fn=face_recognition_system, inputs=[mode, camera_input], outputs=output)
    clear_btn.click(fn=lambda: (gr.update(value=None), gr.update(value=None), ""), 
                    inputs=None, 
                    outputs=[mode, camera_input, output])
    
demo.launch()
