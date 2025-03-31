import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import albumentations as A
from torchvision import transforms
import time

# Initialize MTCNN and ResNet
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Global variable to store enrollment embeddings
if "enrollment_embeddings" not in st.session_state:
    st.session_state.enrollment_embeddings = None

# WebRTC configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class FaceTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return self.frame

def align_face(image):
    pil_image = Image.fromarray(image)
    aligned_face = mtcnn(pil_image)
    if aligned_face is None:
        st.warning("No face detected")
        return None
    return aligned_face

def augment_face(image):
    transforms_dict = {
        "rotations": A.Rotate(limit=15, p=1.0),
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        "blur": A.Blur(blur_limit=3, p=1.0),
        "shadow": A.RandomGamma(gamma_limit=(80, 120), p=1.0)
    }
    augmented_images = []
    for transform in transforms_dict.values():
        augmented = transform(image=image)
        augmented_images.append(augmented["image"])
    return augmented_images

def get_facenet_embedding(aligned_face):
    if aligned_face is None:
        return None
    if isinstance(aligned_face, torch.Tensor):
        aligned_face = transforms.ToPILImage()(aligned_face)
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_tensor = preprocess(aligned_face).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(face_tensor)
        embedding = embedding / embedding.norm(p=2)
    return embedding.cpu().numpy()

def enroll_pipeline(captured_image):
    enrollment_embeddings = []
    img = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
    aligned_face = align_face(img)
    if aligned_face is None:
        return "No face detected during enrollment."
    if isinstance(aligned_face, torch.Tensor):
        aligned_face = transforms.ToPILImage()(aligned_face.cpu())
    embedding = get_facenet_embedding(aligned_face)
    if embedding is not None:
        enrollment_embeddings.append(embedding)
    aligned_np = np.array(aligned_face)
    if aligned_np.dtype != np.uint8:
        aligned_np = (aligned_np * 255).astype(np.uint8)
    augmented_images = augment_face(aligned_np)
    for aug_img in augmented_images:
        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        aug_embedding = get_facenet_embedding(Image.fromarray(aug_img))
        if aug_embedding is not None:
            enrollment_embeddings.append(aug_embedding)
    st.session_state.enrollment_embeddings = enrollment_embeddings
    return "Enrollment completed."

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def authenticate(live_image, threshold=0.8):
    if st.session_state.enrollment_embeddings is None:
        return "No enrollment data available. Please enroll a face first."
    img = cv2.cvtColor(live_image, cv2.COLOR_BGR2RGB)
    aligned_live = align_face(img)
    if aligned_live is None:
        return "No face detected in live image."
    live_embedding = get_facenet_embedding(aligned_live)
    if live_embedding is None:
        return "Failed to generate embedding for live image."
    for emb in st.session_state.enrollment_embeddings:
        similarity = cosine_similarity(live_embedding.flatten(), emb.flatten())
        if similarity >= threshold:
            return "Unlocked! Face authenticated."
    return "Authentication failed. Face not recognized."

# Streamlit UI
st.title("Face Recognition System")
st.markdown("Use the webcam to enroll or authenticate your face.")

# Webcam streamer
ctx = webrtc_streamer(
    key="face-recognition",
    video_transformer_factory=FaceTransformer,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

mode = st.radio("Select Mode", ["Enroll Face", "Unlock Face"])

if st.button("Submit"):
    if ctx.video_transformer and ctx.video_transformer.frame is not None:
        frame = ctx.video_transformer.frame
        if mode == "Enroll Face":
            result = enroll_pipeline(frame)
            st.success(result)
        elif mode == "Unlock Face":
            result = authenticate(frame)
            st.write(result)
    else:
        st.error("No webcam frame available. Ensure the webcam is running.")

if st.button("Clear"):
    st.session_state.enrollment_embeddings = None
    st.write("Enrollment data cleared.")
