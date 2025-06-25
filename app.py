import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import zipfile
import tempfile
from pathlib import Path
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import gdown

from models.few_shot_model import SiameseEarNetwork, ElephantIdentifier
from models.ear_detector import SimpleEarDetector
from utils.data_loader import get_transforms

model_path = 'models/best_model.pth'
if not os.path.exists(model_path):
    try:
        st.info("Downloading model...")
        url = 'https://drive.google.com/uc?id=1ljOG8mPgQ7fQuq66Dh_oeuUwjjGMGD2W'
        os.makedirs('models', exist_ok=True)
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()
# Page config
st.set_page_config(
    page_title="Elephant ID System",
    page_icon="🐘",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Initialize model
        model = SiameseEarNetwork(embedding_dim=256)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Get class names
        class_names = checkpoint.get('class_names', [])

        return model, class_names, device
    else:
        return None, [], 'cpu'

def process_image(image, ear_detector, transform, device):
    """Process uploaded image and extract ear region"""
    # Convert PIL to numpy
    image_np = np.array(image)

    # Extract ear region
    ear_region = ear_detector._enhance_image(image_np)
    h, w = ear_region.shape[:2]
    ear_region = ear_region[:int(h*0.4), :]  # Focus on ear region

    # Apply transforms
    if transform:
        transformed = transform(image=ear_region)
        ear_tensor = transformed['image']
        ear_tensor = ear_tensor.unsqueeze(0).to(device)

    return ear_region, ear_tensor

def predict_elephant(model, ear_tensor, class_names, device):
    """Predict elephant identity"""
    with torch.no_grad():
        # Get embedding
        embedding = model(ear_tensor)

        # For demo purposes, we'll use a simple similarity comparison
        # In practice, you'd use the prototype matching from training

        # Mock predictions for demo (replace with actual prototype matching)
        similarities = torch.rand(len(class_names))  # Random similarities for demo
        top_k = 3

        # Get top predictions
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(class_names)))

        predictions = []
        for sim, idx in zip(top_similarities, top_indices):
            if idx < len(class_names):
                predictions.append((class_names[idx], sim.item()))

        return predictions

def process_zip_batch(zip_file, model, transform, ear_detector, device):
    """Process a batch of elephant images from a ZIP file and group by similarity"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Extract
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        image_paths = list(Path(temp_dir).rglob("*.jpg")) + list(Path(temp_dir).rglob("*.png"))

        embeddings = []
        file_map = {}

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                ear_region, ear_tensor = process_image(image, ear_detector, transform, device)

                with torch.no_grad():
                    embedding = model(ear_tensor).cpu().numpy()
                    embeddings.append(embedding[0])
                    file_map[path] = embedding[0]

            except Exception as e:
                print(f"Error processing {path.name}: {e}")

        if not embeddings:
            return None, "No valid images processed."

        # Step 2: Group by similarity (naive threshold clustering)
        similarity_threshold = 0.85
        grouped = []
        used = set()
        embeddings = np.array(embeddings)

        for i, emb in enumerate(embeddings):
            if i in used:
                continue

            group = [list(file_map.keys())[i]]
            used.add(i)

            for j in range(i + 1, len(embeddings)):
                if j not in used:
                    sim = cosine_similarity([emb], [embeddings[j]])[0][0]
                    if sim >= similarity_threshold:
                        group.append(list(file_map.keys())[j])
                        used.add(j)

            grouped.append(group)

        # Step 3: Save to folders
        output_dir = tempfile.mkdtemp()

        for idx, group in enumerate(grouped):
            group_path = os.path.join(output_dir, f"Elephant_Group_{idx+1}")
            os.makedirs(group_path, exist_ok=True)

            for img_path in group:
                shutil.copy(img_path, os.path.join(group_path, os.path.basename(img_path)))

        # Step 4: Zip the grouped folder
        zip_output_path = shutil.make_archive(os.path.join(output_dir, "grouped_elephants"), 'zip', output_dir)

        return zip_output_path, None

def main():
    st.title("🐘 Asian Elephant Individual Identification System")
    st.markdown("Upload an elephant image to identify the individual using ear pattern recognition")

    # Load model
    model, class_names, device = load_model()

    if model is None:
        st.error("❌ Model not found! Please train the model first using `python train.py`")
        st.stop()

    st.success(f"✅ Model loaded successfully! Trained on {len(class_names)} elephants")

    # Initialize ear detector and transforms
    ear_detector = SimpleEarDetector()
    transform = get_transforms('val')

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This system uses AI to identify individual Asian elephants based on their unique ear patterns.")

        st.header("📊 Model Info")
        st.write(f"**Elephants in database:** {len(class_names)}")
        st.write(f"**Device:** {device}")
        st.write(f"**Model:** Siamese Network with EfficientNet")

        st.header("🔧 How it works")
        st.write("1. Upload elephant image")
        st.write("2. System extracts ear region")
        st.write("3. AI analyzes ear patterns")
        st.write("4. Returns top matches")

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an elephant image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of an elephant showing the ear region"
        )

        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process image
            with st.spinner("Processing image..."):
                try:
                    ear_region, ear_tensor = process_image(image, ear_detector, transform, device)

                    # Display processed ear region
                    st.subheader("🔍 Extracted Ear Region")
                    st.image(ear_region, caption="Ear Region (AI Focus Area)", use_column_width=True)

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.stop()

    with col2:
        if uploaded_file is not None:
            st.header("🎯 Identification Results")

            with st.spinner("Identifying elephant..."):
                try:
                    # Get predictions
                    predictions = predict_elephant(model, ear_tensor, class_names, device)

                    if predictions:
                        st.subheader("🏆 Top Matches")

                        for i, (elephant_id, confidence) in enumerate(predictions):
                            # Create confidence bar
                            confidence_pct = confidence * 100

                            # Color based on confidence
                            if confidence_pct > 70:
                                color = "🟢"
                            elif confidence_pct > 50:
                                color = "🟡"
                            else:
                                color = "🔴"

                            st.write(f"**{i+1}. {elephant_id}** {color}")
                            st.progress(confidence_pct / 100)
                            st.write(f"Confidence: {confidence_pct:.1f}%")
                            st.write("---")

                        # Best match
                        best_match = predictions[0]
                        if best_match[1] > 0.7:
                            st.success(f"🎉 **Best Match:** {best_match[0]} ({best_match[1]*100:.1f}% confidence)")
                        elif best_match[1] > 0.5:
                            st.warning(f"⚠️ **Possible Match:** {best_match[0]} ({best_match[1]*100:.1f}% confidence)")
                        else:
                            st.info("ℹ️ **Low Confidence:** This elephant may not be in our database")

                    else:
                        st.warning("No predictions available")

                except Exception as e:
                    st.error(f"Error during identification: {str(e)}")

    # ZIP file batch processing section
    st.markdown("---")
    st.header("📂 Batch Identification from ZIP (Group and Download)")
    zip_file = st.file_uploader("Upload a ZIP file of new elephant images", type=["zip"])

    if zip_file:
        with st.spinner("Processing and grouping images..."):
            zip_result, error = process_zip_batch(zip_file, model, transform, ear_detector, device)

            if error:
                st.error(error)
            else:
                st.success("🎉 Grouping completed!")

                with open(zip_result, "rb") as f:
                    st.download_button(
                        label="📥 Download Grouped Images ZIP",
                        data=f,
                        file_name="grouped_elephants.zip",
                        mime="application/zip"
                    )

    # Additional info
    st.markdown("---")
    st.markdown("### 📋 Usage Tips")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**🖼️ Image Quality**")
        st.write("• Clear, well-lit images")
        st.write("• Ear region visible")
        st.write("• Minimal obstruction")

    with col2:
        st.write("**📐 Best Angles**")
        st.write("• Side profile preferred")
        st.write("• Both ears visible if possible")
        st.write("• Close-up shots work best")

    with col3:
        st.write("**⚠️ Limitations**")
        st.write("• Requires training data")
        st.write("• Performance varies with image quality")
        st.write("• New elephants need to be added to database")

if __name__ == "__main__":
    main()
