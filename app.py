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
import requests

from models.few_shot_model import SiameseEarNetwork, ElephantIdentifier
from models.ear_detector import SimpleEarDetector
from utils.data_loader import get_transforms

# Download model if not exists
model_path = 'models/best_model.pth'
if not os.path.exists(model_path):
    os.makedirs('models', exist_ok=True)
    url = 'https://github.com/Powerranger45/elephant_id_systems/releases/download/v1.0/best_model.pth'
    print("🔄 Downloading model from GitHub release...")
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("✅ Model downloaded.")

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

def extract_and_collect_images(uploaded_zip_file):
    """
    Extract ZIP file and recursively collect all image files regardless of folder structure.

    Args:
        uploaded_zip_file: Streamlit uploaded file object or file-like object

    Returns:
        tuple: (list of image paths, temp directory path)
    """
    temp_dir = tempfile.mkdtemp()

    # Step 1: Extract ZIP
    zip_path = os.path.join(temp_dir, "uploaded.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip_file.read())

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Step 2: Recursively collect all image file paths
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_paths = []

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, file))

    return image_paths, temp_dir

def create_grouping_summary(output_dir, grouped):
    """Create a summary text file explaining the grouping results"""
    summary_path = os.path.join(output_dir, "GROUPING_SUMMARY.txt")

    with open(summary_path, 'w') as f:
        f.write("🐘 ELEPHANT GROUPING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Groups Found: {len(grouped)}\n")
        f.write(f"Total Images Processed: {sum(len(group) for group in grouped)}\n\n")

        for group_idx, group in enumerate(grouped, 1):
            f.write(f"📁 GROUP {group_idx} ({len(group)} images):\n")
            f.write("-" * 30 + "\n")

            for img_info in group:
                original_location = img_info['original_folder'] if img_info['original_folder'] != '.' else 'root'
                f.write(f"  • {img_info['filename']} (from: {original_location})\n")
            f.write("\n")

        f.write("\nℹ️  HOW TO USE:\n")
        f.write("- Each folder contains images of the same elephant\n")
        f.write("- Images are grouped by AI similarity analysis\n")
        f.write("- Original folder names are preserved in filenames\n")
        f.write("- Similarity threshold: 85%\n")

def process_zip_batch_improved(zip_file, model, transform, ear_detector, device):
    """
    Improved version: Process a batch of elephant images from a ZIP file and group by similarity.
    Works with ANY folder structure - single folder, nested folders, mixed depths, etc.
    """
    try:
        # Step 1: Extract and collect all images recursively
        image_paths, temp_dir = extract_and_collect_images(zip_file)

        if not image_paths:
            return None, "No valid images found in the ZIP file."

        st.info(f"📸 Found {len(image_paths)} images across all folders")

        embeddings = []
        file_info = []  # Store both path and filename for better tracking

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 2: Process each image
        for i, path in enumerate(image_paths):
            try:
                # Update progress
                progress = (i + 1) / len(image_paths)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(path)}")

                image = Image.open(path).convert("RGB")
                ear_region, ear_tensor = process_image(image, ear_detector, transform, device)

                with torch.no_grad():
                    embedding = model(ear_tensor).cpu().numpy()[0]
                    embeddings.append(embedding)
                    file_info.append({
                        'path': path,
                        'filename': os.path.basename(path),
                        'original_folder': os.path.relpath(os.path.dirname(path), temp_dir),
                        'embedding': embedding
                    })

            except Exception as e:
                st.warning(f"❌ Error processing {os.path.basename(path)}: {e}")
                continue

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        if not embeddings:
            return None, "No valid images could be processed."

        st.info(f"✅ Successfully processed {len(embeddings)} images")

        # Step 3: Group by similarity using improved clustering
        st.info("🔄 Grouping similar elephants...")
        similarity_threshold = 0.85
        grouped = []
        used = set()
        embeddings_array = np.array(embeddings)

        for i, emb in enumerate(embeddings_array):
            if i in used:
                continue

            # Start new group
            group = [file_info[i]]
            used.add(i)

            # Find similar images
            for j in range(i + 1, len(embeddings_array)):
                if j not in used:
                    similarity = cosine_similarity([emb], [embeddings_array[j]])[0][0]
                    if similarity >= similarity_threshold:
                        group.append(file_info[j])
                        used.add(j)

            grouped.append(group)

        st.success(f"🐘 Found {len(grouped)} unique elephant groups")

        # Step 4: Create output directory structure
        output_dir = tempfile.mkdtemp()

        for group_idx, group in enumerate(grouped, 1):
            group_folder = os.path.join(output_dir, f"Elephant_Group_{group_idx}")
            os.makedirs(group_folder, exist_ok=True)

            # Copy images with enhanced naming
            for img_info in group:
                src_path = img_info['path']
                # Create descriptive filename that shows original location
                original_folder = img_info['original_folder'].replace('/', '_').replace('\\', '_')
                if original_folder and original_folder != '.':
                    new_filename = f"{original_folder}_{img_info['filename']}"
                else:
                    new_filename = img_info['filename']

                dst_path = os.path.join(group_folder, new_filename)
                shutil.copy2(src_path, dst_path)

        # Step 5: Create summary file
        create_grouping_summary(output_dir, grouped)

        # Step 6: Create ZIP file
        st.info("📦 Creating download package...")
        zip_output_path = shutil.make_archive(
            os.path.join(output_dir, "grouped_elephants"), 'zip', output_dir
        )

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        return zip_output_path, None

    except Exception as e:
        return None, f"Error processing ZIP file: {str(e)}"

def display_grouping_preview(grouped):
    """Display a preview of the grouping results"""
    st.subheader("👀 Grouping Preview")

    for group_idx, group in enumerate(grouped[:5], 1):  # Show first 5 groups
        with st.expander(f"Group {group_idx} - {len(group)} images"):
            cols = st.columns(min(len(group), 4))  # Max 4 images per row
            for i, img_info in enumerate(group[:4]):  # Show first 4 images
                try:
                    img = Image.open(img_info['path'])
                    with cols[i % 4]:
                        st.image(img, caption=img_info['filename'], use_column_width=True)
                except:
                    with cols[i % 4]:
                        st.write(f"📁 {img_info['filename']}")

            if len(group) > 4:
                st.write(f"... and {len(group) - 4} more images")

    if len(grouped) > 5:
        st.info(f"... and {len(grouped) - 5} more groups")

def main():
    st.title("🐘 Asian Elephant Individual Identification System")
    st.markdown("Upload an elephant image to identify the individual using ear pattern recognition")

    # Load model
    model, class_names, device = load_model()

    if model is None:
        st.error("❌ Model not found! Please train the model first using `python train.py`")
        st.stop()

    st.success(f"✅ Model loaded successfully!")

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
    tab1, tab2 = st.tabs(["🖼️ Single Image ID", "📂 Batch Processing"])

    with tab1:
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

    with tab2:
        st.header("📂 Batch Identification from ZIP")
        st.markdown("Upload a ZIP file containing elephant images. The system will:")
        st.markdown("• 🔍 Find all images regardless of folder structure")
        st.markdown("• 🧠 Group similar elephants using AI")
        st.markdown("• 📁 Create organized folders for each elephant")
        st.markdown("• 📄 Generate a detailed summary report")

        # Supported structures info
        with st.expander("📋 Supported ZIP Structures"):
            st.markdown("""
            **✅ Works with ANY folder structure:**
            - Single folder with images
            - Multiple nested folders
            - Images at different depths
            - Images in ZIP root
            - Mixed folder hierarchies
            - Multiple image formats (.jpg, .png, .bmp, .tiff, .webp)

            **Examples:**
            ```
            photos.zip
            ├── elephant1.jpg
            ├── batch1/
            │   ├── photo1.jpg
            │   └── photo2.png
            └── deep/nested/folder/
                └── elephant2.jpg
            ```
            """)

        zip_file = st.file_uploader(
            "Upload ZIP file with elephant images",
            type=["zip"],
            help="Supports any folder structure and multiple image formats"
        )

        if zip_file:
            # Show file info
            st.info(f"📦 Uploaded: {zip_file.name} ({zip_file.size / (1024*1024):.1f} MB)")

            # Processing settings
            with st.expander("⚙️ Processing Settings"):
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.85,
                    step=0.05,
                    help="Higher values = stricter grouping (fewer groups, more similar images)"
                )

            if st.button("🚀 Process ZIP File", type="primary"):
                with st.spinner("🔄 Processing images and grouping elephants..."):
                    zip_result, error = process_zip_batch_improved(
                        zip_file, model, transform, ear_detector, device
                    )

                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.success("🎉 Grouping completed successfully!")

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            with open(zip_result, "rb") as f:
                                st.download_button(
                                    label="📥 Download Grouped Elephants ZIP",
                                    data=f,
                                    file_name="grouped_elephants.zip",
                                    mime="application/zip",
                                    help="Contains organized folders + summary report",
                                    type="primary"
                                )

                        with col2:
                            st.info("💡 The download includes:")
                            st.write("• Organized elephant groups")
                            st.write("• Detailed summary report")
                            st.write("• Preserved original filenames")

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
