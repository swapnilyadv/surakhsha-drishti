import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# 1. LIGHTWEIGHT SPATIAL-TEMPORAL ARCHITECTURE (MobileNetV3-LSTM)
# ============================================================
class TemporalViolenceClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super().__init__()
        # Load lightweight MobileNetV3 Small as spatial feature extractor
        mobilenet = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        # Extract features (output size is 576 channels)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Freeze spatial backbone parameters to prevent overfitting
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(input_size=576, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Input shape: (B, seq_len, C, H, W) e.g., (B, 16, 3, 160, 160)
        batch_size, seq_len, C, H, W = x.size()
        
        # Collapse batch and temporal dimension for CNN extraction
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        
        # Spatial features: (B * seq_len, 576, 1, 1)
        features = self.pool(self.backbone(x_reshaped))
        features = features.view(batch_size, seq_len, 576)
        
        # LSTM processes the temporal sequence
        lstm_out, _ = self.lstm(features)
        
        # Pull output from the final sequence step: (B, hidden_dim)
        out = lstm_out[:, -1, :]
        
        # Logits: (B, num_classes)
        logits = self.fc(out)
        return logits

# ============================================================
# 2. SEQUENCE LOADER WITH SPECIALIZED DEMO AUGMENTATIONS
# ============================================================
class VideoSequenceDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=16, augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment

    def __len__(self):
        return len(self.video_paths)

    def apply_webcam_and_phone_augmentations(self, frame):
        """Applies advanced augmentations optimizing for phone videos & live demos."""
        h, w, c = frame.shape
        
        # A. Phone Screen Glare Simulation (Radial gradient reflections)
        if random.random() < 0.4:
            cx, cy = random.randint(0, w), random.randint(0, h)
            x_arr = np.arange(w)
            y_arr = np.arange(h)
            xv, yv = np.meshgrid(x_arr, y_arr)
            dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
            # Create a localized glare reflection spotlight
            glare = np.exp(-dist / (w * random.uniform(0.15, 0.45))) * random.uniform(80, 140)
            glare = np.expand_dims(glare, axis=-1)
            frame = np.clip(frame + glare, 0, 255)

        # B. Phone Display Jitter (Color cast scaling)
        if random.random() < 0.4:
            # Shift colors slightly toward colder (blueish) tones to represent LCD screen outputs
            r_scale = random.uniform(0.85, 1.05)
            g_scale = random.uniform(0.9, 1.1)
            b_scale = random.uniform(0.95, 1.15)
            frame = frame.astype(np.float32)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * r_scale, 0, 255)
            frame[:, :, 1] = np.clip(frame[:, :, 1] * g_scale, 0, 255)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * b_scale, 0, 255)

        # C. Webcam Pan/Tilt Directional Motion Blur
        if random.random() < 0.3:
            size = random.choice([3, 5, 7])
            kernel = np.zeros((size, size))
            # Create a diagonal blur vector
            np.fill_diagonal(kernel, 1.0 / size)
            frame = cv2.filter2D(frame.astype(np.uint8), -1, kernel)

        # D. CCTV Low-Light / Darkness Jitter
        if random.random() < 0.3:
            factor = random.uniform(0.45, 0.75)
            frame = np.clip(frame * factor, 0, 255)

        return frame.astype(np.uint8)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Fallback for corrupt headers
        if total_frames <= 0:
            cap.release()
            return torch.zeros((self.sequence_length, 3, 160, 160), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        # Retrieve video FPS (standardize to 30 if unavailable)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30
            
        # Target a 4-second clip segment (120 frames) to speed up loading and keep action tight
        clip_frames = int(fps * 4)
        if total_frames > clip_frames:
            # Slices centered 4 seconds of video
            start_frame = max(0, int(total_frames / 2) - int(clip_frames / 2))
            end_frame = min(total_frames, start_frame + clip_frames)
        else:
            start_frame = 0
            end_frame = total_frames

        # Define 16 target indices uniformly distributed across the 4-second clip
        sampled_indices = np.linspace(start_frame, end_frame - 1, self.sequence_length, dtype=int)
        sampled_set = set(sampled_indices)

        # Ultra-efficient Grab-and-Retrieve loop: decodes ONLY the 16 target frames
        frames_dict = {}
        frame_idx = start_frame
        
        # Move capture cursor to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while frame_idx < end_frame:
            # Grab frame structure (no H264 decode, extremely fast!)
            ret = cap.grab()
            if not ret:
                break
            
            # Decode frame ONLY if index matches target list
            if frame_idx in sampled_set:
                ret_dec, frame = cap.retrieve()
                if ret_dec and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_dict[frame_idx] = frame
            frame_idx += 1
        cap.release()

        # Rebuild final sequence resized to 160x160
        sequence = []
        for i in sampled_indices:
            if i in frames_dict:
                frame = frames_dict[i]
            else:
                # Safe zero fallback
                frame = np.zeros((160, 160, 3), dtype=np.uint8)

            frame = cv2.resize(frame, (160, 160))
            
            if self.augment:
                frame = self.apply_webcam_and_phone_augmentations(frame)

            # Normalize to ImageNet parameters
            frame_norm = frame.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frame_norm = (frame_norm - mean) / std

            # Transpose to shape: (C, H, W)
            frame_transposed = np.transpose(frame_norm, (2, 0, 1))
            sequence.append(frame_transposed)

        sequence_tensor = torch.tensor(np.array(sequence), dtype=torch.float32)
        return sequence_tensor, torch.tensor(label, dtype=torch.long)

# ============================================================
# 3. TRAINING & VALIDATION PIPELINE
# ============================================================
def prepare_data_lists(dataset_dir):
    """Gathers all normal and violence/harassment video paths and labels."""
    video_paths = []
    labels = []
    
    # Standardize classes to binary categorization
    # Support folder names 'violence' and 'harassment' as positive category, 'normal' as negative
    categories = {"normal": 0, "violence": 1, "harassment": 1}
    for category, label in categories.items():
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            continue
        for file in os.listdir(cat_dir):
            if file.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                video_paths.append(os.path.join(cat_dir, file))
                labels.append(label)
                
    return video_paths, labels

def main():
    # Support both backend/../Dataset and root/Dataset
    dataset_dir = "/Users/swapnil/Desktop/my project/surakhsha-drishti/Dataset"
    output_dir = "/Users/swapnil/Desktop/my project/surakhsha-drishti/backend/models"
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 Scanning and preparing dataset...")
    video_paths, labels = prepare_data_lists(dataset_dir)
    total_samples = len(video_paths)
    print(f"📊 Found {total_samples} total video clips: {labels.count(1)} positive (violence/harassment), {labels.count(0)} normal.")

    if total_samples == 0:
        print("❌ Dataset folder empty. Please seed and run again.")
        return

    # Train/Validation Split (80/20)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split = int(0.8 * total_samples)
    
    train_paths = [video_paths[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    val_paths = [video_paths[i] for i in indices[split:]]
    val_labels = [labels[i] for i in indices[split:]]

    print(f"📈 Training set size: {len(train_paths)} clips.")
    print(f"📉 Validation set size: {len(val_paths)} clips.")

    # Create Datasets and DataLoaders
    train_dataset = VideoSequenceDataset(train_paths, train_labels, sequence_length=16, augment=True)
    val_dataset = VideoSequenceDataset(val_paths, val_labels, sequence_length=16, augment=False)

    # Use num_workers=0 and a smaller batch_size=4 to completely eliminate RAM bloat and OOM (Signal 9) failures
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Targeting compute device: {device}")

    model = TemporalViolenceClassifier(hidden_dim=128, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    epochs = 10
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")
    print("🎓 Starting temporal violence classifier training loop...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_preds, train_true = [], []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_preds, val_true, val_probs = [], [], []
        val_running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(targets.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss = val_running_loss / len(val_dataset)
        val_acc = accuracy_score(val_true, val_preds)
        val_prec = precision_score(val_true, val_preds, zero_division=0)
        val_rec = recall_score(val_true, val_preds, zero_division=0)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)

        # Print EXACT format requested
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_prec:.4f}")
        print(f"Recall: {val_rec:.4f}")
        print(f"F1 Score: {val_f1:.4f}")
        print("-" * 30)
        
        scheduler.step(val_acc)

        # Save checkpoint ONLY if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Saved best checkpoint to {best_model_path} with Acc: {val_acc:.4f}\n")

    print("\n🔌 Exporting optimized unified ONNX graph for real-time inference...")
    onnx_model_path = os.path.join(output_dir, "violence_model.onnx")
    
    # Load best state for ONNX export
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Pre-warm with 160x160 representative sequence
    dummy_input = torch.randn(1, 16, 3, 160, 160).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["logits"],
        dynamic_axes={
            "input_sequence": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"✓ Unified ONNX model exported and saved to: {onnx_model_path}")
    print("🎉 Training pipeline operations completed successfully!")

if __name__ == "__main__":
    main()
