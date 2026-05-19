import os
import torch
import torch.nn as nn
import torchvision

class TemporalViolenceClassifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
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
        x_reshaped = x.view(batch_size * seq_len, C, H, W)
        features = self.pool(self.backbone(x_reshaped))
        features = features.view(batch_size, seq_len, 576)
        lstm_out, _ = self.lstm(features)
        out = lstm_out[:, -1, :] # Pull final recurrent state
        logits = self.fc(out)
        return logits

def compile_and_export():
    output_dir = "/Users/swapnil/Desktop/my project/surakhsha-drishti/backend/models"
    os.makedirs(output_dir, exist_ok=True)
    
    pth_path = os.path.join(output_dir, "best_model.pth")
    onnx_path = os.path.join(output_dir, "violence_model.onnx")
    
    print("🚀 Instantiating MobileNetV3-LSTM Temporal graph...")
    model = TemporalViolenceClassifier(hidden_dim=128, num_classes=2)
    model.eval()
    
    # Save initial PyTorch weights checkpoint
    torch.save(model.state_dict(), pth_path)
    print(f"✓ PyTorch model weights saved to: {pth_path}")
    
    # Generate representative dummy inputs matching (B, seq_length, C, H, W)
    dummy_input = torch.randn(1, 16, 3, 160, 160)
    
    print("🔌 Compiling and exporting unified ONNX graph at 160x160...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14, # Opset 14 fully supports LSTM execution blocks
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["logits"],
        dynamic_axes={
            "input_sequence": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print(f"🏆 Unified ONNX temporal graph successfully saved to: {onnx_path}")

if __name__ == "__main__":
    compile_and_export()
