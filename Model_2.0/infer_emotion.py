import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
import argparse
import os

# ================= CONFIG =================
MODEL_PATH = "./best_afew_va.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
DEFAULT_TAU = 0.08
# ==========================================


# ============== MODEL =====================
def build_model(weights_path, device, dropout_p=0.2):
    model = torchvision.models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(512, 2)
    )

    state = torch.load(weights_path, map_location="cpu")
    model_dict = model.state_dict()
    filtered_state = {
        k: v for k, v in state.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(filtered_state)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    return model


# ============== PREPROCESS =================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============== EMOTION MAPPING =============
def emotion_from_valence(valence, tau):
    if valence > tau:
        return "Positive"
    elif valence < -tau:
        return "Negative"
    else:
        return "Neutral"


# ============== INFERENCE ==================
def infer_image(image_path, model, device, tau=DEFAULT_TAU):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        valence, arousal = pred[0].cpu().numpy()

    emotion = emotion_from_valence(valence, tau)

    return {
        "valence": float(valence),
        "arousal": float(arousal),
        "emotion": emotion
    }


# ============== CLI ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion inference on a single image")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = parser.parse_args()

    model = build_model(MODEL_PATH, DEVICE)
    result = infer_image(args.image_path, model, DEVICE, tau=args.tau)

    print(f"Valence : {result['valence']:.4f}")
    print(f"Arousal : {result['arousal']:.4f}")
    print(f"Emotion : {result['emotion']}")
