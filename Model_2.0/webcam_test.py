import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms

# ================= CONFIG =================
MODEL_PATH = "./best_afew_va.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROP_OUT = 0.2
TAU = 0.07  # neutral threshold for valence
IMG_SIZE = 224
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
    filtered_state = {k: v for k, v in state.items()
                      if k in model_dict and model_dict[k].shape == v.shape}
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
def emotion(valence, tau=TAU):
    if valence > tau:
        return "Positive"
    elif valence < -tau:
        return "Negative"
    else:
        return "Neutral"


# ============== LOAD MODEL =================
model = build_model(MODEL_PATH, DEVICE)

# ============== WEBCAM LOOP ================
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess and add batch dimension
    x = transform(rgb).unsqueeze(0).to(DEVICE)

    # Model inference
    with torch.no_grad():
        pred = model(x)
        valence, arousal = pred[0].cpu().numpy()

    # Map valence to emotion
    emo = emotion(valence)

    # Overlay prediction
    text = f"Emotion: {emo} | Valence: {valence:.2f} | Arousal: {arousal:.2f}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
