import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
import argparse


def load_model(weights_path, device):
    model = torchvision.models.convnext_tiny(weights=None)
    ckpt = torch.load(weights_path, map_location='cpu')
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    
    features_state = {k.replace('features.', ''): v for k, v in state.items() if k.startswith('features.')}
    classifier_state = {k.replace('classifier.', ''): v for k, v in state.items() if k.startswith('classifier.')}
    
    model.features.load_state_dict(features_state)
    
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LayerNorm(768, eps=1e-6),
        torch.nn.Linear(768, 512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 2)
    )
    
    model.classifier.load_state_dict(classifier_state)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame, device):
    frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = (tensor - mean) / std
    
    return tensor.unsqueeze(0).to(device)


def classify_emotion(valence, arousal):
    v_threshold = 7.15
    a_threshold = 4.68
    
    if valence >= v_threshold:
        if arousal >= a_threshold:
            return "Happy & Excited", (0, 255, 0)
        else:
            return "Content & Calm", (0, 200, 0)
    else:
        if arousal >= a_threshold:
            return "Angry & Tense", (0, 0, 255)
        else:
            return "Sad & Melancholic", (100, 0, 150)


def draw_emotion_overlay(frame, valence, arousal, emotion, color):
    h, w = frame.shape[:2]
    
    cv2.rectangle(frame, (10, 10), (450, 180), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (450, 180), color, 3)
    
    cv2.putText(frame, f"Valence: {valence:.2f}/10", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"Arousal: {arousal:.2f}/10", (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, emotion, (25, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.line(frame, (10, h//2), (w-10, h//2), (100, 100, 100), 2)
    cv2.line(frame, (w//2, 10), (w//2, h-10), (100, 100, 100), 2)
    
    cv2.putText(frame, "Positive Valence", (w//2 + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, "Negative Valence", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, "High Arousal", (w-200, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(frame, "Low Arousal", (w-150, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return frame


def main(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    print("Loading model...")
    model = load_model(args.weights_path, device)
    print("Model ready. Opening camera...")
    
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera_id}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    print("Live emotion detection - Press 'q' to quit, 's' to save frame")
    print("="*60)
    
    frame_count = 0
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            input_tensor = preprocess_frame(frame, device)
            output = model(input_tensor)
            pred = (output.cpu().numpy()[0] * 9.0 + 1.0)
            
            valence = float(pred[0])
            arousal = float(pred[1])
            emotion, color = classify_emotion(valence, arousal)
            
            frame_display = draw_emotion_overlay(frame, valence, arousal, emotion, color)
            cv2.imshow("Emotion Detection", frame_display)
            
            frame_count += 1
            if frame_count % 15 == 0:
                print(f"Frame {frame_count}: V={valence:.2f} A={arousal:.2f} | {emotion}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"emotion_capture_{frame_count}.jpg"
                cv2.imwrite(filename, frame_display)
                print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="./checkpoints/best_head_ls.pt")
    parser.add_argument("--camera_id", type=int, default=0)
    args = parser.parse_args()
    
    main(args)
