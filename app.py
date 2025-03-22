from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms, models
from PIL import Image
import io
import warnings

app = Flask(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# Load the model
model_path = "mobilenetv2_finetuned.pth"
device = torch.device("cpu")
model = models.mobilenet_v2(pretrained=False)

num_classes = 38
model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=num_classes)
print("Loading model...")
model.load_state_dict(torch.load(model_path, map_location=device))
print("Model loaded successfully.")

model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Class labels
class_labels = [
    "Apple Apple Scab", "Apple Black Rot", "Apple Cedar Apple Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry (Including Sour) Powdery Mildew", "Cherry (Including Sour) Healthy",
    "Corn (Maize) Cercospora Leaf Spot Gray Leaf Spot", "Corn (Maize) Common Rust",
    "Corn (Maize) Northern Leaf Blight", "Corn (Maize) Healthy", "Grape Black Rot",
    "Grape Esca (Black Measles)", "Grape Leaf Blight (Isariopsis Leaf Spot)", "Grape Healthy",
    "Orange Haunglongbing (Citrus Greening)", "Peach Bacterial Spot", "Peach Healthy",
    "Pepper, Bell Bacterial Spot", "Pepper, Bell Healthy", "Potato Early Blight",
    "Potato Late Blight", "Potato Healthy", "Raspberry Healthy", "Soybean Healthy",
    "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
    "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites Two-Spotted Spider Mite",
    "Tomato Target Spot", "Tomato Tomato Yellow Leaf Curl Virus", "Tomato Tomato Mosaic Virus",
    "Tomato Healthy"
]

# Predict function
def predict_image(image_bytes):
    """Predict the plant disease for a given image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure RGB mode
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image.to(device))
            predicted_class = torch.argmax(output, dim=1).item()

        return class_labels[predicted_class]
    except Exception as e:
        print("Error in prediction:", str(e))
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/solution')
def solution():
    return render_template('solution.html')

# Flask route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    print("File received:", file.filename)
    print("Image size:", len(image_bytes), "bytes")

    predicted_label = predict_image(image_bytes)
    if predicted_label:
        print("Predicted label:", predicted_label)
        return jsonify({"prediction": predicted_label})
    else:
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
