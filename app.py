import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 页面设置
st.set_page_config(page_title="Chest X-Ray Classifier", layout="wide")
st.title("🫁 Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to detect pneumonia")

# 加载模型
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("chest_xray_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Grad-CAM
features, gradients = [], []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def predict_with_gradcam(img):
    features.clear()
    gradients.clear()
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_prob = F.softmax(output, dim=1)[0][pred_class].item()
    
    model.zero_grad()
    output[0, pred_class].backward()
    
    grads = gradients[0].cpu().data.numpy()[0]
    fmap = features[0].cpu().data.numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = np.uint8(255 * cam)
    cam = np.array(Image.fromarray(cam).resize((224, 224)))
    
    return pred_class, pred_prob, cam

# 上传图片
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original X-Ray")
        st.image(img, use_container_width=True)
    
    with st.spinner("Analyzing..."):
        pred_class, pred_prob, cam = predict_with_gradcam(img)
    
    classes = ["NORMAL", "PNEUMONIA"]
    result = classes[pred_class]
    
    with col2:
        st.subheader("Grad-CAM Heatmap")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(cam, cmap="jet")
        ax.axis("off")
        st.pyplot(fig)
    
    with col3:
        st.subheader("Overlay")
        img_resized = img.resize((224, 224))
        fig, ax = plt.subplots()
        ax.imshow(img_resized)
        ax.imshow(cam, cmap="jet", alpha=0.5)
        ax.axis("off")
        st.pyplot(fig)
    
    # 结果
    st.markdown("---")
    if result == "PNEUMONIA":
        st.error(f"### Prediction: {result} ({pred_prob:.1%})")
    else:
        st.success(f"### Prediction: {result} ({pred_prob:.1%})")
    
    st.info("⚠️ This is for educational purposes only. Always consult a medical professional.")

