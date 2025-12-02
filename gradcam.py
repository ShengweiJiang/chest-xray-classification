import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("chest_xray_model.pth", map_location=device))
model = model.to(device)
model.eval()

# 存储中间层输出
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# 注册hook到最后一个卷积层
target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(image_path):
    # 清空之前的记录
    features.clear()
    gradients.clear()
    
    # 加载图像
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # 前向传播
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    pred_prob = F.softmax(output, dim=1)[0][pred_class].item()
    
    # 反向传播
    model.zero_grad()
    output[0, pred_class].backward()
    
    # 计算Grad-CAM
    grads = gradients[0].cpu().data.numpy()[0]
    fmap = features[0].cpu().data.numpy()[0]
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = np.uint8(255 * cam)
    cam = np.array(Image.fromarray(cam).resize((224, 224)))
    
    # 显示结果
    classes = ["NORMAL", "PNEUMONIA"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    img_resized = img.resize((224, 224))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")
    
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    axes[2].imshow(img_resized)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Prediction: {classes[pred_class]} ({pred_prob:.1%})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=150)
    plt.show()
    print(f"\nPrediction: {classes[pred_class]} ({pred_prob:.1%})")
    print("Result saved to gradcam_result.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认用测试集的一张图
        test_image = "chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"
        print(f"Using default test image: {test_image}")
    else:
        test_image = sys.argv[1]
    generate_gradcam(test_image)

