import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import gdown

# -----------------------------
# Generator Architecture
# -----------------------------
class Generator(nn.Module):
    def __init__(self, img_channels=3, label_dim=10, feature_dim=64):
        super(Generator, self).__init__()
        self.label_dim = label_dim
        in_channels = img_channels + label_dim
        nf = feature_dim
        self.down1 = nn.Conv2d(in_channels, nf, kernel_size=7, stride=1, padding=3)
        self.down2 = nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1)

        res_blocks = []
        res_channels = nf * 4
        for _ in range(6):
            res_blocks.append(nn.Sequential(
                nn.Conv2d(res_channels, res_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(res_channels, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(res_channels, res_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(res_channels, affine=False)
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.up1 = nn.ConvTranspose2d(res_channels, nf*2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(nf, img_channels, kernel_size=7, stride=1, padding=3)

        self.actvn = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)

    def forward(self, img, target_label):
        label_maps = target_label[:, :, None, None]
        label_maps = label_maps.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, label_maps], dim=1)
        x = self.actvn(self.down1(x))
        x = self.actvn(self.down2(x))
        x = self.actvn(self.down3(x))
        for res_block in self.res_blocks:
            residual = x
            out = res_block(x)
            x = self.actvn(out + residual)
        x = self.actvn(self.up1(x))
        x = self.actvn(self.up2(x))
        x = self.tanh(self.out_conv(x))
        return x

# -----------------------------
# Load model checkpoint
# -----------------------------
@st.cache_resource
def load_generator(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(img_channels=3, label_dim=10, feature_dim=64)
    state_dict = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state_dict)
    G.to(device)
    G.eval()
    return G, device

def download_checkpoint():
    url = "https://drive.google.com/uc?id=15FZADGaAUwj1tmHRwGYOAxqFkhuDL5_p"
    output = "generator_epoch_60.pth"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return output

# -----------------------------
# Preprocessing
# -----------------------------
transform_ops = transforms.Compose([
    transforms.CenterCrop(200),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def denormalize(tensor_img):
    return tensor_img.mul(0.5).add(0.5)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Age Transformation GAN Demo")

ckpt_path = download_checkpoint()
G, device = load_generator(ckpt_path)

uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
age = st.slider("Target age", min_value=1, max_value=90, value=30)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img_tensor = transform_ops(img).unsqueeze(0).to(device)
    group_idx = min((age - 1) // 10, 9)
    label = torch.tensor([group_idx], device=device)
    one_hot = F.one_hot(label, num_classes=10).float().to(device)
    with torch.no_grad():
        gen_out = G(img_tensor, one_hot)
    out_img = denormalize(gen_out.squeeze(0).cpu())
    pil_img = transforms.ToPILImage()(out_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(pil_img, caption=f"Transformed to age group {group_idx}", use_column_width=True)
