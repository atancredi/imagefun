import hashlib
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from skimage import color
from sklearn.preprocessing import normalize
import umap

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocess for resnet50
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def setup_model():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    return model, device


def file_hash(path):
    """Return MD5 hash of file or None if not readable."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def image_vector(image: Image.Image, model):
    img = image.convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)
    feat = None
    with torch.no_grad():
        feat = model(tensor).squeeze().cpu().numpy()
    return feat


def palette_vector(palette_rgb):

    palette = np.array(palette_rgb, dtype=np.float32)
    # norm
    if palette.max() > 1.0:
        palette /= 255.0
        
    # convert rgb to lab
    # transform shape (1,4,3) to (4,3)
    # 4 colors, 3 channels
    palette_lab = color.rgb2lab(palette[np.newaxis, :, :])[0]
    
    # sort by luminance to align the vectors
    sorted_indices = np.argsort(palette_lab[:, 0]) 
    sorted_palette = palette_lab[sorted_indices]
    
    # flatten dimensions to 3*4
    flat_vector = sorted_palette.flatten()
    
    # l2 normalization - needed for using cosine similarity
    norm_vector = normalize(flat_vector.reshape(1, -1), norm='l2').flatten()
    
    return norm_vector


def full_image_vector(resnet_vecs, palette_vecs, alpha_mix = 0.7):
    # L2 norm
    resnet_vecs = resnet_vecs.reshape(-1, 1)
    resnet_norm = normalize(resnet_vecs, norm='l2')
    resnet_norm = resnet_norm.reshape(1, -1)

    palette_vecs = palette_vecs.reshape(-1, 1)
    palette_norm = normalize(palette_vecs, norm='l2')
    palette_norm = palette_norm.reshape(1, -1)
    
    # weighted mix
    combined_features = np.hstack([alpha_mix * resnet_norm, (1 - alpha_mix) * palette_norm])

    # umap dimensionality reduction to 10 dim
    reducer = umap.UMAP(n_neighbors=30, n_components=10, min_dist=0.0, metric='cosine')
    full_features = reducer.fit_transform(combined_features)
    full_features = full_features.squeeze()

    return full_features
