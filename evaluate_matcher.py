import torch
from model import SiameseNetwork
from PIL import Image
import torchvision.transforms as T
import numpy as np

def load_img(p): return T.Compose([T.Resize((128,128)),T.ToTensor()])(Image.open(p).convert('L'))

def match(imgA_path, imgB_path, model, thresh=0.5):
    a = load_img(imgA_path).unsqueeze(0).to(device)
    b = load_img(imgB_path).unsqueeze(0).to(device)
    with torch.no_grad():
        e1,e2 = model(a,b)
        dist = torch.norm(e1-e2, dim=1).item()
    return dist < thresh, dist

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load('siamese_matcher.pt'))
model.eval()

# Example test
is_match, score = match('Test/3_recon.png','Test/3_org.png', model)
print(f"Match? {is_match}, distance {score:.4f}")