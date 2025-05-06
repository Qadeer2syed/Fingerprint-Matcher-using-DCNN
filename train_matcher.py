import torch
from torch.utils.data import DataLoader
from model import SiameseNetwork
from datasets import FingerprintPairDataset
from losses import contrastive_loss
from torch.optim import AdamW
from tqdm import tqdm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
users = [f'user{i}' for i in range(1,9)]  # 8 users for training
train_ds = FingerprintPairDataset('data', users)
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
model = SiameseNetwork().to(device)
opt = AdamW(model.parameters(), lr=1e-3)

# Train
for epoch in range(20):
    total=0
    for x1,x2,y in tqdm(loader):
        x1,x2,y = x1.to(device), x2.to(device), y.to(device)
        e1,e2 = model(x1,x2)
        loss = contrastive_loss(e1,e2,y)
        opt.zero_grad(); loss.backward(); opt.step()
        total+=loss.item()
    print(f"Epoch {epoch+1} - Loss {total/len(loader):.4f}")
    torch.save(model.state_dict(), 'siamese_matcher.pt')
# Save model
