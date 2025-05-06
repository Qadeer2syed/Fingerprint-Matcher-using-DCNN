import os, random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

def load_image(path):
    return T.Compose([T.Resize((128,128)), T.ToTensor()])(Image.open(path).convert("L"))

class FingerprintPairDataset(Dataset):
    def __init__(self, root_dir, users, transform=None, pos_ratio=0.5):
        """
        root_dir: base data folder
        users: list of user IDs to include
        pos_ratio: fraction of positive pairs
        """
        self.transform = transform or (lambda x: x)
        # map user -> list of image paths
        self.user_images = {u: sorted(os.listdir(os.path.join(root_dir,u))) for u in users}
        self.root = root_dir
        self.users = users
        self.pos_ratio = pos_ratio
        # precompute all pairs? Or sample on the fly
        self.pairs = []
        # we'll sample on the fly in __getitem__

    def __len__(self):
        # define arbitrary length
        return 100000

    def __getitem__(self, idx):
        if random.random() < self.pos_ratio:
            # positive pair
            user = random.choice(self.users)
            imgs = random.sample(self.user_images[user], 2)
            label = 1
        else:
            # negative pair
            user1, user2 = random.sample(self.users, 2)
            imgs = [random.choice(self.user_images[user1]), random.choice(self.user_images[user2])]
            label = 0
        img1 = load_image(os.path.join(self.root, user1 if label==0 else user, imgs[0]))
        img2 = load_image(os.path.join(self.root, user2 if label==0 else user, imgs[1]))
        return self.transform(img1), self.transform(img2), torch.tensor(label, dtype=torch.float)