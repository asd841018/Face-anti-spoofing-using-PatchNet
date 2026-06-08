import json
from pathlib import Path
cwd = Path(__file__).parent.absolute()
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def uniform_crop(img, grid=3):
    """Split an image into ``grid x grid`` non-overlapping patches.

    For the default ``grid=3`` this yields the 9 patches used at inference time,
    ordered left-to-right then top-to-bottom. The last row/column absorbs any
    remainder so the patches always tile the whole image.

    Args:
        img: HxWxC numpy image.
        grid: number of splits per axis (3 -> 9 patches).

    Returns:
        A list of ``grid * grid`` patch crops (numpy arrays).
    """
    h, w = img.shape[:2]
    patch_h, patch_w = round(h / grid), round(w / grid)
    patches = []
    for i in range(grid):
        for j in range(grid):
            y0, x0 = i * patch_h, j * patch_w
            # last row/column extends to the image border (matches the old 2*p: slices)
            y1 = (i + 1) * patch_h if i < grid - 1 else h
            x1 = (j + 1) * patch_w if j < grid - 1 else w
            patches.append(img[y0:y1, x0:x1, :])
    return patches


class AMOuluDataset(Dataset):
    def __init__(self, root_folder, mode=None, transform=None):
        self.root_folder = root_folder
        if mode == 'train':
            list_path = os.path.join(root_folder, 'metas/AM_train_item.json')
        elif mode == 'valid':
            list_path = os.path.join(root_folder, 'metas/AM_val_item.json')
        else:
            list_path = os.path.join(root_folder, 'metas/test_item.json')

        with open(list_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[str(idx)]
        # read image and convert BGR (OpenCV) -> RGB
        img = cv2.imread(os.path.join(self.root_folder, data_item['path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # fine-grained label (0~5 = live, 6~29 = spoof); binary live/spoof flag
        label = data_item['label']
        binary = int(label > 5)  # 0 = live, 1 = spoof

        if self.mode == 'train':
            # Two augmented views of the SAME image: one is classified by the
            # AM-Softmax head, and the pair feeds the self-supervised similarity loss.
            view1 = self.transform(image=img)['image']
            view2 = self.transform(image=img)['image']
            return view1, view2, label, binary

        # valid / test: crop 9 uniform patches and transform each one.
        # Stacked into [num_patches, C, H, W] so the eval loop can score the
        # whole image in a single batched forward pass.
        patches = [self.transform(image=p)['image'] for p in uniform_crop(img)]
        return torch.stack(patches), label, binary

if __name__ == "__main__":
    transform = A.Compose(
                    [
                    A.Resize(height=160, width=160, p=1.0),
                    A.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensor(),
                    ]
                )
    train_data = AMOuluDataset(root_folder='/mnt/training_dataset/face_dataset/Oulu_align',\
                            mode='valid',\
                            transform=transform)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    # valid/test yields stacked patches: [batch, num_patches, C, H, W]
    for i, (patches, labels, binary) in enumerate(train_loader):
        print(patches.shape)
        print(labels, binary)
        break
