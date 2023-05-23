import json
from pathlib import Path
cwd = Path(__file__).parent.absolute()
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor

class uniform_crop(object):
    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        patch_h, patch_w = round(h/3), round(w/3)
        # top
        img_1 = img[0:patch_h, 0:patch_w, :]
        img_2 = img[0:patch_h, patch_w:2*patch_w, :]
        img_3 = img[0:patch_h, 2*patch_w:, :]
        # middle
        img_4 = img[patch_h:2*patch_h, 0:patch_w, :]
        img_5 = img[patch_h:2*patch_h, patch_w:2*patch_w, :]
        img_6 = img[patch_h:2*patch_h, 2*patch_w:, :]
        # bottom
        img_7 = img[2*patch_h:, 0:patch_w, :]
        img_8 = img[2*patch_h:, patch_w:2*patch_w, :]
        img_9 = img[2*patch_h:, 2*patch_w:, :]
        return img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8, img_9


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
        # input
        img = cv2.imread(os.path.join(self.root_folder, data_item['path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label
        label = data_item['label']
        if label <= 5:
            binary = 0
        else:
            binary = 1
        if self.mode == 'train':
            if self.transform:
                img1 = self.transform(image=img)['image']
                img2 = self.transform(image=img)['image']
            return img1, img2, label, binary
        else:
            trans = uniform_crop()
            img1, img2, img3, img4, img5, img6, img7, img8, img9 = trans(img=img)
            if self.transform:
                img1 = self.transform(image=img1)['image']
                img2 = self.transform(image=img2)['image']
                img3 = self.transform(image=img3)['image']
                img4 = self.transform(image=img4)['image']
                img5 = self.transform(image=img5)['image']
                img6 = self.transform(image=img6)['image']
                img7 = self.transform(image=img7)['image']
                img8 = self.transform(image=img8)['image']
                img9 = self.transform(image=img9)['image']
            return [img1, img2, img3, img4, img5, img6, img7, img8, img9], label, binary

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
    for i, (img1, img2, img3, img4, img5, img6, img7, img8, img9, labels, binary) in enumerate(train_loader):
        print(img1.shape)
        print(img2.shape)
        print(img3.shape)
        print(img4.shape)
        print(img5.shape)
        print(img6.shape)
        print(img7.shape)
        print(img8.shape)
        print(img9.shape)
