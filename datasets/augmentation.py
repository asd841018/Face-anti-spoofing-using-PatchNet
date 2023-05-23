import random
import cv2
import math
import numpy as np
import torch

# array
class RandomResize(object):
    def __init__(self, img_shape=None, depth_shape=None):
        self.img_shape = img_shape
        self.depth_shape = depth_shape
    
    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        img_h, img_w = img.shape[0], img.shape[1]
        img_start_x = int(np.random.randint((img_w-self.img_shape[1]), size=1))  # 起始的寬
        img_start_y = int(np.random.randint((img_h-self.img_shape[0]), size=1))  # 起始的高
        
        ratio_x = img_start_x / img_w
        ratio_y = img_start_y / img_h
        ratio_w = self.img_shape[1] / img_w
        ratio_h = self.img_shape[0] / img_h
        
        depth_h, depth_w = map_x.shape[0], map_x.shape[1]
        depth_start_x = round(depth_w * ratio_x)  # 起始的寬
        depth_start_y = round(depth_h * ratio_y)  # 起始的高
        
        length_w = round(depth_w*(self.img_shape[1] / img_w))
        length_h = round(depth_h*(self.img_shape[0] / img_h))  
        
        new_depth = map_x[depth_start_y:depth_start_y+length_h, depth_start_x:depth_start_x+length_w]
        new_depth = cv2.resize(new_depth, self.depth_shape)
        
        new_img = img[img_start_y:img_start_y+self.img_shape[0], img_start_x:img_start_x+self.img_shape[1], :]
        return {'image_x': new_img, 'map_x': new_depth, 'spoofing_label': spoofing_label}



class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']

        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]

        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_map_x = map_x/255.0                 # [0,1]
        return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}


class RandomRotate90(object):
    """Rotate 90 degree the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))
        p = random.random()
        if p < 0.5:
            q = random.random()
            if q < 0.5:
                # 順時鐘90度
                new_image_x = cv2.rotate(image_x, cv2.ROTATE_90_CLOCKWISE)  
                new_map_x = cv2.rotate(map_x, cv2.ROTATE_90_CLOCKWISE)
                return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
            else:
                # 逆時鐘90度
                new_image_x = cv2.rotate(image_x, cv2.ROTATE_90_COUNTERCLOCKWISE)  
                new_map_x = cv2.rotate(map_x, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}


class RandomVerticalFlip(object):
    """Vertically flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 0)
            new_map_x = cv2.flip(map_x, 0)


            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)


            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']

        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)

        map_x = np.array(map_x)


        spoofing_label_np = np.array([0],dtype=np.compat.long)
        spoofing_label_np[0] = spoofing_label


        return {'image_x': torch.from_numpy(image_x.astype(float)).float(), 'map_x': torch.from_numpy(map_x.astype(float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.compat.long)).long()}