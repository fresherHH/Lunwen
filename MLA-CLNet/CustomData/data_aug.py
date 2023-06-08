from albumentations.augmentations.transforms import VerticalFlip, HorizontalFlip
from skimage import io
import os
from tqdm import tqdm
from random import choice

path_img = r'C:\Users\HH\Desktop\SgCasia2\\train\img'
path_mask = r'C:\Users\HH\Desktop\SgCasia2\\train\mask'

for i in tqdm(os.listdir(path_img)):
    name = i.split('.')[0]
    img = os.path.join(path_img, i)
    mask = os.path.join(path_mask, '{}_gt.png'.format(name))
    image = io.imread(img)
    mask = io.imread(mask)

    list_quality = [50, 60, 70, 80, 90]
    quality = choice(list_quality)
    io.imsave(os.path.join(path_img, '{}_q.jpg'.format(name)), image, quality=quality)
    io.imsave(os.path.join(path_mask, '{}_q_gt.png'.format(name)), mask)

    whatever_data = "my name"

    augmentation = VerticalFlip(p=1.0)
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = augmentation(**data)
    image_ver, mask_ver, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
    io.imsave(os.path.join(path_img, '{}_ver.jpg'.format(name)), image_ver, quality=100)
    io.imsave(os.path.join(path_mask, '{}_ver_gt.png'.format(name)), mask_ver)

    augmentation = HorizontalFlip(p=1.0)
    data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
    augmented = augmentation(**data)
    image_hor, mask_hor, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
    io.imsave(os.path.join(path_img, '{}_hor.jpg'.format(name)), image_hor, quality=100)
    io.imsave(os.path.join(path_mask, '{}_hor_gt.png'.format(name)), mask_hor)

