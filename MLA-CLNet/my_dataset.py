import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()

        self.flag = "train" if train else "val"
        # data_root = os.path.join(root, "SgCasia", self.flag)
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "img")) if i.endswith(".jpg") or i.endswith(".tif")]

        self.img_list = [os.path.join(data_root, "img", i) for i in img_names]


        # self.roi_mask = [os.path.join(data_root, "mask", i.split(".")[0] + f"_gt.png")
        #                for i in img_names]
        # # check files
        # for i in self.roi_mask:
        #     if os.path.exists(i) is False:
        #         raise FileNotFoundError(f"file {i} does not exists.")

        #
        # self.manual =  [os.path.join(data_root, "mask", i.split(".")[0] + f"_gt.png")
        #                  for i in img_names]
        self.manual = [os.path.join(data_root, "mask", i.split(".")[0][:-1] + f"forged.tif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")



    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        manual = Image.open(self.manual[idx]).convert('L')
        manual = np.array(manual) / 255

        # roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # roi_mask = 255 - np.array(roi_mask)

        mask = np.clip(manual, a_min=0, a_max=255)
        # print(mask)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__== "__main__":

    # data_root = os.path.join(r"C:\Users\HH\Desktop\SgCasia2", "train")
    data_root = r"C:\Users\HH\Desktop\Dataset\Coverage\\train\\"
    assert os.path.exists(data_root), f"path '{data_root}' does not exists."

    img_names = [i for i in os.listdir(os.path.join(data_root, "img")) if i.endswith(".jpg") or i.endswith(".tif")]

    img_list = [os.path.join(data_root, "img", i) for i in img_names]

    mask = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith(".tif")]

    mask_names = [i for i in os.listdir(os.path.join(data_root, "mask")) if i.endswith(".jpg") or i.endswith(".tif")]

    for i in mask_names:
        if not i.endswith("d.tif"):
            file_name = os.path.join(r"C:\Users\HH\Desktop\Dataset\Coverage\mask", i)
            os.remove(file_name)


    # print(img_names[:10])
    # print(mask[:10])
    # cnt = 0
    # for i in img_names:
    #     flag = False
    #     for j in mask:
    #         if j.split("_gt")[0] == i.split(".")[0]:
    #             flag = True
    #     if flag == False:
    #         cnt += 1
    #         file_name = os.path.join(r"C:\Users\HH\Desktop\SgCasia2\train\img", i)
    #         os.remove(file_name)
    #         print(i)
    #
    # for i in mask:
    #     flag = False
    #     for j in img_names:
    #         if j.split(".")[0] == i.split("_gt")[0]:
    #             flag = True
    #     if flag == False:
    #         cnt += 1
    #         file_name = os.path.join(r"C:\Users\HH\Desktop\SgCasia2\train\mask", i)
    #         os.remove(file_name)
    #         print(i)
    #
    #
    # print(cnt)

    # roi_mask = [os.path.join(data_root, "mask", i.split(".")[0] + f"_gt.png")
    #                  for i in img_names]
    #
    # # check files
    # for i in roi_mask:
    #     if os.path.exists(i) is False:
    #         raise FileNotFoundError(f"file {i} does not exists.")


    #删除格式不一样的图片

    for i in img_names:
        # if i.endswith("forged.tif"):
        #     pass
        # else:
        #     path_name = os.path.join(data_root, "mask", i)
        #     os.remove(path_name)
        #     cnt += 1

        img_path = os.path.join(data_root, "img", i)
        img = cv2.imread(img_path)
        mask_path = os.path.join(data_root, "mask", i.split(".")[0][:-1] + f"forged.tif")
        mask = cv2.imread(mask_path)

        if mask.shape != img.shape:
            os.remove(img_path)
            os.remove(mask_path)
            print("hello world")
            print(img.shape, mask.shape)

        # cv2.imwrite(r'C:\\Users\\ps\Desktop\\Coverage\\val\\mask\\' + i.split(".")[0] + f".png", img)




