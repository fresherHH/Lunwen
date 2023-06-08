import os
import os.path as osp
import random
import shutil


def split_train_val(p=0.9):

    random.seed(0)

    src_root_path = r'C:\Users\HH\Desktop\SgCasia2\train'
    dst_root_path = r'C:\Users\HH\Desktop\SgCasia2\val'


    src_path_img = f'{src_root_path}/img'
    dst_path_img = f'{dst_root_path}/img'
    src_path_mask = f'{src_root_path}/mask'
    dst_path_mask = f'{dst_root_path}/mask'
    os.makedirs(dst_path_img, exist_ok=True)
    os.makedirs(dst_path_mask, exist_ok=True)
    for filename in os.listdir(src_path_img):
        filename_mask = filename.split(".")[0] + f"_gt.png"
        if random.random() > p:
            src_file_img = f'{src_path_img}/{filename}'
            dst_file_img = f'{dst_path_img}/{filename}'
            src_file_mask = f'{src_path_mask}/{filename_mask}'
            dst_file_mask = f'{dst_path_mask}/{filename_mask}'
            print(src_file_img, src_file_mask)

            shutil.move(src_file_img, dst_file_img)
            shutil.move(src_file_mask, dst_file_mask)

def leaky_label():
    root_path = ''
    dirs = ['OK_train', 'NG4_train', 'NG2_train', 'NG5_train', 'NG1_train', 'NG6_ train', 'NG3_train', 'NG7_train']
    result = dict()
    for dir in dirs:
        images = os.listdir(osp.join(root_path, dir))
        label_ids = [image.split('_')[-2] for image in images]
        label_ids = list(set(label_ids))
        label_ids.sort()
        result[dir] = label_ids
    return result



def check_unique_label(retult_d):
    for k, v in retult_d.items():
        for item in v:
            for _k, _v in retult_d.items():
                if _k != k and item in _v:
                    print(item, k, _k)


def main():
    split_train_val()


if __name__ == '__main__':
    main()
    pass
