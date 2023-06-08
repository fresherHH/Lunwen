
import os, random, shutil

if __name__=="__main__":

    # src = r"F:\experiment\dataset\VOCdevkit\VOC2012\JPEGImages"
    #
    # imglist = os.listdir(src)
    #
    # print(len(imglist))
    # print(imglist[0])
    # for i in range(len(imglist)):
    #     if i < 10553:
    #         shutil.copy(os.path.join(src, imglist[i]), r"C:\Users\HH\Desktop\DATASET\Train\BG")
    #
    #     elif i < 10553 + 5019:
    #         shutil.copy(os.path.join(src, imglist[i]), r"C:\Users\HH\Desktop\DATASET\Test\BG")

    src = r"C:\Users\HH\Desktop\DATASET\Train\BG"
    train_bg = []
    train_fg = []
    train_gt = []
    test_bg = []
    test_fg = []
    test_gt = []
    for name in os.listdir(src):
        train_bg.append(name)



    with open(r"C:\Users\HH\Desktop\DATASET\Train\bg.txt", 'w') as f:
        for name in train_bg:
            f.write(name + '\n')

    with open(r"C:\Users\HH\Desktop\DATASET\Train\fg.txt", 'w') as f:
        for name in train_fg:
            f.write(name + '\n')

    with open(r"C:\Users\HH\Desktop\DATASET\Train\gt.txt", 'w') as f:
        for name in train_gt:
            f.write(name + '\n')

    with open(r"C:\Users\HH\Desktop\DATASET\Test\bg.txt", 'w') as f:
        for name in test_bg:
            f.write(name + '\n')

    with open(r"C:\Users\HH\Desktop\DATASET\Test\fg.txt", 'w') as f:
        for name in test_fg:
            f.write(name + '\n')

    with open(r"C:\Users\HH\Desktop\DATASET\Test\gt.txt", 'w') as f:
        for name in test_gt:
            f.write(name + '\n')