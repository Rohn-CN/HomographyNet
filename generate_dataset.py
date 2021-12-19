import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse


# 将图像随机裁剪为patchsize大小，针对过大的图像:size>2*patch_size
def random_patch(image, patch_size: int):
    h, w = image.shape
    lefttop_x = np.random.randint(0, w - patch_size + 1)
    lefttop_y = np.random.randint(0, h - patch_size + 1)
    return image[lefttop_y:lefttop_y + patch_size, lefttop_x:lefttop_x + patch_size]


# 将图像resize为正方形
def resize_square(image, patch_size):
    h, w = image.shape
    if h == w:
        return cv2.resize(image, (patch_size, patch_size))
    else:
        l = min(w, h)
        c = image[(h - l) // 2:(h - l) // 2 + l + 1,
                  (w - l) // 2:(w - l) // 2 + l + 1]
        return cv2.resize(c, (patch_size, patch_size))


def random_crop(image: "square_shape", crop_size: int, random_bias: int):
    l = image.shape[0]
    left_top = np.random.randint(
        random_bias, l - random_bias - crop_size - 1, (2,))
    rect = np.array([[left_top[0], left_top[1]],
                     [left_top[0], left_top[1] + crop_size],
                     [left_top[0] + crop_size, left_top[1]],
                     [left_top[0] + crop_size, left_top[1] + crop_size]]).reshape(4, -1)
    random_bias_np = np.random.randint(-random_bias, random_bias, size=(4, 2))
    rect_trans = rect + random_bias_np

    # 计算变换
    H = cv2.getPerspectiveTransform(np.float32(rect_trans), np.float32(rect))
    image_trans = cv2.warpPerspective(image, H, dsize=image.shape)
    # 裁切&叠加
    crop1 = image[left_top[1]:left_top[1] + crop_size,
                  left_top[0]:left_top[0] + crop_size]
    crop2 = image_trans[left_top[1]:left_top[1] +
                        crop_size, left_top[0]:left_top[0] + crop_size]
    crop = np.dstack((crop1, crop2))
    # 补齐三通道方便存储
    channel_zero = np.zeros((crop.shape[0], crop.shape[1], 3))
    channel_zero[:, :, :2] = crop.copy()
    return channel_zero, random_bias_np.reshape(-1)


def image_process(image, crop_size, random_bias, save_folder, new_name, f):
    crop, bias = random_crop(image, crop_size, random_bias)
    cv2.imwrite(os.path.join(save_folder, new_name), crop)
    str_bias = list(map(str, bias))
    str_bias = ",".join(str_bias)
    gt_str = new_name + "," + str_bias + "\n"
    f.write(gt_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", default=r"./imageset")
    parser.add_argument("--save_path", default=r"./gen_dataset")
    parser.add_argument("--csv_file", default=r"./gt.csv")
    args = parser.parse_args()
    image_folder = args.image_folder
    save_folder = args.save_path
    csv_folder = args.csv_file

    f = open(csv_folder, "w")
    f.write("image,dx1,dy1,dx2,dy2,dx3,dy3,dx4,dy4\n")

    crop_size = 128
    patch_size = 512
    random_bias = 32
    crop_num = 3
    image_list = os.listdir(image_folder)
    for image_name in tqdm(image_list):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if min(image.shape) < 2 * patch_size:
            image = resize_square(image, patch_size)
            for i in range(crop_num):
                new_name = image_name[:-4] + "_" + str(i) + ".jpg"
                image_process(image, crop_size, random_bias,
                              save_folder, new_name, f)
        else:
            patch_num = min(image.shape) // patch_size ** 2
            for i in range(patch_num):
                image = random_patch(image, patch_size)
                for j in range(crop_num):
                    new_name = image_name[:-4] + "_" + \
                        str(i) + "_" + str(j) + ".jpg"
                    image_process(image, crop_size, random_bias,
                                  save_folder, new_name, f)

    f.close()
