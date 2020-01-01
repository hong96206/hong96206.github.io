from PIL import Image
import numpy as np
import cv2 as cv
import os


def gifSplit(src_path, dest_path, suffix="png"):
    img = Image.open(src_path)
    for i in range(img.n_frames):
        img.seek(i)
        new = Image.new("RGBA", img.size)
        new.paste(img)
        new.save(os.path.join(dest_path, "%d.%s" % (i, suffix)))


# yield的功能类似于return，但是不同之处在于它返回的是生成器
def gifSplitToArray(src_path, dest_path):
    img = Image.open(src_path)
    for i in range(img.n_frames):
        img.seek(i)
        new = Image.new("RGBA", img.size)
        new.paste(img)
        arr = np.array(new).astype(np.uint8)  # image: img (PIL Image):
        yield arr[:, :, 2::-1]  # 逆序（RGB 转BGR), 舍弃alpha通道, 输出数组供openCV使用


def create_gif(imagesPath, gif_name, duration=0.3, reverse=False):
    import imageio
    fileNames = os.listdir(imagesPath)
    frames = []  # 列表，用于存储各个帧
    fileList = os.listdir(imagesPath)
    if reverse:
        fileList.sort(reverse=True)
    for file in fileList:
        fullpath = os.path.join(imagesPath, file)
        frames.append(imageio.imread(fullpath))  # 添加帧数据
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)  # 合成


if __name__ == '__main__':
    # 先将gif分解为png文件存放于pics下
    gifSplit('tiga.gif', r'./pics')
    # 将pics下的png文件逆序生成merged.gif文件
    create_gif(r"./pics", "merged.gif", duration=0.3, reverse=True)
