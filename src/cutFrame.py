import os
import cv2
import numpy as np
import torchvision.transforms as transforms

sample_size = 128

transform = transforms.Compose([transforms.Resize([sample_size, sample_size]), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])


class Preprocess:
    def __init__(self, data_path, save_path, frames=48):
        self.data_path = data_path
        self.save_path = save_path
        self.frames = frames

        os.makedirs(self.save_path, exist_ok=True)
        listdir = os.listdir(os.path.join(self.data_path))
        for i in range(0, len(listdir)):
            if not os.path.exists(os.path.join(self.save_path, listdir[i][:-4])):
                os.mkdir(os.path.join(self.save_path, listdir[i][:-4]))

    def cut_images(self, folder_path):

        if len(os.listdir(os.path.join(self.save_path, os.path.basename(folder_path)[:-4]))) == self.frames:
            return

        images = []  # list
        capture = cv2.VideoCapture(folder_path)

        fps_all = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        # 取整数部分
        timeF = int(fps_all / self.frames)
        n = 1

        # 对一个视频文件进行操作
        while capture.isOpened():
            ret, frame = capture.read()
            if ret is False:
                break
            # 每隔timeF帧进行存储操作
            if (n % timeF == 0):
                image = frame  # frame是PIL
                images.append(image)
            n = n + 1

        capture.release()
        lenB = len(images)
        # 将列表随机去除一部分元素，剩下的顺序不变

        for o in range(0, int(lenB - self.frames)):
            # 删除一个长度内随机索引对应的元素，不包括len(images)即不会超出索引
            del images[np.random.randint(0, len(images))]
            # images.pop(np.random.randint(0, len(images)))
        lenF = len(images)

        for i in range(0, lenF):
            basename = os.path.basename(folder_path)[:-4]
            cv2.imwrite(os.path.join(os.path.join(self.save_path, basename, "{:06}.jpg".format(i))), images[i])
            print(os.path.join(os.path.join(self.save_path, basename, "{:06}.jpg".format(i))))

    def begin(self):
        listdir = os.listdir(os.path.join(self.data_path))
        for i in range(0, len(listdir)):
            self.cut_images(os.path.join(self.data_path, listdir[i]))

