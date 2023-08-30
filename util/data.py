# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2022-07-27 20:31:10
   @LastEditors  : Please set LastEditors
   @LastEditTime : 2022-07-28 22:57:11
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add descriptioon
'''
import os
import random
import torch
import numpy as np
import scipy.io as io
import torchvision.transforms as transforms


def random_spilt(root_dir, train_radio):
    # 每类样本按比例随机划分训练和测试
    # 返回训练和测试样本的路径列表
    train_items = []
    test_items = []
    for (root, dirs, files) in os.walk(root_dir):
        dirs.sort()     # 类别排序
        # 设置训练和测试比例并分配索引
        shuffled_indices = np.random.permutation(len(files))
        train_size = int(len(files) * train_radio)
        train_indices = shuffled_indices[:train_size]
        # test_indices = shuffled_indices[train_size:]
        for i, f in enumerate(files):
            if f.endswith("png") or f.endswith("mat"):
                r = root.split('/')
                lr = len(r)
                if i in train_indices:
                    train_items.append((f, r[lr - 2] + "/" + r[lr - 1], root))
                else:
                    test_items.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== TrainSet %d items, TestSet %d items" % (len(train_items), len(test_items)))
    return train_items, test_items


def index_classes(items):
    # 统计类别数量
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class Event2Frame_RCS(object):
    """ Convert DVS event streams to frames
    Args:
    """

    def __init__(self, img_size, tr, ts):
        self.img_size = img_size
        self.tr = tr    # time resolution
        self.ts = ts    # timesteps

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([2, self.img_size[0], self.img_size[1], self.ts], dtype=int)  # frame
        maxT = np.max(events['TD'].ts)
        length = int(self.ts * self.tr)
        start = random.randrange(0, maxT - length - 1, self.tr)
        end = start + length
        for j in range(start, end, int(self.tr)):    # tr ms 的帧
            idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 1)
            idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 2)
            frame[0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1, int((j - start) / self.tr)] = 1.0
            frame[1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1, int((j - start) / self.tr)] = 1.0
            # im = Image.fromarray((frame[0, ..., int((j-start) / self.tr)] * 255).astype(np.uint8), "L")  # numpy 转 image类
            # im.save(os.path.join('./', str(j / self.tr) + '.png'))
        return np.reshape(frame, (2, self.img_size[0], self.img_size[1], self.ts))


class Event2Frame(object):
    """ Convert DVS event streams to frames
    Args:
    """

    def __init__(self, img_size, tr, ts):
        self.img_size = img_size
        self.tr = tr    # time resolution
        self.ts = ts    # timesteps

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([2, self.img_size[0], self.img_size[1], self.ts], dtype=int)  # frame
        for j in range(0, int(self.ts * self.tr), int(self.tr)):    # tr ms 的帧
            idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 1)
            idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + self.tr) & (events['TD'].p == 2)
            frame[0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1, int(j / self.tr)] = 1.0
            frame[1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1, int(j / self.tr)] = 1.0
        #     im = Image.fromarray((frame[int(j / self.tr), :] * 255).astype(np.uint8), "L")  # numpy 转 image类
        #     im.save(os.path.join('./', str(j / self.tr) + '.png'))
        return np.reshape(frame, (2, self.img_size[0], self.img_size[1], self.ts))


class Event2Frame_FULL(object):
    """ Convert DVS full event streams to frames
    Args:
    """

    def __init__(self, input_shape, ts):
        self.input_shape = input_shape
        self.ts = ts          # timesteps

    def __call__(self, sample):
        """
        Args:
            sample: mat
        Returns:
            frame: numpy array
        """
        events = io.loadmat(sample, squeeze_me=True, struct_as_record=False)
        frame = np.zeros([self.ts, self.input_shape[0], self.input_shape[1], self.input_shape[2]], dtype=np.float32)  # frame
        tr = events['TD'].ts[-1] // self.ts
        for j in range(0, int(self.ts * tr), int(tr)):    # tr ms 的帧
            idx_n = (events['TD'].ts >= j) & (events['TD'].ts < j + tr) & (events['TD'].p == 1)
            idx_p = (events['TD'].ts >= j) & (events['TD'].ts < j + tr) & (events['TD'].p == 2)
            frame[int(j / tr), 0, events['TD'].y[idx_n] - 1, events['TD'].x[idx_n] - 1] = 1.0
            frame[int(j / tr), 1, events['TD'].y[idx_p] - 1, events['TD'].x[idx_p] - 1] = 1.0
        return frame


class CIFAR10_DVS(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 transform=None,
                 target_transform=None):
        self.all_items = data_list
        self.transform = transform
        self.target_transform = target_transform
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        classname = self.all_items[index][1]
        filepath = self.all_items[index][2]

        img = os.path.join(str(filepath), str(filename))
        target = self.idx_classes[classname]
        # print(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_items)


class CIFAR10_DVS_Aug(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        # if self.train:
        new_data = []
        for t in range(data.size(0)):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        data = torch.stack(new_data, dim=0)
        if self.transform is not None:
            flip = random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


if __name__ == '__main__':
    root = "/opt/data/durian/CIFAR10DVS/events_pt"
    train_path = root + '/train'
    val_path = root + '/test'
    train_data = CIFAR10_DVS_Aug(root=train_path, transform=False)
    val_data = CIFAR10_DVS_Aug(root=val_path)
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True,
                                            pin_memory=True, num_workers=1)
    val_ldr = torch.utils.data.DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                          pin_memory=True, num_workers=1)

    for idx, (ptns, labels) in enumerate(train_ldr):
        print(ptns.shape, labels)
