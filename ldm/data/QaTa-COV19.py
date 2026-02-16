import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2


class QaTaCOV19(Dataset):
    """MosMedData Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, train_dir, val_dir, test_dir, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.mode = mode
        assert mode in ["train", "val", "test"]
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256))
        ])
        # TODO: more data transformation

        print(f"[Dataset]: QaTa-COV19 with 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        # segmentation = Image.open(example["file_path_"].replace("Original", "GroundTruth")).convert("RGB")
        # image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix
        img_path = example["file_path_"]

        # 分离目录和文件名
        img_dir, img_name = os.path.split(img_path)

        # 构建 mask 路径：将 Images 替换为 Ground-truths，文件名前加 mask_
        mask_dir = img_dir.replace("Images", "Ground-truths")
        mask_path = os.path.join(mask_dir, f"mask_{img_name}")

        # 读取 mask（3通道 RGB，和原来一样）
        segmentation = Image.fromarray(
            cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        )
        #segmentation = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"].replace("images", "Ground-truths").replace(".png", "mask_")), cv2.COLOR_BGR2RGB))
        image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]), cv2.COLOR_BGR2RGB))

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        segmentation = (np.array(segmentation) > 128).astype(np.float32)
        if self.mode == "test":
            example["segmentation"] = segmentation
        else:
            example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):
        # 获取每个目录下的所有图片文件
        if self.mode == "train":
            all_imgs = glob.glob(os.path.join(self.train_dir, "*.png"))
        elif self.mode == "val":
            all_imgs = glob.glob(os.path.join(self.val_dir, "*.png"))
        elif self.mode == "test":
            all_imgs = glob.glob(os.path.join(self.test_dir, "*.png"))
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test!")

        return all_imgs

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class QaTaCOV19Train(QaTaCOV19):
    def __init__(self, train_dir="data/QaTa-COV19/train/Images", **kwargs):
        super().__init__(train_dir=train_dir, val_dir=None, test_dir=None, mode="train", **kwargs)


class QaTaCOV19Validation(QaTaCOV19):
    def __init__(self, val_dir="data/QaTa-COV19/test/Images", **kwargs):
        super().__init__(train_dir=None, val_dir=val_dir, test_dir=None, mode="val", **kwargs)


class QaTaCOV19Test(QaTaCOV19):
    def __init__(self, test_dir="data/QaTa-COV19/test/Images", **kwargs):
        super().__init__(train_dir=None, val_dir=None, test_dir=test_dir, mode="test", **kwargs)