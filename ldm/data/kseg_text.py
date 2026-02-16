import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2


class KSEGBase(Dataset):
    """KSEG Dataset Base with optional text conditioning
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
        - `text` is optional text description string (will be encoded in the model)
    """

    def __init__(self, data_root, size=256, interpolation="nearest", mode=None,
                 num_classes=2, use_text=False, text_source=None, debug=False):
        """
        Args:
            data_root: 数据集根目录，例如 "data/Kvasir-SEG_text"
            mode: "train", "val", or "test"
            use_text: 是否使用文本条件
            text_source: 文本标注文件路径（可选，默认自动查找）
            debug: 是否打印调试信息
        """
        self.data_root = data_root
        self.mode = mode
        self.debug = debug
        assert mode in ["train", "val", "test"]

        # 构建具体路径
        self.split_dir = os.path.join(data_root, mode)
        self.image_dir = os.path.join(self.split_dir, "frames")
        self.mask_dir = os.path.join(self.split_dir, "masks")

        # 检查目录是否存在
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # 解析数据列表
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])

        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]

        # Data augmentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        # Text conditioning setup
        self.use_text = use_text
        if self.use_text:
            # 如果没有提供text_source，自动查找
            if text_source is None:
                text_source = os.path.join(self.split_dir, f"KSeg_{mode}.xlsx")

            if os.path.exists(text_source):
                self.text_annotations = self._load_text_annotations(text_source)
                if self.text_annotations:
                    print(f"[Dataset] Loaded {len(self.text_annotations)} text annotations from {text_source}")
                else:
                    print(f"[Warning] Failed to load text annotations, will use default descriptions")
            else:
                self.text_annotations = None
                print(f"[Warning] Text file not found: {text_source}, will use default descriptions")
        else:
            self.text_annotations = None

        print(f"[Dataset]: KSEG {mode} with {num_classes} classes, {self._length} samples, use_text={self.use_text}")

    def _load_text_annotations(self, text_source):
        """加载文本标注文件
        支持格式: .xlsx, .xls, .json, .txt
        """
        if not os.path.exists(text_source):
            print(f"[Warning] Text source file not found: {text_source}")
            return None

        text_dict = {}

        try:
            if text_source.endswith('.xlsx') or text_source.endswith('.xls'):
                # 从Excel读取
                import pandas as pd
                df = pd.read_excel(text_source)

                # 检查列名（支持多种可能的列名）
                filename_col = None
                text_col = None

                # 查找filename列
                for col in ['filename', 'image', 'image_name', 'file_name', 'Image', 'name']:
                    if col in df.columns:
                        filename_col = col
                        break

                # 查找text列
                for col in ['text_description', 'text', 'description', 'caption', 'label']:
                    if col in df.columns:
                        text_col = col
                        break

                if filename_col and text_col:
                    text_dict = dict(zip(df[filename_col], df[text_col]))
                    print(f"[Dataset] Using columns: '{filename_col}' and '{text_col}'")
                else:
                    print(f"[Warning] Excel columns not found. Available columns: {df.columns.tolist()}")
                    return None

            elif text_source.endswith('.json'):
                # 从JSON读取
                import json
                with open(text_source, 'r', encoding='utf-8') as f:
                    text_dict = json.load(f)

            elif text_source.endswith('.txt'):
                # 简单txt格式: filename\ttext_description
                with open(text_source, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):  # 跳过空行和注释
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            text_dict[parts[0]] = parts[1]
                        else:
                            print(f"[Warning] Skipping malformed line: {line}")

            else:
                raise ValueError(f"Unsupported text file format: {text_source}")

        except Exception as e:
            print(f"[Error] Failed to load text annotations: {e}")
            import traceback
            traceback.print_exc()
            return None

        return text_dict if text_dict else None

    def _get_text_for_image(self, image_filename):
        """获取图像对应的文本描述

        尝试多种key格式匹配，如果都找不到则返回默认描述
        """
        if self.text_annotations is not None:
            base_name = os.path.basename(image_filename)

            # 尝试不同的key格式
            possible_keys = [
                base_name,                              # cju0qoxqj9q6s0835b43399p4.png
                base_name.replace('.png', ''),          # cju0qoxqj9q6s0835b43399p4
                base_name.replace('.jpg', ''),          # 如果是jpg
                base_name.replace('.jpeg', ''),         # 如果是jpeg
                f"mask_{base_name}",                    # mask_cju0qoxqj9q6s0835b43399p4.png
                os.path.splitext(base_name)[0],         # 去除任何扩展名
            ]

            for key in possible_keys:
                if key in self.text_annotations:
                    text = self.text_annotations[key]
                    # 处理可能的空值、NaN或无效字符串
                    if text and isinstance(text, str):
                        text = str(text).strip()  # 确保是字符串并去除空格
                        # 过滤无效值
                        if text and text.lower() not in ['nan', 'none', 'null', '', 'n/a']:
                            return text

        # 如果没有找到，返回默认描述
        return "polyp lesion in colon mucosa"

    def __getitem__(self, i):
        # Read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image_path = example["file_path_"]
        image_name = os.path.basename(image_path)

        # 构建mask路径（同名文件在masks文件夹）
        mask_path = os.path.join(self.mask_dir, image_name)

        # 读取mask和image
        try:
            mask_img = cv2.imread(mask_path)
            if mask_img is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
            segmentation = Image.fromarray(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB))

            img_img = cv2.imread(image_path)
            if img_img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            image = Image.fromarray(cv2.cvtColor(img_img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[Error] Failed to load image/mask:")
            print(f"  Image: {image_path}")
            print(f"  Mask:  {mask_path}")
            raise e

        # Resize
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BICUBIC)

        # Data augmentation (only for training)
        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        # Process segmentation
        segmentation = (np.array(segmentation) > 128).astype(np.float32)
        if self.mode == "test":
            example["segmentation"] = segmentation
        else:
            example["segmentation"] = ((segmentation * 2) - 1)  # range: binary -1 and 1

        # Process image
        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.  # range from -1 to 1
        example["image"] = image

        # Process text (只传字符串，不编码！DataLoader会自动组成List[str])
        if self.use_text:
            text_description = self._get_text_for_image(image_path)
            example["text"] = text_description  # ← 字符串，不是embeddings
            example["text_raw"] = text_description
        else:
            example["text"] = ""  # 空字符串
            example["text_raw"] = ""

        example["class_id"] = np.array([-1])  # doesn't matter for binary seg
        example["file_path"] = image_path  # 用于调试

        # Sanity checks
        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1., \
            f"Segmentation out of range: [{np.min(segmentation)}, {np.max(segmentation)}]"
        assert np.max(image) <= 1. and np.min(image) >= -1., \
            f"Image out of range: [{np.min(image)}, {np.max(image)}]"

        # Debug output (仅前3个样本)
        if self.debug and i < 3:
            print(f"\n[Debug Sample {i}]")
            print(f"  Image: {os.path.basename(image_path)}")
            print(f"  Text: {example['text'][:80]}..." if len(example['text']) > 80 else f"  Text: {example['text']}")
            print(f"  Image shape: {example['image'].shape}, range: [{example['image'].min():.2f}, {example['image'].max():.2f}]")
            print(f"  Seg shape: {example['segmentation'].shape}, range: [{example['segmentation'].min():.2f}, {example['segmentation'].max():.2f}]")

        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):
        """解析数据列表：读取frames文件夹中的所有图像"""
        # 支持多种图像格式
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        all_imgs = []

        for pattern in image_patterns:
            all_imgs.extend(glob.glob(os.path.join(self.image_dir, pattern)))

        all_imgs = sorted(all_imgs)

        if len(all_imgs) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        print(f"[Dataset] Found {len(all_imgs)} images in {self.image_dir}")
        return all_imgs

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        """确保mask和image使用相同的随机变换"""
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class KSEGTrain(KSEGBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/Kvasir-SEG_text", mode="train", **kwargs)


class KSEGValidation(KSEGBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/Kvasir-SEG_text", mode="val", **kwargs)


class KSEGTest(KSEGBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/Kvasir-SEG_text", mode="test", **kwargs)