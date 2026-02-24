import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2


class QaTaCovidBase(Dataset):
    """QaTa-COV19 Dataset Base with optional text conditioning

    Dataset structure expected:
        data/QaTa-COV19/
        ├── train/
        │   ├── images/          # CT images
        │   ├── masks/           # Segmentation masks (infection masks)
        │   └── QaTaCovid_train.xlsx  # Text annotations (optional)
        ├── val/
        │   ├── images/
        │   ├── masks/
        │   └── QaTaCovid_val.xlsx
        └── test/
            ├── images/
            ├── masks/
            └── QaTaCovid_test.xlsx

    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal (range -1 to 1)
        - `text` is optional text description string (will be encoded in the model)
        - Supports both lung and infection segmentation masks
    """

    def __init__(self, data_root, size=256, interpolation="bicubic", mode=None,
                 num_classes=2, use_text=False, text_source=None,
                 mask_type="infection", debug=False):
        """
        Args:
            data_root: 数据集根目录，例如 "data/QaTa-COV19"
            mode: "train", "val", or "test"
            size: 图像resize大小
            interpolation: 插值方法 ("nearest", "bilinear", "bicubic")
            num_classes: 类别数（通常是二分类）
            use_text: 是否使用文本条件
            text_source: 文本标注文件路径（可选，默认自动查找）
            mask_type: "infection" (感染区域) 或 "lung" (肺部区域)
            debug: 是否打印调试信息
        """
        self.data_root = data_root
        self.mode = mode
        self.debug = debug
        self.mask_type = mask_type
        assert mode in ["train", "val", "test"]
        assert mask_type in ["infection", "lung"], f"mask_type must be 'infection' or 'lung', got {mask_type}"

        # 构建具体路径
        self.split_dir = os.path.join(data_root, mode)
        self.image_dir = os.path.join(self.split_dir, "images")

        # 根据mask类型选择mask目录
        if mask_type == "infection":
            self.mask_dir = os.path.join(self.split_dir, "masks")
        else:  # lung
            self.mask_dir = os.path.join(self.split_dir, "lung_masks")

        # 检查目录是否存在
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            print(f"[Warning] Mask directory not found: {self.mask_dir}")
            print(f"[Info] Trying alternative mask directory...")
            # 尝试其他可能的目录名
            alternative_dirs = [
                os.path.join(self.split_dir, "infection_masks"),
                os.path.join(self.split_dir, "Infection_masks"),
                os.path.join(self.split_dir, "Lung_masks"),
                os.path.join(self.split_dir, "masks_infection"),
            ]
            for alt_dir in alternative_dirs:
                if os.path.exists(alt_dir):
                    self.mask_dir = alt_dir
                    print(f"[Info] Using alternative directory: {alt_dir}")
                    break
            else:
                raise ValueError(f"Mask directory not found. Tried: {self.mask_dir} and alternatives")

        # 解析数据列表
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])

        self.size = size
        # 支持多种插值方法
        interpolation_dict = {
            'nearest': PIL.Image.NEAREST,
            'bilinear': PIL.Image.BILINEAR,
            'bicubic': PIL.Image.BICUBIC
        }
        self.interpolation = interpolation_dict.get(interpolation, PIL.Image.BICUBIC)

        # Data augmentation (针对CT图像)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # QaTa-COV19是胸部CT，可以考虑轻微旋转
            # transforms.RandomRotation(degrees=5),
        ])

        # Text conditioning setup
        self.use_text = use_text
        if self.use_text:
            # 如果没有提供text_source，自动查找
            if text_source is None:
                text_source = os.path.join(self.split_dir, f"QaTaCovid_{mode}.xlsx")

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

        print(
            f"[Dataset]: QaTa-COV19 {mode} with {num_classes} classes, {self._length} samples, mask_type={mask_type}, use_text={self.use_text}")

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
                for col in ['filename', 'image', 'image_name', 'file_name', 'Image', 'Filename',
                            'name', 'ID', 'id', 'patient_id', 'case_id']:
                    if col in df.columns:
                        filename_col = col
                        break

                # 查找text列
                for col in ['text_description', 'text', 'description', 'caption', 'label',
                            'findings', 'report', 'Text', 'clinical_findings', 'diagnosis']:
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
                base_name,  # Patient_001_slice_10.png
                base_name.replace('.png', ''),
                base_name.replace('.jpg', ''),
                base_name.replace('.jpeg', ''),
                base_name.replace('.nii.gz', ''),  # NIfTI格式
                base_name.replace('.dcm', ''),  # DICOM
                f"mask_{base_name}",
                f"infection_{base_name}",
                os.path.splitext(base_name)[0],  # 去除扩展名
                # 提取患者ID (如 Patient_001)
                '_'.join(base_name.split('_')[:2]) if '_' in base_name else base_name,
            ]

            for key in possible_keys:
                if key in self.text_annotations:
                    text = self.text_annotations[key]
                    # 处理可能的空值、NaN或无效字符串
                    if text and isinstance(text, str):
                        text = str(text).strip()
                        # 过滤无效值
                        if text and text.lower() not in ['nan', 'none', 'null', '', 'n/a']:
                            return text

        # 默认描述根据mask类型
        if self.mask_type == "infection":
            return "multifocal ground-glass opacities and consolidation in lung parenchyma"
        else:  # lung
            return "bilateral lung fields with normal parenchyma"

    def _normalize_ct_image(self, image_array):
        """CT图像的窗宽窗位归一化

        Args:
            image_array: numpy array, CT HU值

        Returns:
            normalized_array: [0, 255] 范围的uint8数组
        """
        # 肺窗：窗宽1500, 窗位-600
        window_center = -600
        window_width = 1500

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        # Clip和归一化
        image_array = np.clip(image_array, img_min, img_max)
        image_array = ((image_array - img_min) / (img_max - img_min) * 255.0)

        return image_array.astype(np.uint8)

    def __getitem__(self, i):
        # Read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image_path = example["file_path_"]
        image_name = os.path.basename(image_path)

        # 构建mask路径 - QaTa-COV19可能有多种命名格式
        possible_mask_names = [
            image_name,  # 同名
            image_name.replace('.png', '_mask.png'),  # xxx_mask.png
            image_name.replace('.jpg', '_mask.jpg'),
            'mask_' + image_name,  # mask_xxx.png
            image_name.replace('image', 'mask'),  # image_xxx -> mask_xxx
            image_name.replace('ct', 'mask'),
            image_name.replace('Image', 'Mask'),
            image_name.replace('.png', '_infection_mask.png'),  # infection mask
            image_name.replace('.png', '_lung_mask.png'),  # lung mask
        ]

        mask_path = None
        for possible_name in possible_mask_names:
            candidate = os.path.join(self.mask_dir, possible_name)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            # 如果都找不到，使用默认路径
            mask_path = os.path.join(self.mask_dir, image_name)

        # 读取mask和image
        try:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError(f"Failed to read mask: {mask_path}")

            segmentation = Image.fromarray(mask_img)

            # 读取CT图像
            img_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img_img is None:
                # 尝试读取彩色图像
                img_img = cv2.imread(image_path)
                if img_img is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                # 转为灰度
                img_img = cv2.cvtColor(img_img, cv2.COLOR_BGR2GRAY)

            # CT图像归一化（如果需要）
            # 如果图像已经是0-255的PNG，可以跳过
            # img_img = self._normalize_ct_image(img_img)

            # 转为RGB格式（模型需要3通道）
            image = Image.fromarray(cv2.cvtColor(img_img, cv2.COLOR_GRAY2RGB))

        except Exception as e:
            print(f"[Error] Failed to load image/mask:")
            print(f"  Image: {image_path}")
            print(f"  Mask:  {mask_path}")
            raise e

        # Resize
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=self.interpolation)

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

        # Process text
        if self.use_text:
            text_description = self._get_text_for_image(image_path)
            example["text"] = text_description
            example["text_raw"] = text_description
        else:
            example["text"] = ""
            example["text_raw"] = ""

        example["class_id"] = np.array([-1])  # binary segmentation
        example["file_path"] = image_path

        # Sanity checks
        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1., \
            f"Segmentation out of range: [{np.min(segmentation)}, {np.max(segmentation)}]"
        assert np.max(image) <= 1. and np.min(image) >= -1., \
            f"Image out of range: [{np.min(image)}, {np.max(image)}]"

        # Debug output
        if self.debug and i < 3:
            print(f"\n[Debug Sample {i}]")
            print(f"  Image: {os.path.basename(image_path)}")
            print(f"  Mask type: {self.mask_type}")
            print(f"  Text: {example['text'][:80]}..." if len(example['text']) > 80 else f"  Text: {example['text']}")
            print(
                f"  Image shape: {example['image'].shape}, range: [{example['image'].min():.2f}, {example['image'].max():.2f}]")
            print(
                f"  Seg shape: {example['segmentation'].shape}, range: [{example['segmentation'].min():.2f}, {example['segmentation'].max():.2f}]")

        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):
        """解析数据列表：读取images文件夹中的所有图像"""
        # 支持多种图像格式
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
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


class QaTaCovidTrain(QaTaCovidBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/QaTa-COV19", mode="train", **kwargs)


class QaTaCovidValidation(QaTaCovidBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/QaTa-COV19", mode="val", **kwargs)


class QaTaCovidTest(QaTaCovidBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/QaTa-COV19", mode="test", **kwargs)