"""Image-only dataset with a fixed, patient-disjoint manifest."""
from PIL import Image, ImageOps
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FILL = (124, 116, 104)


class LesionDataset(Dataset):
    def __init__(self, manifest, training=False, image_size=224):
        self.frame = pd.read_csv(manifest)
        self.image_size = image_size
        steps = []
        if training:
            steps = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                     T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR, fill=FILL),
                     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
        self.transform = T.Compose(steps + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.image_path) as image:
            # Decoder-side downsampling speeds up large JPEGs; originals remain intact.
            image.draft('RGB', (self.image_size * 2, self.image_size * 2))
            image = ImageOps.exif_transpose(image).convert('RGB')
            image = ImageOps.contain(image, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
            canvas = Image.new('RGB', (self.image_size, self.image_size), FILL)
            canvas.paste(image, ((self.image_size-image.width)//2, (self.image_size-image.height)//2))
        return self.transform(canvas), torch.tensor(float(row.target)), row.image_name
