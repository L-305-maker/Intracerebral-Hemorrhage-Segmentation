import numpy as np
from PIL import Image

try:
    import albumentations as A
except ImportError:
    A = None


class ResizeOnlyTransform:
    def __init__(self, image_size=128):
        self.image_size = image_size

    def __call__(self, image, mask):
        image = Image.fromarray(image).resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )
        mask = Image.fromarray(mask).resize(
            (self.image_size, self.image_size),
            Image.NEAREST
        )
        return {
            "image": np.array(image),
            "mask": np.array(mask),
        }


def get_train_transforms(image_size=128):
    if A is None:
        return ResizeOnlyTransform(image_size)

    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.Rotate(limit=5, border_mode=0, p=1.0),
                    A.ShiftScaleRotate(
                        shift_limit=0.02,
                        scale_limit=0.02,
                        rotate_limit=0,
                        border_mode=0,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


def get_val_transforms(image_size=128):
    if A is None:
        return ResizeOnlyTransform(image_size)

    return A.Compose(
        [A.Resize(image_size, image_size, interpolation=1)],
        additional_targets={"mask": "mask"},
    )
