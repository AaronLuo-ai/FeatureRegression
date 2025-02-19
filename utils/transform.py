import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, adjust_brightness

class EncoderTransform(transforms.Compose):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):


    def __call__(self, *args, **kwargs):