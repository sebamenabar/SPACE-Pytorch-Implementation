import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


ATTRIBUTES = {
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "size": ["large", "small"],
    "shape": ["cube", "sphere", "cylinder"],
    "material": ["rubber", "metal"],
}
OTHER_ATTRIBUTES = ["rotation", "3d_coords", "pixel_coords"]


def process_objects(objects):
    out = {attr_name: [] for attr_name in list(ATTRIBUTES.keys()) + OTHER_ATTRIBUTES}
    for obj in objects:
        for attr_name, attr_values in ATTRIBUTES.items():
            out[attr_name].append(attr_values.index(obj[attr_name]))
        for attr_name in OTHER_ATTRIBUTES:
            out[attr_name].append(obj[attr_name])
    return out


class CLEVRDataset(Dataset):
    def __init__(self, root_dir, split="train", image_sizes=((64, 64), (128, 128))):
        super().__init__()

        self.image_sizes = image_sizes
        self.root_dir = root_dir
        self.split = split

        with open(os.path.join(root_dir, "scenes", f"CLEVR_{split}_scenes.json")) as f:
            self.scenes = json.load(f)["scenes"]
        self.preprocessed_scenes = self.preprocess_scenes()
        self.transforms = [
            T.Compose((T.Resize(img_sz), T.ToTensor())) for img_sz in image_sizes
        ]

    def load_image(self, img_filename):
        return Image.open(
            os.path.join(self.root_dir, "images", self.split, img_filename)
        ).convert("RGB")

    def preprocess_scenes(self):
        preprocessed_scenes = []
        for scene in self.scenes:
            s = {
                "image_filename": scene["image_filename"],
                "image_index": scene["image_index"],
            }
            s["objects"] = process_objects(scene["objects"])
            s["num_objects"] = len(scene["objects"])
            preprocessed_scenes.append(s)
        return preprocessed_scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.preprocessed_scenes[index]
        image = self.load_image(scene["image_filename"])
        images = [transform(image) for transform in self.transforms]

        return images, scene["objects"], scene["num_objects"]


def collate_fn(batch):
    images = [b[0] for b in batch]
    scenes = [b[1] for b in batch]
    lengths = [b[2] for b in batch]

    out_images = [[] for _ in range(len(images[0]))]
    for sample_images in images:
        for i, img in enumerate(sample_images):
            out_images[i].append(img)
    out_images = [torch.stack(images, 0) for images in out_images]

    out_attrs = {attr_name: [] for attr_name in scenes[0].keys()}
    for scene in scenes:
        for attr_name, attr_values in scene.items():
            out_attrs[attr_name].append(torch.tensor(attr_values))
    out_attrs = {
        attr_name: pad_sequence(attr_values, True, -1, min_len=10)
        for attr_name, attr_values in out_attrs.items()
    }

    return out_images, out_attrs, torch.tensor(lengths)


def pad_sequence(sequences, batch_first=False, padding_value=0, min_len=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(*[s.size(0) for s in sequences], min_len)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
