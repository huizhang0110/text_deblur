import torch
from torch.utils import data
from PIL import Image


class BlurDataset(data.Dataset):
    def __init__(self, list_file, orig_transform, blur_transform):
        super(BlurDataset, self).__init__()
        self.all_orig_file = []
        self.all_blur_file = []
        self.all_kernel_file = []
        
        self.orig_transform = orig_transform
        self.blur_transform = blur_transform

        with open(list_file) as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                orig_f, blur_f, kernel_f = line.split(" ")
                self.all_orig_file.append(orig_f)
                self.all_blur_file.append(blur_f)
                self.all_kernel_file.append(kernel_f)

    def __len__(self):
        return len(self.all_orig_file)

    def __getitem__(self, index):
        orig_f = self.all_orig_file[index]
        blur_f = self.all_blur_file[index]
        try:
            orig_img = Image.open(orig_f).convert("RGB")
            blur_img = Image.open(blur_f).convert("RGB")
        except Exception as e:
            print("Error image: ", orig_f)
            return self[(idx + 1) % len(self)]

        if self.orig_transform is not None:
            orig_img = self.orig_transform(orig_img)
        if self.blur_transform is not None:
            blur_img = self.blur_transform(blur_img)
        return orig_img, blur_img


if __name__ == "__main__":
    blurdataset = BlurDataset("./data_test_13348.txt", None, None)
    orig_img, blur_img = blurdataset[0]
    orig_img.show()
    blur_img.show()
