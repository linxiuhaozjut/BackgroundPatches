import torch
import numpy as np

class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm

class PreprocessorMM(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view((1, 6, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view((1, 6, 1, 1)).cuda()

    def process(self, img_tensor: torch.Tensor, patch_arr: np.ndarray, top_left: tuple, mask_array: np.ndarray):#img_arr: np.ndarray
        # Deal with the image patch
        #img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor = img_tensor.unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,6,H,W)
        return img_tensor_norm

    # def process(self, img_arr: np.ndarray, patch_arr: np.ndarray, top_left: tuple, mask_array: np.ndarray):
    #     # Deal with the image patch
    #     # 1. Image to tensor
    #     img_tensor = torch.tensor(img_arr).float().cuda().permute(2, 0, 1)  # [C, H, W]
    #     print("img_tensor_shape: ", img_tensor.shape)
    #     img_clone = img_tensor.clone()
    #
    #     # 2. Patch array to tensor (check channel order)
    #     patch_tensor = torch.tensor(patch_arr).float().cuda().permute(2, 0, 1)  # [C, ph, pw]
    #     print("patch_tensor_shape: ", patch_tensor.shape)
    #
    #     # 3. Composite patch
    #     px, py = top_left
    #     _, ph, pw = patch_tensor.shape
    #     for i in range(4):
    #         x_start = px + i * pw
    #         x_end = x_start + pw
    #         img_tensor[:, py:py + ph, x_start:x_end] = patch_tensor
    #
    #     # 4. Restore masked pixels (mask: [H, W])
    #     mask_tensor = torch.tensor(mask_array == 255).cuda()  # bool mask
    #     mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
    #
    #     # 5. Use where to restore original pixels under the mask
    #     img_tensor = torch.where(mask_tensor, img_clone, img_tensor)
    #
    #     # 6. Batch dimension and normalization
    #     img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    #     img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std
    #
    #     return img_tensor_norm


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
