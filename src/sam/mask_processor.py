import numpy as np
import torch

from torchvision.transforms import ToTensor, Resize
from torchvision.utils import save_image

def get_seg_img(mask, image):
    image = image.copy()
    mask_image = image.copy()
    # here I remove the masking operation.
    mask_image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    mask_img = mask_image[y:y+h, x:x+w, ...]
    mask = mask_img.copy().sum(axis=2)
    mask[mask != 0] = 255
    return seg_img, mask_img, mask

def pad_img(img):
    h, w = img.shape[:2]
    l = max(w,h)
    if img.ndim == 3:
        pad = np.zeros((l,l,3), dtype=np.uint8)
        if h > w:
            pad[:,(h-w)//2:(h-w)//2 + w, :] = img
        else:
            pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    else:
        pad = np.zeros((l,l), dtype=np.uint8)
        if h > w:
            pad[:,(h-w)//2:(h-w)//2 + w] = img
        else:
            pad[(w-h)//2:(w-h)//2 + h, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(masks_lvl, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    
    seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
    iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
    stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

    scores = stability * iou_pred
    keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
    masks_lvl = filter(keep_mask_nms, masks_lvl)
    return masks_lvl

def mask2segmap(masks, image, padding=True):
    seg_img_list = []
    mask_img_list = []
    mask_map_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img, mask_image, mask_map = get_seg_img(mask, image)
        if padding:
            seg_img = Resize(224)(ToTensor()(pad_img(seg_img)))
            mask_img = Resize(224)(ToTensor()(pad_img(mask_image)))
            mask_map = Resize(224)(ToTensor()(pad_img(mask_map)))
        else:
            seg_img = ToTensor()(seg_img)
            mask_img = ToTensor()(mask_image)
            mask_map = ToTensor()(mask_map)
            
        seg_img_list.append(seg_img)
        mask_img_list.append(mask_img)
        mask_map_list.append(mask_map)
        seg_map[masks[i]['segmentation']] = i
    seg_imgs = torch.stack(seg_img_list, axis=0).to("cuda")
    mask_imgs = torch.stack(mask_img_list, axis=0).to("cuda")
    mask_maps = torch.stack(mask_map_list, axis=0).to("cuda")
    return seg_imgs, mask_imgs, mask_maps, seg_map

def mask_processor(image, mask_generator, save_folder=None):
    masks = mask_generator.generate(image)
    masks = masks_update(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    seg_images, mask_images, mask_maps, seg_map = mask2segmap(masks, image)
    
    if save_folder is not None:
        (save_folder / "seg_img").mkdir(parents=True, exist_ok=True)
        (save_folder / "mask_img").mkdir(parents=True, exist_ok=True)
        (save_folder / "mask").mkdir(parents=True, exist_ok=True)
        for i, seg_img in enumerate(seg_images):
            save_image(seg_img, save_folder / "seg_img" / f"{i:03d}.png")
        for i, mask_img in enumerate(mask_images):
            save_image(mask_img, save_folder / "mask_img" / f"{i:03d}.png")
        for i, mask_map in enumerate(mask_maps):
            save_image(mask_map, save_folder / "mask" / f"{i:03d}.png")

    seg_map = np.tile(seg_map, (4, 1, 1))
    
    return seg_images, mask_images, seg_map