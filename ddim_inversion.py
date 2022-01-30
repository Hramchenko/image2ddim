from tqdm import tqdm
import torch
import numpy as np
from torchvision import transforms
import cv2

@torch.no_grad()
def p_sample_ddim(self, x, c, t, index):
    b, *_, device = *x.shape, x.device
    e_t = self.model.apply_model(x, t, c)

    alphas =  self.ddim_alphas
    alphas_prev = self.ddim_alphas_prev
    sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
    sigmas = self.ddim_sigmas
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)

    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_prev).sqrt() * e_t
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0

@torch.no_grad()
def sample(self, x_T):
    b = x_T.shape[0]
    device = x_T.device
    cond = None
    img = x_T
    log_every_t = 10
    timesteps = self.ddim_timesteps

    intermediates = {'x_inter': [img], 'pred_x0': [img]}
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    print(f"Running DDIM Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='DDIM', total=total_steps)

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        outs = p_sample_ddim(self, img, cond, ts, index=index)
        img, pred_x0 = outs
        if index % log_every_t == 0 or index == total_steps - 1:
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)
    return img, intermediates

@torch.no_grad()
def p_sample_ddim_rev(self, x, c, t, index):
    b, *_, device = *x.shape, x.device
    e_t = self.model.apply_model(x, t, c)

    alphas =  self.ddim_alphas
    alphas_next = self.ddim_alphas_next
    sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_next = torch.full((b, 1, 1, 1), alphas_next[index], device=device)

    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # direction pointing to x_t
    dir_xt = (1. - a_next).sqrt() * e_t
    x_prev = a_next.sqrt() * pred_x0 + dir_xt
    return x_prev, pred_x0

@torch.no_grad()
def sample_rev(self, x_T):
    b = x_T.shape[0]
    device = x_T.device
    cond = None
    img = x_T
    log_every_t = 10
    timesteps = self.ddim_timesteps

    intermediates = {'x_inter': [img], 'pred_x0': [img]}
    time_range = timesteps

    total_steps = timesteps.shape[0]
    print(f"Running DDIM Reverse Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='DDIM Reverse', total=total_steps)

    for i, step in enumerate(iterator):
        index = i
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        outs = p_sample_ddim_rev(self, img, cond, ts, index=index)
        img, pred_x0 = outs
        if index % log_every_t == 0 or index == total_steps - 1:
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)
    return img, intermediates

def to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img)
    return img_tensor

def to_image(img_tensor):
    img = (255*(img_tensor + 1)/2).permute(1, 2, 0).detach().cpu().numpy()
    img = img.clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def img_to_latents(diffusion, img_tensor):
    model = diffusion.model
    device = img_tensor.device
    orig_lats = model.encode_first_stage(img_tensor.unsqueeze(0).to(device))
    lats_rev, _ = sample_rev(diffusion, orig_lats)
    return lats_rev

def latents_to_img(diffusion, lats_rev):
    model = diffusion.model
    orig_lats_, _ = sample(diffusion, lats_rev)
    orig_decoded_ = model.decode_first_stage(orig_lats_)
    return orig_decoded_