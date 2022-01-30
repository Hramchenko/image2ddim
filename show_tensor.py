import matplotlib.pyplot as plt
import numpy as np

def show_tensor(tensor):
    img = 255*(tensor + 1)/2
    img = img.clip(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    plt.imshow(img)
    plt.show()
