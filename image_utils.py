"""
Author: Moustafa Alzantot (malzantot@ucla.edu)

"""
import numpy as np
import matplotlib as mpl
from imageio import imread
from skimage.transform import resize
import matplotlib.pyplot as plt


def load_image(img_path, resize_dim=299):
    img = resize(imread(img_path),(resize_dim,resize_dim)).astype(np.float32)
    return img

def display_image(img):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.axis('off')
    ax.imshow(img)
    return fig

def display_image_and_mask(img, mask):
    fig, axes = plt.subplots(1,3, figsize=(40,10))
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Input Image')
    
    axes[1].imshow(img)
    axes[1].imshow(mask[:,:,0], cmap='jet', alpha=0.8)
    axes[1].axis('off')
    
    axes[2].imshow(mask[:,:,0], cmap='jet')
    axes[2].axis('off')
    return fig