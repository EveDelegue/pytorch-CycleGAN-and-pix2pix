import lpips as lp
import torch as tc
import skimage
import numpy as np

def ssim_score(im1,im2):
    ima = (1+im1.numpy())/2
    imb = (1+im2.numpy())/2
    d = skimage.metrics.structural_similarity(ima, imb,data_range=1)
    return 1-float(d.squeeze())


loss_fn_alex = lp.LPIPS(net='alex') # best forward scores

def lpips_score(im1,im2):
    d = loss_fn_alex(im1, im2)
    return float(d.squeeze())

def l2_score(im1,im2):
    d = tc.norm(im1-im2)
    return float(d.squeeze())

