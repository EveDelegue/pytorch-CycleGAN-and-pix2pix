import lpips as lp
import torch as tc



loss_fn_alex = lp.LPIPS(net='alex') # best forward scores

def lpips_score(im1,im2):
    d = loss_fn_alex(im1, im2)
    return float(d.squeeze())

def l2_score(im1,im2):
    d = tc.norm(im1-im2)
    return float(d.squeeze())