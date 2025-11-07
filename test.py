import torch
import matplotlib.pyplot as plt
import networks
from visualize_utils import *


encoder = networks.ResnetEncoder(18, False).to(torch.device("cuda"))
decoder = networks.DecomposeDecoder(encoder.num_ch_enc).to(torch.device("cuda"))
image_loader = ImageLoader(1, 1)
image = image_loader.single(1)

feats = encoder(image)
outputs = decoder(feats)