import matplotlib.pyplot as plt
import numpy as np
import spectral
import torch
import torch.nn.functional as F

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from cnn_model import CNNModel
from matplotlib.colors import LinearSegmentedColormap

net = CNNModel(2)
net = net.eval()
net.load_state_dict(torch.load('models/cnn_model_cheating.pt', map_location=torch.device('cpu')))

sample, label = dataset_test[0]
sample = torch.from_numpy(sample.transpose((2, 0, 1)))
sample = sample.unsqueeze(0).unsqueeze(0)
output = net(sample)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

pred_label_idx.squeeze_()

integrated_gradients = IntegratedGradients(net)
attributions_ig = integrated_gradients.attribute(sample, target=pred_label_idx, n_steps=200)

default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

attr_np = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0))
image_np = np.transpose(sample.squeeze().cpu().detach().numpy(), (1,2,0))
_ = viz.visualize_image_attr(attr_np,
                             image_np,
                             method='heat_map',
                             cmap=default_cmap,
                             show_colorbar=True,
                             sign='positive',
                             outlier_perc=1)

spectral.imshow(image_np)
plt.show()

max_x = np.max(attr_np, axis=0)
max_y = np.max(max_x, axis=0)

top_10_channels = max_y.argsort()[-10:][::-1]
for i in top_10_channels:
    print("Wavelength ", wavelength[i], "with max attr. ", max_y[i])

top_20_channels = max_y.argsort()[-20:][::-1]
for i in top_20_channels:
    print("Wavelength ", wavelength[i], "with max attr. ", max_y[i])

mean = np.mean(attr_np, axis=(0, 1))
plt.plot(wavelength, mean)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Mean attribution')
plt.title('Mean attribution of wavelengths (3D CNN)')
plt.tight_layout()
plt.savefig('int_gradients_2.pdf')
plt.show()