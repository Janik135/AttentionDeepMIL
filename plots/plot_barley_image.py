import matplotlib.patches as patches
import matplotlib.pyplot as plt
import spectral

filename = '/Users/janik/Downloads/UV_Gerste/5dai/WT_1-_2019_10_11_13_44_51/data.hdr'

img = data_hs[filename][0]
bboxes = data_hs[filename][1]['bboxes']
image = spectral.get_rgb(img)
#image = img[:, :, :]
#image = spectral.get_rgb(image)
fig = plt.figure()
#figure = plt.gcf()  # get current figure
#figure.set_size_inches(32, 18)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image)
for bbox in bboxes:
    [(min_y, min_x), (max_y, max_x)] = bbox
    rect = patches.Rectangle((min_y, min_x), max_y-min_y, max_x-min_x, linewidth=1,
                         edgecolor='r', facecolor="none")
    ax.add_patch(rect)
plt.axis('off')
plt.savefig('barley.pdf', bbox_inches='tight')
plt.show()

'''
# for grapevine
fig = plt.figure()
figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18)
for i in range(len(test_dataset)):
    ax = fig.add_subplot(1, 5, i+1)
    sample = test_dataset[i]
    image = spectral.get_rgb(sample['image'], bands=[55, 41, 12])
    ax.imshow(image)
    ax.axis('off')
plt.tight_layout()
plt.savefig('grapevine.pdf', bbox_inches='tight')
plt.show()
'''