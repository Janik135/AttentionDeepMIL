import glob
import os
import pickle

data_path = "/Users/janik/Downloads/UV_Gerste/"
parsed_data_path = os.path.join(data_path, "parsed_data")
current_path = os.path.join(parsed_data_path, "*.p")

filenames = sorted(list(set(glob.glob(current_path))))
data_hs = dict()
data_pos = []

for filename in filenames:
    #filename = filenames[0]
    data_pos_file = []
    bbox_obj_dict = pickle.load(open(filename, "rb"))
    data_pos.append(bbox_obj_dict)
    hs_img_path = os.path.join(os.path.join(data_path, "{}dai".format(bbox_obj_dict["label_dai"])),
                               bbox_obj_dict["filename"] + "/data.hdr")