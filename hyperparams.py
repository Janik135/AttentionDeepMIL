

class WT_Run0():
    def __init__(self):
        # dataset params
        self.genotype = ["WT"]
        self.inoculated = [0, 1]
        self.dai = ["5"]
        self.test_size = 0.8
        self.signature_pre_clip = 0
        self.signature_post_clip = 1
        self.batch_size = 128
        self.savgol_filter_params = [7, 3]
        self.num_samples_file = -1
        self.max_num_balanced_inoculated = 50000
        #self.num_samples_file = 1300
        #self.max_num_balanced_inoculated = 13000

        # model params
        self.num_classes = 2
        self.hidden_layer_size = 64
        self.num_heads = 2

        # training params
        #self.lr = 0.001
        #self.num_epochs = 5000
        #self.lr_scheduler_steps = 5000

        #self.test_epoch = 100


class WT_Run1(WT_Run0):
    def __init__(self):
        super(WT_Run1, self).__init__()

        # still balanced
        self.num_samples_file = 130
        self.max_num_balanced_inoculated = 1300


class WT_Run2(WT_Run0):
    def __init__(self):
        super(WT_Run2, self).__init__()

        # still balanced
        self.num_samples_file = 5000
        self.max_num_balanced_inoculated = 50000


class WT_Run3(WT_Run0):
    def __init__(self):
        super(WT_Run3, self).__init__()

        # still balanced
        self.num_samples_file = -1
        self.max_num_balanced_inoculated = 130000


class WT_Run4(WT_Run1):
    def __init__(self):
        super(WT_Run4, self).__init__()
        self.dai=["1", "2", "3", "4", "5"]


class P01_Run0(WT_Run0):
    def __init__(self):
        super(P01_Run0, self).__init__()
        self.genotype = ["P01"]


class P22_Run0(WT_Run0):
    def __init__(self):
        super(P22_Run0, self).__init__()
        self.genotype = ["P22"]


class P01_Run2(WT_Run2):
    def __init__(self):
        super(P01_Run2, self).__init__()
        self.genotype = ["P01"]


class P22_Run2(WT_Run2):
    def __init__(self):
        super(P22_Run2, self).__init__()
        self.genotype = ["P22"]


class P01_Run3(WT_Run3):
    def __init__(self):
        super(P01_Run3, self).__init__()
        self.genotype = ["P01"]


class P22_Run3(WT_Run3):
    def __init__(self):
        super(P22_Run3, self).__init__()
        self.genotype = ["P22"]


dict_classes = {
    "WT_0": WT_Run0,
    "WT_1": WT_Run1,
    "WT_2": WT_Run2,
    "WT_3": WT_Run3,
    #"WT_4": WT_Run4,
    "P01_0": P01_Run0,
    #"P01_1": P01_Run1,
    "P01_2": P01_Run2,
    "P01_3": P01_Run3,
    "P22_0": P22_Run0,
    #"P22_1": P22_Run1,
    "P22_2": P22_Run2,
    "P22_3": P22_Run3,
}


def get_param_class(s):
    if s in list(dict_classes.keys()):
        return dict_classes[s]()
    else:
        raise ValueError("unkown config")
