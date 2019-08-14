
from optparse import OptionParser
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import Dense, Activation

from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.nn.initialization_method import *

from bigdl.nn.layer import *
from bigdl.dataset.dataset import *

from enum import Enum


class ShortcutType(Enum):
    CIFAR = 1
    IMAGENET = 2
    OTHERS = 3


def get_cifar_data(url, sc=None, data_type="train"):
    path = os.path.join(url, data_type)
    return SeqFileFolder.files_to_image_frame(url=path, sc=sc, class_num=100)


def get_convolution(n_input_plane, n_output_plane, kernel_w, kernel_h, stride_w, stride_h, pad_w,
                    pad_h, n_group, propagate_back=True, optnet=True, weight_decay=1e-4):
    w_reg = L2Regularizer(weight_decay)
    b_reg = L2Regularizer(weight_decay)
    if optnet:
        conv = SpatialShareConvolution(n_input_plane=n_input_plane,
                                       n_output_plane=n_output_plane, kernel_w=kernel_w,
                                       kernel_h=kernel_h, stride_w=stride_w, stride_h=stride_h,
                                       pad_w=pad_w, pad_h=pad_h, n_group=n_group,
                                       propagate_back=propagate_back, w_reg=w_reg, b_reg=b_reg)
    else:
        conv = SpatialConvolution(n_input_plane=n_input_plane, n_output_plane=n_output_plane,
                                  kernel_w=kernel_w, kernel_h=kernel_h, stride_w=stride_w,
                                  stride_h=stride_h, pad_w=pad_w, pad_h=pad_h, n_group=n_group,
                                  propagate_back=propagate_back, w_reg=w_reg, b_reg=b_reg)
    conv.set_init_method(MsraFiller(False), Zeros)
    return conv


def sbn(n_output, eps=1e-3, momentum=0.1, affine=True):
    return SpatialBatchNormalization(n_output, eps, momentum, affine).set_init_method(Ones, Zeros)


def layer(block, features, count, stride):
    s = Sequential()
    for i in range(1, count + 1):
        if i == 1:
            s.add(block(features, stride))
        else:
            s.add(block(features, 1))
    return s


def shortcut(n_input_plane, n_output_plane, stride):



def bottleneck(n, stride, optnet):
    global i_channels
    n_input_plane = i_channels
    i_channels = n * 4

    s = Sequential()
    s.add(get_convolution(n_input_plane, n, 1, 1, 1, 1, 0, 0, optnet=optnet))\
        .add(sbn(n))\
        .add(ReLU(True))\
        .add(get_convolution(n, n, 3, 3, stride, stride, 1, 1, optnet=optnet))\
        .add(sbn(n))\
        .add(ReLU(True))\
        .add(get_convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet=optnet))\
        .add(sbn(n * 4).set_init_method(Zeros, Zeros))

    bn = Sequential().add(ConcatTable().add(s).add())


def resnet_model(class_num, dataset, optnet):
    global i_channels
    model = Sequential()
    if dataset == "ImageNet":
        i_channels = 64
        model.add(get_convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet=optnet,
                                  propagate_back=False)).add(sbn(64)).add(ReLU(True)).add(
            SpatialMaxPooling(3, 3, 2, 2, 1, 1)).add()
    else:
        model.add()





if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--folder", dest="folder", default="",
                      help="the directory to reach the data")
    parser.add_option("-b", "--batchSize", dest="batchSize",
                      help="should be n*nodeNumber*coreNumber")
    parser.add_option("--nEpochs", dest="nEpochs", help="number of epochs to train")
    parser.add_option("--learningRate", dest="learningRate", default=0.1)
    parser.add_option("--warmupEpoch", dest="warmupEpoch", help="warm up epochs")
    parser.add_option("--maxLr", dest="maxLr", default=3.2,
                      help="max learning rate, default to 3.2")
    parser.add_option("--cache", dest="cache", help="directory to store snapshot")
    parser.add_option("--classes", dest="classes", default=100,
                      help="number of classes, default to 100")

    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="resnet50", conf=create_spark_conf())
    redire_spark_logs()
    show_bigdl_info_logs()
    init_engine()

    i_channels = 0

    model = resnet50.ResNet50(include_top=False, input_shape=(3, 224, 224), weights=None)
