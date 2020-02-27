import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import json
import torch
from time import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def load_imu_data(path):

    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)

    return load_dict


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = list(filter(lambda i : i != 1,dim))
    return strided.reshape(dim)

def segMent(x, y ):
    data_x = []
    data_y = []
    count = 0
    for i in range(len(x)):
        dataX = x[i]
        dataY = y[i]
        peaks, _ = find_peaks(dataX, height=2.0, distance=30)
        # plt.plot(x)
        # plt.plot(peaks, x[peaks], "x")
        # plt.show()
        t = np.arange(0, 128, 1)
        if len(peaks) > 2:
            for j in range(len(peaks) - 1):
                x_new = np.zeros(128)
                y_new = np.zeros(128)
                length = peaks[j + 1] - peaks[j]
                padding_index = (int)((128 - length) / 2)
                for k in range(length + 6):
                    x_new[padding_index + k] = dataX[peaks[j] + k - 8]
                    y_new[padding_index + k] = dataY[peaks[j] + k - 8]

                # label = np.zeros(128)
                # for m in range(128):
                #     if y_new[m]>0:
                #         label[m] = x_new[m]

                data_x.append(x_new)
                data_y.append(y_new)
                # plt.plot(t, x_new)
                # plt.plot(t, label, "X")
                # plt.show()
        else:
            count = count + 1
    return np.array(data_x), np.array(data_y)


def seqLength(data):
    b, c, w, h = data.size() #5 1 72 3
    data = data.view(b, w, h)
    len_data = []
    for i in range(b):
        count = 0
        for j in range(h):
            if data[i][0][j] != 0:
                count = count + 1
        len_data.append(count)

    len_data = np.array(len_data)
    len_data = torch.from_numpy(len_data)
    return data.view(b, c, w, h), len_data


def diffseq(data):
    b, c, w, h = data.size()
    data = data.numpy()
    data = data.reshape(b,-1)

    X = []
    for i in range(b):
        x_i = []
        for j in range(w-1):
            x = data[i][j] = data[i][j+1] - data[i][j]
            x_i.append(x)
        x_i.append(0)
        X.append(x_i)

    X = np.array(X)
    X = X.reshape(b, c, w, h)
    return torch.from_numpy(X)

def BlockSliding(data, label = False):
    if label:
        b, w = data.size()
        X = []
        for i in range(b):
            x = torch.squeeze(data[i])
            x = np.array(x)
            x = sliding_window(x, 8, 4)

            x = x.reshape(x.shape[0] * x.shape[1])
            X.append(x)
        X = np.array(X)
        X = torch.from_numpy(X)
        return X.view(b, -1)
    else:
        b, c, w, h = data.size()
        X = []
        for i in range(b):
            x = torch.squeeze(data[i])
            x = np.array(x)
            x = sliding_window(x, 8, 4)

            x = x.reshape(x.shape[0]*x.shape[1])
            X.append(x)
        X = np.array(X)
        X = torch.from_numpy(X)
        return X.view(b,c,-1,h)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


t = None
def timeit(name = ''):
    global t
    if t is None:
        print('timer start')
        t = time()
        return
    print(name,int((time()-t)*1000))
    t = time()


def get_f1_score(pred_choice, target):
    # TP    predict and label both eq to 1
    tp = ((pred_choice == 1) & (target.data == 1)).cpu().sum()
    # TN    predict and label both eq to 0
    tn = ((pred_choice == 0) & (target.data == 0)).cpu().sum()
    # FN    predict 0 label 1
    fn = ((pred_choice == 0) & (target.data == 1)).cpu().sum()
    # FP    predict 1 label 0
    fp = ((pred_choice == 1) & (target.data == 0)).cpu().sum()
    return tp, tn, fn, fp

def ToDirection(quat,vec):
    x_2 = quat[1] * 2.0
    y_2 = quat[2] * 2.0
    z_2 = quat[3] * 2.0
    x2_2 = quat[1] * x_2
    y2_2 = quat[2] * y_2
    z2_2 = quat[3] * z_2
    xy_2 = quat[1] * y_2
    xz_2 = quat[1] * z_2
    yz_2 = quat[2] * z_2
    wx_2 = quat[0] * x_2
    wy_2 = quat[0] * y_2
    wz_2 = quat[0] * z_2
    vec3 = np.zeros(3,)
    vec3[0] = (1 - (y2_2 + z2_2)) * vec[0] + (xy_2 - wz_2) * vec[1] + (xz_2 + wy_2) * vec[2]
    vec3[1] = (xy_2 + wz_2) * vec[0] + (1 - (x2_2 + z2_2)) * vec[1] + (yz_2 - wx_2) * vec[2]
    vec3[2] = (xz_2 - wy_2) * vec[0] + (yz_2 + wx_2) * vec[1] + (1 - (x2_2 + y2_2)) * vec[2]
    return vec3