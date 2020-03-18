CHECK_POINTS = 10

EPOCH = 500

BASE_lr = 3*10e-5

pretrain = False

Data_length = 72

Short_Length = 8

attention = False

acc3channel = True

channel = 1

SLIDING_WINDOW_LENGTH = 24
# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

CHANNELS_OBJECT = 113 #51

BATCH_SIZE = 50
# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18
# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

CHECK_POINTS = 10
# Number filters convolutional layers
NUM_FILTERS = 64
# Size filters convolutional layers
FILTER_SIZE = 3
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128
NUM_LSTM_LAYERS = 4