#
DATAFORMAT = 0
CONTAIN_NULLCLASS = 1
NO_NLSTM = 0

SEVER = 0

#
CHANNELS_OBJECT = 51


CHANNELS_NUM_113 = 105
CHANNELS_NUM_50_TO_42  = 42
# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 50#113

NB_SENSOR_CHANNELS_113 = 113
# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24
# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

BASE_lr =3* 10e-5
STEP = 5
EPOCH = 300
# Batch Size
BATCH_SIZE = 100
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

