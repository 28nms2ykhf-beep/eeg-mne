import torch

DATA_ROOT = "/home/sv25/Desktop/eeg_epochs_output"
ANNOTATION_CSV = "/home/sv25/Desktop/ground_truth(Sheet1).csv"

BATCH_SIZE = 32
SAMPLES_PER_SESSION = 200
NUM_ELECTRODES = 18
CHUNK_SIZE = 4096
NUM_CLASSES = 2

F1 = 8
F2 = 16
D = 2
KERNEL_1 = 64
KERNEL_2 = 16
DROPOUT = 0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 30
TEST_RATIO = 0.2
VAL_RATIO = 0.1
SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
