# model.py
from torchegg.models import EEGNet

def create_eegnet(chunk_size, num_electrodes, F1, F2, D, num_classes,
                  kernal1, kernal2, dropout):
    model = EEGNet(chunk_size=chunk_size,
                   num_electrodes=num_electrodes,
                   F1=F1,
                   F2=F2,
                   D=D,
                   num_classes=num_classes,
                   kernal1=kernal1,
                   kernal2=kernal2,
                   dropout=dropout)
    return model
