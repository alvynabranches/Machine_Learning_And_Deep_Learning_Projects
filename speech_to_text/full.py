import torch
import torchaudio
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
from .utils import TextProcess
import argparse
from collections import OrderedDict

# NOTE: add time stretch
class SpecAugment(nn.Module):
    def __init__(self, rate, policy=3, freq_mask=15, time_mask=35):
        super(SpecAugment, self).__init__()
        
        self.rate = rate
        
        self.specaug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        
        self.specaug2 = nn.Sequential(
            torchaudio.transforms.FrequencytMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencytMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask)
        )
        
        policies = {1: self.policy1, 2: self.policy2, 3: self.policy3}
        self._forward = policies[policy]
        
    def forward(self, x):
        return self._forward(x)
    
    def policy1(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug(x)
        return x
    
    def policy2(self, x):
        probability = torch.rand(1, 1).item()
        if self.rate > probability:
            return self.specaug2(x)
        return x
    
    def policy3(self, x):
        probability = torch.rand(1, 1).item()
        if probability > 0.5:
            return self.policy1(x)
        return self.policy2(x)

class LogMelSpec(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=128, win_length=160, hop_length=80):
        super(LogMelSpec, self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, 
            win_length=win_length, hop_length=hop_length
        )
        
        def forward(self, x):
            x = self.transform(x) # mel spectrogram
            x = np.log(x + 1e-14) # logrithmic, add small value to avoid inf
            return x
    
def get_featurizer(sample_rate, n_feats=81):
    return LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)

class Data(torch.utils.data.Dataset):
    # this makes it easier to be overide in argeparse
    parameters = {
        'sample_rate': 8000, 'n_feats': 81, 
        'specaug_rate': 0.5, 'specaug_policy': 3,
        'time_mask': 70, 'freq_mask': 15
    }
    
    def __init__(self, json_path, sample_rate, n_feats, specaug_rate, specaug_policy, 
                 time_mask, freq_mask, valid=False, shuffle=True, text_to_int=True, log_ex=True):
        self.log_ex = log_ex
        self.text_process = TextProcess()
        
        print(f'Loading data json file from {json_path}')
        self.data = pd.read_json(json_path, lines=True)
        
        if valid:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80)
            )
        else:
            self.audio_transforms = torch.nn.Sequential(
                LogMelSpec(sample_rate=sample_rate, n_mels=n_feats, win_length=160, hop_length=80),
                SpecAugment(specaug_rate, specaug_policy, freq_mask, time_mask)
            )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        file_path = ''
        try:
            file_path = self.data.key.iloc[idx]
            waveform, _ = torchaudio.load(file_path)
            label = self.text_process.text_to_int_sequence(self.data['text'].iloc[idx])
            spectrogram = self.audio_transforms(waveform) # channel, features, time
            spec_len = spectrogram.shape[-1] // 2
            label_len = len(label)
            if spec_len < label_len:
                raise Exception('spectrogram len is bigger then label len')
            if spectrogram[0] > 1:
                raise Exception(f'dual channel, skipping audio file {file_path}')
            if spectrogram.shape[2] > 1650:
                raise Exception(f'spectrogram to big. size {spectrogram.shape[2]}')
            if label_len == 0:
                raise Exception(f'label len is zero... skipping {file_path}')
        except Exception as e:
            if self.log_ex:
                print(str(e), file_path)
                
            return self.__getitem__(idx-1 if idx!=0 else idx + 1)
        
    def describe(self):
        return self.data.describe()

def collate_fn_padd(data):
    '''
        Padds batch of variable length
        
        NOTE: it converts things ToTensor manually since the ToTensor transform
        assume it takes in images rather than arbitrary tensors. 
    '''
    # print(data)
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (spectrogram, label, input_length, label_length) in data:
        if spectrogram is None:
            continue
        # print(spectrogram.shape)
        spectrograms.append(spectrogram.squeeze(0).transpose(0, 1))
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
        input_lengths.append(input_length)
        label_lengths.append(label_length)
        
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    input_lengths = input_lengths
    # print(spectrograms.shape)
    label_lengths = label_lengths
    #  ## compute mask
    # mask = (batch != 0).cuda(gpu)
    # return batch, lengths, mask
    return spectrograms, labels, input_lengths, label_lengths

# model.py
class ActDropNormCNN1D(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormCNN1D, self)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
        
    def forward(self, x):
        x = x.transpose(1, 2)
        # x = self.norm(self.dropout(F..gelu(X)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x

class SpeechRecognition(nn.Module):
    hyper_parameters = {
        'num_classes': 29, 
        'n_feats': 81, 
        'dropout': 0.1, 
        'hidden_size': 1024, 
        'num_layers': 1 
    }
    
    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 10, 2, padding=10//2), 
            ActDropNormCNN1D(n_feats, dropout), 
        )
        self.dense = nn.Sequential(
            nn.Linear(n_feats, 128), 
            nn.LayerNorm(128), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(128, 128), 
            nn.LayerNorm(128), 
            nn.GELU(), 
            nn.Dropout(dropout), 
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, 
                            num_layers=num_layers, dropout=0.0, 
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)
    
    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))
        
    def forward(self, x, hidden):
        x = x.squeeze(1) # batch, feature, time
        x = self.cnn(x) # batch, time, feature
        x = self.dense(x) # batch, time, feature
        x = x.transpose(0, 1) # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout2(F.gelu(self.layer_norm2(out))) # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)
    
# optimize_graph.py
""" Freezes and optimize the model. Use after training."""

def trace(model):
    model.eval()
    x = torch.rand(1, 81, 300)
    hidden = model._init_hidden(1)
    traced = torch.jit.trace(model, (x, hidden))
    return traced

def main(args):
    print("loading model from ", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    h_params = SpeechRecognition.hyper_parameters
    model = SpeechRecognition(**h_params)
    
    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace('model.', '') # remove 'model.'
        new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        
        print('tracing model...')
        traced_model = trace(model)
        print('saving to', args.save_path)
        traced_model.save(args.save_path)
        print('Done!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing the wakeword engine')
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True, 
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True, 
                        help='path ot save optimized model')
    
    args= parser.parse_args()
    main(args)