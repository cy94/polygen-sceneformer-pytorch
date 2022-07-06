import argparse
import copy
from nltk.tokenize import word_tokenize
from torchvision.transforms import Compose
import torch
from torch.utils.data import DataLoader, Subset

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from transforms.scene import SeqToTensor, Augment_rotation, Augment_jitterring, \
    Get_cat_shift_info, Padding_joint, Add_Relations, Add_Descriptions, Add_Glove_Embeddings, Sentence_to_indices
from datasets.suncg_shift_seperate_dataset_deepsynth import SUNCG_Dataset
from separate_models.scene_shift_cat import scene_transformer
from utils.config import read_config
from tqdm import tqdm
import pickle


seed_everything(1)

symbols = []
idxs = {'<pad>':0}
max_sentences = 3
max_length = 50

def add_symbol(symbol):
    """Add a symbol to the dictionary and return its index
    If the symbol already exists in the dictionary this just returns
    the index"""
    if symbol not in idxs:
        idxs[symbol] = len(idxs)
        symbols.append(symbol)
    return idxs[symbol]

def run_training(cfg):
    transforms = [Augment_rotation(), Augment_jitterring(), Get_cat_shift_info(cfg), Add_Relations(), Add_Descriptions()]
    transforms.append(Padding_joint(cfg))
    transforms.append(SeqToTensor())
    t = Compose(transforms)

    trainval_set = SUNCG_Dataset(data_folder=cfg['data']['data_path'], list_path=cfg['data']['list_path'], transform=t)
    for ndx,sample in enumerate(tqdm(trainval_set)):
        if ndx == 1100:
            break
        sentence = ''.join(sample['description'][:max_sentences])
        tokens = list(word_tokenize(sentence))
        # pad to maximum length
        tokens += ['<pad>'] * (max_length - len(tokens))
        indices = []
        for symbol in tokens:
            indices.append(add_symbol(symbol))
    data_folder = cfg['data']['data_path']
    with open(f'{data_folder}/voc_dic.pkl', "wb") as f:
            pickle.dump(idxs, f, pickle.HIGHEST_PROTOCOL)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path', help='Path to config file')
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    run_training(cfg)