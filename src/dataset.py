import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CSL_Continuous(Dataset):
    def \
            __init__(self, data_path, dict_path, corpus_path, frames=12, train=True, transform=None):
        super(CSL_Continuous, self).__init__()
        self.data_path = data_path
        self.dict_path = dict_path
        self.corpus_path = corpus_path
        self.frames = frames
        self.train = train
        self.transform = transform
        self.num_sentences = len(os.listdir(self.data_path))
        self.signers = 50
        self.repetition = 5
        if self.train:
            self.videos_per_folder = int(0.8 * self.signers * self.repetition)
        else:
            self.videos_per_folder = 2
        self.dict = {'<pad>': 0, '<sos>': 1, '<eos>': 2}  # 补全、句首、句尾
        self.output_dim = 3
        try:
            dict_file = open(self.dict_path, 'r', encoding='utf-8')
            for line in dict_file.readlines():
                line = line.strip().split('\t')
                if '（' in line[1] and '）' in line[1]:
                    for delimeter in ['（', '）', '、']:
                        line[1] = line[1].replace(delimeter, " ")
                    words = line[1].split()
                else:
                    words = [line[1]]
                for word in words:
                    self.dict[word] = self.output_dim
                self.output_dim += 1
        except Exception as e:
            raise
        self.data_folder = []
        try:
            self.data_folder = [data_path]
        except Exception as e:
            raise
        self.corpus = {}
        self.unknown = set()
        try:
            corpus_file = open(self.corpus_path, 'r', encoding='utf-8')
            for line in corpus_file.readlines():
                line = line.strip().split()
                raw_sentence = (line[1] + '.')[:-1]
                paired = [False for i in range(len(line[1]))]
                for token in sorted(self.dict, key=len, reverse=True):
                    index = raw_sentence.find(token)
                    if index != -1 and not paired[index]:
                        line[1] = line[1].replace(token, " " + token + " ")
                        for i in range(len(token)):
                            paired[index + i] = True
                tokens = [self.dict['<sos>']]
                for token in line[1].split():
                    if token in self.dict:
                        tokens.append(self.dict[token])
                    else:
                        self.unknown.add(token)
                tokens.append(self.dict['<eos>'])
                self.corpus[line[0]] = tokens
        except Exception as e:
            raise
        length = [len(tokens) for key, tokens in self.corpus.items()]
        self.max_length = max(length)
        for key, tokens in self.corpus.items():
            if len(tokens) < self.max_length:
                tokens.extend([self.dict['<pad>']] * (self.max_length - len(tokens)))

    def read_images_new(self, folder_path):
        images = []
        for i in range(0, self.frames):
            image = Image.open(os.path.join(folder_path, '{:06d}.jpg'.format(i)))
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.permute(1, 0, 2, 3)
        return images

    def __len__(self):
        return self.num_sentences * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[0]
        selected_folders = [top_folder]
        if self.train:
            selected_folder = selected_folders[0]
        else:
            try:
                selected_folder = selected_folders[0]
            except:
                for j in range(250 - len(selected_folders)):
                    selected_folders.append(selected_folders[0])
                selected_folder = selected_folders[0]
        images = self.read_images_new(selected_folder)
        tokens = torch.LongTensor(self.corpus['{:06d}'.format(1)])
        return images, tokens
