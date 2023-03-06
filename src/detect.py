import torch
import tqdm
import sys
from torch import nn
from cutFrame import Preprocess
from dataset import CSL_Continuous
from models.Seq2Seq import Encoder, Decoder, Seq2Seq
import os
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../picture',
                        type=str, help='Data path for testing')
    parser.add_argument('--label_path', default='../label/dictionary.txt',
                        type=str, help='Label path for testing')
    parser.add_argument('--model', default='3dresnet18',
                        type=str, help='Choose a model for testing')
    parser.add_argument('--model_path', default='../trainedModel/gestureDetection.pth',
                        type=str, help='Model state dict path')
    parser.add_argument('--num_classes', default=500,
                        type=int, help='Number of classes for testing')
    parser.add_argument('--batch_size', default=2,
                        type=int, help='Batch size for testing')
    parser.add_argument('--sample_size', default=128,
                        type=int, help='Sample size for testing')
    parser.add_argument('--sample_duration', default=48,
                        type=int, help='Sample duration for testing')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, dont use cuda')
    parser.add_argument('--cuda_devices', default='0',
                        type=str, help='Cuda visible devices')
    args = parser.parse_args()

    preprocess = Preprocess('../video', '../picture', 48)
    preprocess.begin()

    enc_hid_dim = 512
    emb_dim = 256
    dec_hid_dim = 512
    dropout = 0.5
    clip = 1
    log_interval = 100

    data_path = args.data_path
    label_path = args.label_path
    corpus_path = "../label/corpus.txt"
    model_path = args.model_path

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = torch.device("cpu")

    num_classes = args.num_classes
    batch_size = args.batch_size
    sample_size = args.sample_size
    sample_duration = args.sample_duration

    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    listdir = os.listdir(os.path.join(data_path))
    for i in range(0, len(listdir)):
        single_path = os.path.join(data_path, listdir[i])
        test_set = CSL_Continuous(data_path=single_path, dict_path=label_path,
                                  corpus_path=corpus_path, frames=sample_duration, train=False, transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        encoder = Encoder(lstm_hidden_size=enc_hid_dim, arch="resnet18").to(device)
        decoder = Decoder(output_dim=507, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim,
                          dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
        model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        model.eval()
        all_pred = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(test_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, labels, 0)
                output_dim = outputs.shape[-1]
                outputs = outputs[1:].view(-1, output_dim)
                target = labels.permute(1, 0)[1:].reshape(-1).cpu().numpy()
                prediction = torch.max(outputs, 1)[1]
                all_pred.extend(prediction.cpu().numpy())

        print('video name:'+str(listdir[i]))
        flag = 0
        for (num) in all_pred:
            if num == 2 and flag % 2 == 0:
                print('。')
                break
            elif num == 0:
                continue
            else:
                file = open('../label/dictionary.txt', 'r')
                for (tnum, value) in enumerate(file):
                    if tnum == num - 3 and flag % 2 == 0:
                        res = value
                        if tnum == 503:
                            res = value[4:]
                        else:
                            res = value[4:-1]
                        sys.stdout.write(res)
                file.close()
            flag += 1
