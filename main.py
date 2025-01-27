import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from data_loader import get_dataloaders
from faceXhubert import FaceXHuBERT
from torch.utils.data import DataLoader

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("losses.png")
    plt.close()


def trainer(args, train_loader, valid_loader, model, optimizer, criterion, epoch):
    save_path = os.path.join(args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for e in range(1, epoch + 1):
        model.train()
        for i, (audio, vertice, template, one_hot, file_name) in enumerate(train_loader):
            vertice = str(vertice[0])
            vertice = np.load(vertice,allow_pickle=True)
            vertice = vertice.astype(np.float32)
            vertice = torch.from_numpy(vertice)
            vertice = torch.unsqueeze(vertice,0)
            audio, vertice, template, one_hot = (
                audio.to(args.device),
                vertice.to(args.device),
                template.to(args.device),
                one_hot.to(args.device),
            )
            optimizer.zero_grad()

            vertice_out, loss = model(audio, template, vertice, one_hot, criterion, use_teacher_forcing=True)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{e}], Step [{i}], Loss: {loss.item()}")
        if e % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_path,'{}_model.pth'.format(e)))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for audio, vertice, template, one_hot, _ in valid_loader:
                vertice = str(vertice[0])
                vertice = np.load(vertice,allow_pickle=True)
                vertice = vertice.astype(np.float32)
                vertice = torch.from_numpy(vertice)
                vertice = torch.unsqueeze(vertice,0)
                audio, vertice, template, one_hot = (
                    audio.to(args.device),
                    vertice.to(args.device),
                    template.to(args.device),
                    one_hot.to(args.device),
                )
                _, loss = model(audio, template, vertice, one_hot, criterion, use_teacher_forcing=False)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)
        print(f"Epoch [{e}] Validation Loss: {avg_val_loss:.4f}")


@torch.no_grad()
def test(args, model, test_loader,epoch):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        vertice = str(vertice[0])
        vertice = np.load(vertice, allow_pickle=True)
        vertice = vertice.astype(np.float32)
        vertice = torch.from_numpy(vertice)
        vertice = torch.unsqueeze(vertice, 0)
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:,iter,:]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()
            np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()
                np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_condition_"+condition_subject+".npy"), prediction.detach().cpu().numpy())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis using Self-Supervised Speech Representation Learning')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="multiface", help='Name of the dataset folder. eg: multiface')
    parser.add_argument("--vertice_dim", type=int, default=6172*3, help='number of vertices - 6172*3 for multiface dataset')
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default= "wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth vertex data')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=5, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="save", help='path to save the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="1 2 3 6 7 8 9 10 11 12 13")
    parser.add_argument("--val_subjects", type=str, default="1 2 3 6 7 8 9 10 11 12 13")
    parser.add_argument("--test_subjects", type=str, default="1 2 3 4 5 6 7 8 9 10 11 12 13")
    parser.add_argument("--input_fps", type=int, default=50, help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=30, help='fps of the visual data, multiface is available at 30 Hz')
    args = parser.parse_args()

    dataset = get_dataloaders(args)
    train_loader = dataset["train"]
    valid_loader = dataset["valid"]

    model = FaceXHuBERT(args).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()

    trainer(args, train_loader, valid_loader, model, optimizer, criterion, 100)


if __name__ == "__main__":
    main()