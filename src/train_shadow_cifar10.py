import numpy as np
import torch
import csv
import os
import imagehash
import torchvision.transforms as T

from tqdm import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_model, CNN, train_models

transform = T.ToPILImage()
CHALLENGE = "provided_data/cifar10"
INDICE_FILE = "CIFAR10_indice_track.npy"
SHADOW_DIRECTORY = "CarliniShadowModels/CNN_MICO_CIFAR"
LEN_TRAINING = 50000
LEN_CHALLENGE = 100
NUM_SHADOW = 30
LEARNING_RATE = 0.005
BATCH_SIZE = 32
EPOCHS = 50
MOMENTUM = 0
GAMMA = 0.96
SCHEDULER_STEP = 1
DEVICE = 'cuda'

scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final', 'train']

dataset = load_cifar10()

if os.path.exists(INDICE_FILE):
    print("Loading indice dictionary...")
    indice_track = np.load(INDICE_FILE, allow_pickle=True).item()
else:
    collisions = 0
    hash_to_ind = {}
    for i, data in tqdm(enumerate(dataset), desc="Verifying hash uniqueness", total=len(dataset)):
        image_hash = imagehash.phash(transform(data[0]), hash_size=8)
        if image_hash in hash_to_ind:
            collisions += 1
        else:
            hash_to_ind[image_hash] = i

    assert collisions == 0

    indice_track = {}

    for scenario in tqdm(scenarios, desc="scenario"):
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            indice_track[phase] = {}
            for i, model_folder in tqdm(enumerate(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))), desc="model and indice search", total=len(os.listdir(root))):
                path = os.path.join(root, model_folder)
                challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
                challenge_points = challenge_dataset.get_challenges()

                indice_track[phase][i] = set()

                for point in challenge_points:
                    image_hash = imagehash.phash(transform(point[0]), hash_size=8)
                    indice_track[phase][i].add(hash_to_ind[image_hash])

    np.save(INDICE_FILE, indice_track)

optimizer = torch.optim.SGD
criterion = torch.nn.CrossEntropyLoss()

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        shadow_folder = f"{SHADOW_DIRECTORY}/{phase}"
        if not os.path.isdir(shadow_folder):
            os.mkdir(shadow_folder)
            
        train_models(
            model=CNN(),
            indice_track=indice_track[phase],
            dataset=dataset,
            train_size=LEN_TRAINING,
            num_shadow_models=NUM_SHADOW,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            optimizer=optimizer,
            criterion=criterion,
            momentum=MOMENTUM,
            gamma=GAMMA,
            scheduler_step=SCHEDULER_STEP,
            saved_models_dir=shadow_folder,
            device=DEVICE,
        )