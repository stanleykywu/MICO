import numpy as np
import torch
import csv
import os
import imagehash
import torchvision.transforms as T

from tqdm.notebook import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_model, CNN, LIRA

transform = T.ToPILImage()
CHALLENGE = "provided_data/cifar10"
LEN_TRAINING = 50000
LEN_CHALLENGE = 100
NUM_SHADOW = 30
LEARNING_RATE = 0.005
BATCH_SIZE = 32
EPOCHS = 50
MOMENTUM = 0
GAMMA = 0.96
SCHEDULER_STEP = 1

scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final', 'train']

dataset = load_cifar10()

collisions = 0
hash_to_ind = {}
for i, data in tqdm(enumerate(dataset), desc="verifying hash uniqueness", total=len(dataset)):
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

np.save("CIFAR10_indice_track.npy", indice_track)
indice_track = np.load("CIFAR10_indice_track.npy", allow_pickle=True).item()

criterion = torch.nn.CrossEntropyLoss()

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        root = os.path.join(CHALLENGE, scenario, phase)
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
            saved_models_dir='CarliniShadowModels/CNN_MICO_CIFAR',
            device='cuda',
        )