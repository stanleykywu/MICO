import numpy as np
import torch
import csv
import os

from tqdm import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_model, CNN, train_models, offline_attack

CHALLENGE = "provided_data/cifar10"
NUM_SHADOW = 30
LEN_TRAINING = 50000
INDICE_FILE = "CIFAR10_indice_track.npy"

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
        indice_track[scenario] = {}
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            indice_track[scenario][phase] = {}
            for i, model_folder in tqdm(enumerate(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1]))), desc="model and indice search", total=len(os.listdir(root))):
                path = os.path.join(root, model_folder)
                challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
                challenge_points = challenge_dataset.get_challenges()

                indice_track[scenario][phase][i] = set()

                for point in challenge_points:
                    image_hash = imagehash.phash(transform(point[0]), hash_size=8)
                    indice_track[scenario][phase][i].add(hash_to_ind[image_hash])

    np.save(INDICE_FILE, indice_track)


SHADOW_DIRECTORY = "CarliniShadowModels/CNN_MICO_CIFAR"
parameters = {
    'cifar10_inf': {
        "LEN_TRAINING": 50000,
        "LEN_CHALLENGE": 100,
        "NUM_SHADOW": 30,
        "LEARNING_RATE": 0.005,
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
        "MOMENTUM": 0,
        "DEVICE":'cuda',
        "max_physical_batch_size": 128,
        "lr_scheduler_gamma": 0.96,
        "lr_scheduler_step": 1,
        "secure_mode": False,
    },
    'cifar10_hi': {
        "LEN_TRAINING": 50000,
        "LEN_CHALLENGE": 100,
        "NUM_SHADOW": 30,
        "LEARNING_RATE": 0.5,
        "BATCH_SIZE": 512,
        "EPOCHS": 50,
        "MOMENTUM": 0,
        "DEVICE":'cuda',
        "max_physical_batch_size": 128,
        "lr_scheduler_gamma": 0.96,
        "lr_scheduler_step": 1,
        "secure_mode": True,
        # DP Parameters
        "max_grad_norm": 2.6,
        "target_epsilon": 10.0,
        "target_delta": 1e-5,
    },
    'cifar10_lo': {
        "LEN_TRAINING": 50000,
        "LEN_CHALLENGE": 100,
        "NUM_SHADOW": 30,
        "LEARNING_RATE": 0.5,
        "BATCH_SIZE": 512,
        "EPOCHS": 50,
        "MOMENTUM": 0,
        "DEVICE":'cuda',
        "max_physical_batch_size": 128,
        "lr_scheduler_gamma": 0.96,
        "lr_scheduler_step": 1,
        "secure_mode": True,
        # DP Parameters
        "max_grad_norm": 2.6,
        "target_epsilon": 4.0,
        "target_delta": 1e-5,
    }
}
optimizer = torch.optim.SGD
criterion = torch.nn.CrossEntropyLoss()

for scenario in tqdm(scenarios, desc="scenario"):
    scenario_parameter = parameters[scenario]
    for phase in tqdm(phases, desc="phase"):
        shadow_folder = f"{SHADOW_DIRECTORY}/{scenario}/{phase}"
        if not os.path.isdir(shadow_folder):
            os.mkdir(shadow_folder)
            
        train_models(
            dataset_name='cifar10',
            model=CNN(),
            indice_track=indice_track[scenario][phase],
            dataset=dataset,
            train_size=scenario_parameter['LEN_TRAINING'],
            num_shadow_models=scenario_parameter['NUM_SHADOW'],
            lr=scenario_parameter['LEARNING_RATE'],
            batch_size=scenario_parameter['BATCH_SIZE'],
            epochs=scenario_parameter['EPOCHS'],
            optimizer=optimizer,
            criterion=criterion,
            momentum=scenario_parameter['MOMENTUM'],
            saved_models_dir=shadow_folder,
            device=scenario_parameter['DEVICE'],
            dp='target_epsilon' in scenario_parameter,
            max_physical_batch_size=scenario_parameter['max_physical_batch_size'],
            lr_scheduler_gamma=scenario_parameter['lr_scheduler_gamma'],
            lr_scheduler_step=scenario_parameter['lr_scheduler_step'],
            secure_mode=scenario_parameter['secure_mode'],
            max_grad_norm=scenario_parameter.get('max_grad_norm', None),
            target_epsilon=scenario_parameter.get('target_epsilon', None),
            target_delta=scenario_parameter.get('target_delta', None),
        )