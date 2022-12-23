import numpy as np
import torch
import csv
import os

from tqdm.notebook import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_model, CNN, train_models, offline_attack

CHALLENGE = "provided_data/cifar10"
NUM_SHADOW = 30
LEN_TRAINING = 50000

scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final', 'train']

dataset = load_cifar10()

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        root = os.path.join(CHALLENGE, scenario, phase)
        for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
            path = os.path.join(root, model_folder)
            challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
            challenge_points = challenge_dataset.get_challenges()

            # LiRA Attack
            model = load_model('cifar10', path)
            scores, observed_confidence, confs_out = offline_attack(
                model, 
                challenge_points,
                NUM_SHADOW,
                f'CarliniShadowModels/CNN_MICO_CIFAR/{scenario}/{phase}',
                'cuda',
            )
            predictions = np.array([score[0] for score in scores])

            assert np.all((0 <= predictions) & (predictions <= 1))

            with open(os.path.join(path, "lira_prediction.csv"), "w") as f:
                 csv.writer(f).writerow(predictions)