import os
import copy

import numpy as np
from scipy.stats import norm
from tqdm.notebook import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def reset_weights(model):
    """
    Reset the weights of provided model in place.
    """
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def evaluate_accuracy(model, data_loader, device):
    total_correct = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        total_correct += (torch.max(outputs, dim=1)[1] == labels).sum()
    return total_correct / len(data_loader.dataset)

def train_shadow_model(
    model,
    random_sample,
    shadow_model_number,
    criterion,
    lr,
    epochs,
    momentum,
    gamma,
    scheduler_step,
    optimizer,
    saved_models_dir,
    device,
):
    """Helper function to train individual shadow models

    Parameters
    ----------
        model : PyTorch model
        random_sample : PyTorch Dataloader
            The randomly generated dataset to train a single shadow model
        shadow_model_number : int
            Which shadow model we are training
        criterion : torch.nn.Criterion
            Criterion for training
        lr : float
            Learning Rate
        epochs : int
            Number of epochs to run
        Momentum : float
            Momentum parameter
        gamma : float
            Multiplier for learning rate scheduler
        scheduler_step : int
            Number of epochs before multiplying learning rate by gamma
        optimizer : torch.Optimizer
            Optimizer for torch training
        device : str
    """
    shadow_model = copy.deepcopy(model).to(device)
    reset_weights(shadow_model)

    opt = optimizer(
        shadow_model.parameters(), lr=lr, momentum=momentum
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, scheduler_step, gamma)
    shadow_model.train()
    print("\n")
    print("-" * 8)
    print("Training Shadow Model...\n")
    for _ in range(epochs):
        running_loss = 0
        for (inputs, labels) in random_sample:
            opt.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = shadow_model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

    print(
        f"Shadow Model Final Training Error: {running_loss/len(random_sample.dataset):.4}\n"
        + f"Shadow Model Final Training Accuracy: {evaluate_accuracy(shadow_model, random_sample, device)*100:.5}%"
    )
    print("-" * 8)

    shadow_model.eval()
    torch.save(
        shadow_model,
        f"{saved_models_dir}/shadow_model_{shadow_model_number}",
    )

def train_models(
    model,
    indice_track,
    dataset,
    train_size,
    num_shadow_models,
    lr,
    batch_size,
    epochs,
    optimizer,
    criterion,
    momentum,
    gamma,
    scheduler_step,
    saved_models_dir,
    device,
    seed=0,
):
    random = np.random.RandomState(seed)
    used_indices = []
    for phase, indices in indice_track.items():
        used_indices += indices
    used_indices = set(used_indices)
    out_indices = []
    for ind in range(len(dataset)):
        if ind not in used_indices:
            out_indices.append(ind)
    out_indices = np.array(out_indices)
    
    train_size = min(len(out_indices), train_size)
    print(f"There are {len(out_indices)} number of points not used as challenge points")
    print(f"Each shadow model will be trained on: {train_size} points")
    
    shadow_indices = [
        random.choice(
            out_indices,
            train_size,
            replace=False,
        )
        for _ in range(num_shadow_models)
    ]

    for shadow_model in range(num_shadow_models):
        training_subset = torch.utils.data.Subset(
            dataset, shadow_indices[shadow_model]
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=training_subset, batch_size=batch_size, shuffle=True
        )
        train_shadow_model(
            model=model,
            random_sample=train_loader,
            shadow_model_number=shadow_model + 1,
            criterion=criterion,
            lr=lr,
            epochs=epochs,
            momentum=momentum,
            gamma=gamma,
            scheduler_step=scheduler_step,
            optimizer=optimizer,
            saved_models_dir=saved_models_dir,
            device=device,
        )

def logit_scaling(p):
    """Perform logit scaling so that the model's confidence is
    approximately normally distributed

    Parameters
    ----------
        p : torch.Tensor
            A tensor containing some model's confidence scores

    Returns
    -------
        phi(p) : PyTorch.Tensor(float)
            The scaled model confidences
    """
    assert isinstance(p, torch.Tensor)
    # for stability purposes
    return torch.log(p / (1 - p + 1e-50))

def model_confidence(
    model, 
    datapoints,
    device,
):
    """Helper function to calculate the model confidence on provided examples

    Model confidence is defined as softmax probability of the highest probability class

    Parameters
    ----------
        model : PyTorch model
            A Pytorch machine learning model
        datapoints : torch.Dataset
            Dataset to get confidence scores for
        device : str

    Returns
    -------
        model_confidence : List(float)
            softmax(model(x_n)) on the y_n class for nth datapoint
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    y = [data[1] for data in datapoints]

    with torch.no_grad():
        model = model.to(device)
        x = [data[0].to(device) for data in datapoints]
        predictions = model.forward(torch.stack(x))

    softmax_values = softmax(predictions)
    confidences = torch.tensor(
        [confidence[c] for confidence, c in zip(softmax_values, y)]
    )
    return confidences

def offline_attack(
    target_model, 
    challenge_points,
    num_shadow_models,
    saved_models_dir,
    device,
):
    """
    Carlini's offline membership inference attack.

    Parameters
    ----------
        target_model : torch model
            Provided target model
        challenge_points : torch.Tensor
            Provided points to evaluate on
        num_shadow_models : int
        saved_models_dir : str
        device : str

    Returns
    -------
        scores, confs_out : tuple(list(float), list(float))
            The scores of each target point and their out-distribution logit confidences
    """
    confs_out = [[] for _ in range(len(challenge_points))]  # want per point
    scores = []

    for i in tqdm(
        range(num_shadow_models), desc=f"Computing Out-Distribution Logits"
    ):
        shadow_model = torch.load(
            f"{saved_models_dir}/shadow_model_{i+1}",
            map_location=device,
        )
        shadow_model.eval()

        confs_scaled = logit_scaling(
            model_confidence(shadow_model, challenge_points, device)
        )
        for idx, cs in enumerate(confs_scaled):
            confs_out[idx].append(cs)

    observed_confidence = logit_scaling(
        model_confidence(target_model, challenge_points, device)
    )
    for i, shadow_confs in enumerate(
        tqdm(confs_out, desc="Running Likelihood Estimation")
    ):
        shadow_confs = torch.Tensor(shadow_confs)
        shadow_confs = shadow_confs[torch.isfinite(shadow_confs)]
        mean_out = torch.mean(shadow_confs).cpu()
        std_out = torch.std(shadow_confs).cpu()

        score = norm.cdf(
            observed_confidence[i].cpu(), loc=mean_out, scale=std_out + 1e-30
        )
        scores.append((score, shadow_confs))

    return scores, observed_confidence, confs_out
