import os
import copy
import warnings

import numpy as np
from scipy.stats import norm
from tqdm import tqdm, trange
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from prv_accountant.dpsgd import find_noise_multiplier
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch.optim as optim

from .modified_training import train_purchase, train_cifar
from .accountant import PRVAccountant


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

def train_shadow_purchase(
    model,
    random_sample,
    criterion,
    lr,
    batch_size,
    epochs,
    momentum,
    optimizer,
    device,
    dp,
    lr_scheduler_gamma,
    lr_scheduler_step,
    secure_mode,
    max_grad_norm,
    target_epsilon,
    target_delta,
    max_physical_batch_size,
):
    warnings.filterwarnings(action="ignore", module="opacus", message=".*Secure RNG turned off")
    warnings.filterwarnings(action="ignore", module="torch", message=".*Using a non-full backward hook")
    assert ModuleValidator.is_valid(model)
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    # Not the same as batch_size / len(train_dataset)
    sample_rate = 1 / len(random_sample)
    num_steps = int(len(random_sample) * epochs)

    if dp:
        noise_multiplier = find_noise_multiplier(
            sampling_probability=sample_rate,
            num_steps=num_steps,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            eps_error=0.1
        )

        privacy_engine = PrivacyEngine(secure_mode=secure_mode)

        # Override Opacus accountant
        # Revise if https://github.com/pytorch/opacus/pull/493 is merged
        privacy_engine.accountant = PRVAccountant(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            max_steps=num_steps,
            eps_error=0.1,
            delta_error=1e-9)

        model, optimizer, random_sample = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=random_sample,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
            noise_generator=None,
        )

        print(f"Training using DP-SGD with {optimizer.original_optimizer.__class__.__name__} optimizer\n"
             f"  noise multiplier σ = {optimizer.noise_multiplier},\n"
             f"  clipping norm C = {optimizer.max_grad_norm:},\n"
             f"  average batch size L = {batch_size},\n"
             f"  sample rate = {sample_rate},\n"
             f"  for {epochs} epochs ({num_steps} steps)\n"
             f"  to target ε = {target_epsilon}, δ = {target_delta}")

        compute_epsilon: Optional[Callable[[float], float]] = lambda delta: privacy_engine.get_epsilon(delta=delta)
    else:
        print(f"Training using SGD with {optimizer.__class__.__name__} optimizer\n"
             f"  batch size L = {batch_size},\n"
             f"  for {epochs} epochs ({num_steps} steps)")
        compute_epsilon = None

    # Must be initialized after attaching the privacy engine.
    # See https://discuss.pytorch.org/t/how-to-use-lr-scheduler-in-opacus/111718
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

    pbar = trange(epochs, desc="Epoch", unit="epoch", position=0, leave=True, disable=None)
    for epoch in pbar:
        pbar.set_postfix(lr=f"{scheduler.get_last_lr()}")
        train_purchase(
            model=model, 
            device=device,
            train_loader=random_sample, 
            criterion=criterion, 
            optimizer=optimizer, 
            epoch=epoch + 1, 
            batch_size=batch_size,
            max_physical_batch_size=max_physical_batch_size,
            dp=dp,
            compute_epsilon=compute_epsilon
        )
        scheduler.step()
        
    return model


def train_shadow_cifar(
    model,
    random_sample,
    criterion,
    lr,
    batch_size,
    epochs,
    momentum,
    optimizer,
    device,
    dp,
    lr_scheduler_gamma,
    lr_scheduler_step,
    secure_mode,
    max_grad_norm,
    target_epsilon,
    target_delta,
    max_physical_batch_size,
):
    # Supress warnings
    warnings.filterwarnings(action="ignore", module="opacus", message=".*Secure RNG turned off")
    warnings.filterwarnings(action="ignore", module="torch", message=".*Using a non-full backward hook")

    assert ModuleValidator.is_valid(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = optimizer(model.parameters(), lr=lr, momentum=momentum)
    
    # Not the same as batch_size / len(train_dataset)
    sample_rate = 1 / len(random_sample)
    num_steps = int(len(random_sample) * epochs)

    if dp:
        noise_multiplier = find_noise_multiplier(
            sampling_probability=sample_rate,
            num_steps=num_steps,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            eps_error=0.1
        )

        privacy_engine = PrivacyEngine(secure_mode=secure_mode)

        # Override Opacus accountant
        # Revise if https://github.com/pytorch/opacus/pull/493 is merged
        privacy_engine.accountant = PRVAccountant(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            max_steps=num_steps,
            eps_error=0.1,
            delta_error=1e-9)

        model, optimizer, random_sample = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=random_sample,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
            noise_generator=None,
        )

        print(f"Training using DP-SGD with {optimizer.original_optimizer.__class__.__name__} optimizer\n"
             f"  noise multiplier σ = {optimizer.noise_multiplier},\n"
             f"  clipping norm C = {optimizer.max_grad_norm:},\n"
             f"  average batch size L = {batch_size},\n"
             f"  sample rate = {sample_rate},\n"
             f"  for {epochs} epochs ({num_steps} steps)\n"
             f"  to target ε = {target_epsilon}, δ = {target_delta}")

        compute_epsilon: Optional[Callable[[float], float]] = lambda delta: privacy_engine.get_epsilon(delta=delta)
    else:
        print(f"Training using SGD with {optimizer.__class__.__name__} optimizer\n"
             f"  batch size L = {batch_size},\n"
             f"  for {epochs} epochs ({num_steps} steps)")
        compute_epsilon = None

    # Must be initialized after attaching the privacy engine.
    # See https://discuss.pytorch.org/t/how-to-use-lr-scheduler-in-opacus/111718
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)

    pbar = trange(epochs, desc="Epoch", unit="epoch", position=0, leave=True, disable=None)
    for epoch in pbar:
        pbar.set_postfix(lr=f"{scheduler.get_last_lr()}")
        train_cifar(
            model=model, 
            device=device,
            train_loader=random_sample, 
            criterion=criterion, 
            optimizer=optimizer, 
            epoch=epoch + 1, 
            batch_size=batch_size,
            max_physical_batch_size=max_physical_batch_size,
            dp=dp,
            compute_epsilon=compute_epsilon
        )
        scheduler.step()
        
    return model

def train_shadow_model(
    dataset_name,
    model,
    random_sample,
    shadow_model_number,
    criterion,
    lr,
    batch_size,
    epochs,
    momentum,
    optimizer,
    saved_models_dir,
    device,
    dp,
    lr_scheduler_gamma,
    lr_scheduler_step,
    secure_mode,
    max_grad_norm,
    target_epsilon,
    target_delta,
    max_physical_batch_size,
):
    if dataset_name == "purchase100":
        shadow_model = train_shadow_purchase(
            model=model,
            random_sample=random_sample,
            criterion=criterion,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            optimizer=optimizer,
            device=device,
            dp=dp,
            lr_scheduler_gamma=lr_scheduler_gamma,
            lr_scheduler_step=lr_scheduler_step,
            secure_mode=secure_mode,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_physical_batch_size=max_physical_batch_size,
        )
    elif dataset_name == "cifar10":
        shadow_model = train_shadow_cifar(
            model=model,
            random_sample=random_sample,
            criterion=criterion,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            optimizer=optimizer,
            device=device,
            dp=dp,
            lr_scheduler_gamma=lr_scheduler_gamma,
            lr_scheduler_step=lr_scheduler_step,
            secure_mode=secure_mode,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_physical_batch_size=max_physical_batch_size,
        )
    else:
        return ValueError('Invalid dataset')

    shadow_model.eval()
    torch.save(
        shadow_model,
        f"{saved_models_dir}/shadow_model_{shadow_model_number}",
    )

def train_models(
    dataset_name,
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
    saved_models_dir,
    device,
    dp,
    lr_scheduler_gamma,
    lr_scheduler_step,
    secure_mode,
    max_grad_norm,
    target_epsilon,
    target_delta,
    max_physical_batch_size,
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
            dataset_name=dataset_name,
            model=copy.deepcopy(model),
            random_sample=train_loader,
            shadow_model_number=shadow_model + 1,
            criterion=criterion,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            momentum=momentum,
            optimizer=optimizer,
            saved_models_dir=saved_models_dir,
            device=device,
            dp=dp,
            lr_scheduler_gamma=lr_scheduler_gamma,
            lr_scheduler_step=lr_scheduler_step,
            secure_mode=secure_mode,
            max_grad_norm=max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_physical_batch_size=max_physical_batch_size,
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
    verbose,
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
        range(num_shadow_models), desc=f"Computing Out-Distribution Logits", disable=not verbose
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
        tqdm(confs_out, desc="Running Likelihood Estimation", disable=not verbose)
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
