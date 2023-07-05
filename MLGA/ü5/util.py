import random
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_dataset(train_x, train_y, val_x, val_y):
    # plot dataset
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.grid("on")
    plt.xlim(-2, 2.5)
    plt.ylim(-1, 1.5)
    plt.title("Train Data")
    plt.scatter(train_x[train_y == 0, 0], train_x[train_y == 0, 1],
                c="blue", s=10)
    plt.scatter(train_x[train_y == 1, 0], train_x[train_y == 1, 1],
                c="orange", s=10)

    plt.subplot(1, 2, 2)
    plt.grid("on")
    plt.xlim(-2, 2.5)
    plt.ylim(-1, 1.5)
    plt.title("Validation Data")
    plt.scatter(val_x[val_y == 0, 0], val_x[val_y == 0, 1],
                c="blue", s=10)
    plt.scatter(val_x[val_y == 1, 0], val_x[val_y == 1, 1],
                c="orange", s=10)
    plt.show()


def parameters_to_vector(nn_param_group) -> torch.Tensor:
    """Convert parameters in groups to one vector

    Args:
        nn_param_group: nn param group
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    vec = []
    for group in nn_param_group:
        for param in group["params"]:
            vec.append(param.view(-1))
    return torch.cat(vec)


def set_params_to_net(params_vec, nn_param_group) -> None:
    """Rewrite the parameters in network given a vector of parameters

    Args:
        params_vec (Tensor): a single vector represents the parameters of a net.
        nn_param_group: target net to set to
    """
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for group in nn_param_group:
        for param in group["params"]:
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = params_vec[pointer:pointer + num_param].view_as(
                param).data

            # Increment the pointer
            pointer += num_param


def plot_all_models(model_dict, train_x, train_y, plot_density):
    fig, axes = plt.subplots((len(model_dict) + 1) // 2, 2, dpi=150,
                             squeeze=False,
                             figsize=(8, 3 * len(model_dict)))
    p_bar = tqdm(range(len(model_dict)))
    for i in p_bar:
        model_name, model = list(model_dict.items())[i]
    # for i, (model_name, model) in enumerate(model_dict.items()):
        plt.sca(axes[i // 2, i % 2])
        # plot grid for contour plots
        plt_range = np.arange(-4, 5.5, plot_density)
        plt_grid = np.stack(np.meshgrid(plt_range, plt_range), axis=-1)
        plt_grid_shape = plt_grid.shape[:2]
        flat_plt_grid = torch.as_tensor(np.reshape(plt_grid, [-1, 2]),
                                        dtype=torch.float32)
        pred_grid = model.ensemble_inference(flat_plt_grid).reshape(
            *plt_grid_shape, -1).numpy()[..., 0]
        plt.contourf(plt_grid[..., 0], plt_grid[..., 1], pred_grid,
                     levels=100, cmap="seismic_r")
        # plt.colorbar()
        plt.scatter(train_x[train_y == 0, 0], train_x[train_y == 0, 1],
                    c="royalblue", s=5)
        plt.scatter(train_x[train_y == 1, 0], train_x[train_y == 1, 1],
                    c="orange", s=5)
        plt.title(model_name)

        # Logging and updating progress bar
        p_bar.set_postfix({"Plotting": model_name})

    plt.show()


def set_random_seed_globally(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_ace(bin_count, errors, confidences):
    # Get adaptive bins
    _, indices = torch.sort(confidences)
    bins = [b.tolist() for b in torch.chunk(indices, bin_count)]

    # Get bin accuracies and confidences
    bin_accuracies = np.array(
        [errors[b].sum() / len(b) if len(b) > 0 else 0 for b in bins])
    bin_confidences = np.array(
        [confidences[b].sum() / len(b) if len(b) > 0 else 0 for b in bins])

    # Calculate ACE
    ace = 0
    for i in range(len(bins)):
        ace += len(bins[i]) * np.abs(bin_accuracies[i] - bin_confidences[i])
    ace /= len(confidences)

    return ace, bin_confidences, bin_accuracies


def plot_reliability(bin_confidence, bin_accuracy, model_name, axis):
    axis.plot(np.arange(0.5, 1.1, 0.1), np.arange(0.5, 1.1, 0.1),
              label="Perfect Calibration")
    axis.plot(bin_confidence, bin_accuracy, label="Actual Calibration")
    axis.grid(alpha=0.5)
    axis.set_xlim(0.5, 1)
    axis.set_ylim(0.5, 1)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(model_name)
    axis.legend()


def evaluate_models(model_dict, test_data_x, test_data_y):
    # fig, axes = plt.subplots((len(model_dict) + 1) // 2, 2, dpi=150,
    #                          squeeze=False,
    #                          figsize=(8, 3 * len(model_dict)))


    for i, (model_name, model) in enumerate(model_dict.items()):
        pred_y = model.ensemble_inference(
            torch.as_tensor(test_data_x, dtype=torch.float32))
        test_correctness = np.equal(pred_y.argmax(dim=1).numpy(), test_data_y)
        test_accuracy = test_correctness.sum() / test_data_x.shape[0]
        time.sleep(0.3)
        ace, bin_confidences, bin_accuracies = \
            calculate_ace(10, test_correctness, pred_y.max(dim=-1)[0])
        # plot_reliability(bin_confidences, bin_accuracies,
        #                  model_name, axes[i // 2, i % 2])
        print(f"\n{model_name}, Test Accuracy: {test_accuracy}, ACE: {ace}")
        time.sleep(0.3)
    # plt.show()
