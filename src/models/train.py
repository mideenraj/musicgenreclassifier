import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_maker import GtzanDataset

from vgg_net import VGG_Net


import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import math


# TODO: Implement way to visualize training stage
# TODO: Implement metrics
# TODO: Implement a way to dynamically create initial weights for model
# TODO: Implement Cross Validation
# TODO: Implement EARLY STOPPING


# Get Time Stamp
timestamp = datetime.datetime.now().strftime("%m, %d, %Y-%H:%M")

# Get the system information
sysinfo = torch.cuda.get_device_properties(
    0) if torch.cuda.is_available() else "CPU"


# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
# number of data samples propagated through the network before parameters are updated
BATCH_SIZE = 32
EPOCHS = 100  # Number of times to iterate over the dataset
# How much to update the model parameters at each batch/epoch.
# NOTE: Smaller learning rate means slow learning speed, but more stable
LEARNING_RATE = 0.0001

# Data Spliting COnfiguration
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1
TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE

print("Creating model...")
# Construct the model
desired_arch = "VGG13"
MODEL = VGG_Net(architecture=desired_arch).to(DEVICE)
print("Model created")
print(MODEL)
print("-------------------")

# Create a tensorboard writer
writer = SummaryWriter("runs/" + desired_arch +
                       "_" + timestamp + "_" + sysinfo.name)

# Will help with underflowing gradients
# float16 is used to reduce memory usage
# but often doesn't take into account extremely small variations
# We need to scale our gradients so they don't get flushed to zero
SCALER = torch.cuda.amp.GradScaler()


def kaiming_init():
    """
    Initialize the weights of the model with a constant value

    Args:
        fill (float, optional): _description_. Defaults to 0.0.
    """
    for name, param in MODEL.named_parameters():
        print(f"Name: {name}")
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith("layers.0"):
            param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))


def overfit_batch(data_loader, loss_fn, optimizer):
    """
    Overfit a single batch of data

    Args:
        data_loader (DataLoader): The data loader to use
        loss_fn (nn.Module): The loss function to use
        optimizer (torch.optim): The optimizer to use
    """
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    for i in range(100):
        # Forward pass
        y_pred = MODEL(images)
        loss = loss_fn(y_pred, labels)

        # Backward pass
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

        # Zero out the gradients
        optimizer.zero_grad()

        # Report
        print(f"L: {loss.item():.4f}")
        if i % BATCH_SIZE == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
            writer.add_scalar("Loss/Train", loss.item(), i)
            writer.flush()


def train_one_epoch(data_loader, loss_fn, optimizer):
    """
    Train the model for one epoch

    Args:
        data_loader (DataLoader): The data loader to use for training
        loss_fn (nn.Module): The loss function to use
        optimizer (torch.optim): The optimizer to use
    """
    running_loss = 0.0

    for batch, (X, y) in enumerate(data_loader):
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # Forward pass
        y_pred = MODEL(X).to(DEVICE)
        loss = loss_fn(y_pred, y)

        # Backward pass
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()

        # Zero out the gradients
        optimizer.zero_grad()

        # Report
        print(f"Batch {batch+1} loss: {loss.item():.4f}")
        running_loss += loss.item()

    return running_loss


def train(train_loader, val_loader, loss_fn, optimizer):

    for i in range(EPOCHS):
        t0 = time.perf_counter()
        print(f"Epoch {i+1}")
        MODEL.train(True)
        running_loss = train_one_epoch(train_loader, loss_fn, optimizer)

        # We do not need gradients on to do reporting
        t1 = time.perf_counter()
        avg_loss = running_loss / len(train_loader)

        writer.add_scalar("Loss/Train", avg_loss, i + 1)
        test(val_loader, loss_fn, i)
        print(f"Epoch {i+1} took {t1-t0:.2f} seconds")
        print(" ------------------- ")
    writer.flush()

    print("Training finished")


@torch.no_grad()
def test(data_loader, loss_fn, epoch):
    """
    Test the model

    Args:
        data_loader (DataLoader): The data loader to use for testing
        loss_fn (_type_): _description_
    """
    # We do not need gradients on to do reporting
    MODEL.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        # Every data instance is a input and a label
        inputs, labels = data

        # Move the data to the device
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Make a predictions for this batch
        outputs = MODEL(inputs)

        # Calculate the loss and backpropagate
        loss = loss_fn(outputs, labels)

        # Gather data for reporting
        running_loss += loss.item()  # Use .item() to get the value of the tensor

        _, predicted = torch.max(outputs.data, 1)
        # Show Confidence for each class
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        # print(f"Confidence: {confidence.tolist()}")

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / (i + 1)
    accuracy = 100.0*correct / total

    print(f"Test Loss: {avg_loss}")
    print(f"Accuracy: {accuracy}")
    writer.add_scalar("Accuracy/Test", accuracy, epoch)

    # Add the data to tensorboard
    MODEL.train(True)
    writer.flush()


def get_batch_create_img_grid(data_loader):
    """
    Extract one batch of data from the data loader

    Args:
        data_loader (DataLoader): The data loader to extract from
    """
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # Get image size with torchvision with first image
    img_size = torchvision.transforms.functional.get_image_size(images[0])
    print(f"Image size: {img_size}")

    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image("Random Mel Spectrograms", img_grid)
    writer.flush()


def visualize_model(data_loader):
    """
    Use Tensorboard to examine the data flow within the model.

    Args:
        model (nn.Module): The model to visualize
        data_loader (DataLoader): The data loader to use for visualization
    """
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    writer.add_graph(MODEL, images)
    writer.flush()


if __name__ == "__main__":

    ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
    GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"

    ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
    GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

    # Free GPU cache
    # print(torch.cuda.memory_summary())
    # For good practice setup the random seed (Reproducibility)
    torch.manual_seed(0)

    # Load the data
    print("Loading data...")
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_CLOUD,
                         genres_dir=GENRES_DIR_CLOUD,)
    print("Data loaded")
    print("Size of dataset: ", len(gtzan))
    print("-------------------")

    print("Splitting data into train, validation, and test sets")
    # Split the data into training, testing, and validation sets
    train_count = int(len(gtzan) * TRAINING_PERCENTAGE)  # 80% of data
    val_count = int(len(gtzan) * VALIDATION_PERCENTAGE)  # 10% of the data
    test_count = len(gtzan) - train_count - val_count  # 10% of the data
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        gtzan, [train_count, val_count, test_count])
    print("Data split")
    print("-------------------")

    print("Creating data loaders...")
    # Create a dataloader for the training, testing, and validation sets
    training_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_data_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Data loaders created")
    print("-------------------")

    print("SANITY CHECK")
    print("Reporting to Tensorboard Random Mel Spectrogram Images from Training Set")
    # Display a random sample from the dataset
    get_batch_create_img_grid(training_data_loader)
    print("-------------------")

    # Instantiate Loss function and Optimizer
    # CrossEntropyLoss is used for classification
    loss_fn = nn.CrossEntropyLoss()

    # Adam is a popular optimizer for deep learning
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    # Visualize the model
    # visualize_model(training_data_loader)

    # Initialize the model with kaiming initialization
    # kaiming_init()
    # Print model's state_dict

    # (SANITY CHECK) OVERFIT ONE BATCH
    # overfit_batch(training_data_loader, loss_fn, optimizer)

    print("Training model...")
    # Train the model
    t_start = time.perf_counter()

    # train_it_baby(training_data_loader, test_data_loader, loss_fn, optimizer)
    train(training_data_loader, val_data_loader, loss_fn, optimizer)

    t_end = time.perf_counter()
    print(f"Training took {t_end - t_start} seconds")
    print("Model trained")
    print("-------------------")

    # Test the model
    print("Testing model...")
    t_start = time.perf_counter()
    # test(test_data_loader, loss_fn)
    print("Model tested")
    print("-------------------")

    # Save the model
    model_name = MODEL.get_model_name()

    # Get root directory
    path = os.path.join(os.getcwd(), "saved_models")
    print(f"path: {path}")

    # Get path to MGR/src/results directory
    model_filename = f"{model_name}_{timestamp}_{sysinfo.name}.pth"
    model_path = os.path.join(path, model_filename)

    torch.save(MODEL.state_dict(), model_path)
    print("Saved best model")
    print(f"Model saved to {model_path}")
    print("-------------------")
