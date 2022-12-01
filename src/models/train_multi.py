import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset_maker import GtzanDataset
from vgg_net import VGG_Net
import datetime

# Multi-GPU support
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Tensorboard Support
from torch.utils.tensorboard import SummaryWriter
import torch.profiler


ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"

ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

TIMESTAMP = datetime.datetime.now().strftime("%m-%d-%Y--%H:%M")


WRITER = SummaryWriter()


def ddp_setup():
    init_process_group(backend="nccl")


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            save_every: int,
            snapshot_path: str) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data,
        self.val_data = val_data,
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print(f"Loading snapshot from {snapshot_path}")
            self._load_snapshot(snapshot_path)
        # Wrap the model with DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot)["MODEL_STATE"]
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(
            f"Loaded snapshot from {snapshot_path} from epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshots_path = os.path.join(os.getcwd(), "snapshots", "snapshot.pth")
        torch.save(snapshot, snapshots_path)
        print(f"Saved snapshot for epoch {epoch}")

    def _save_checkpoint(self, epoch):
        # We need to access module since it was wrapped with DDP
        checkpoint = self.model.module.state_dict()
        # Save the model
        model_name = self.model.module.get_model_name()

        # Get path to MGR/src/results directory
        model_filename = f"{model_name}_{TIMESTAMP}_Epoch{epoch}.pth"
        model_path = os.path.join(
            os.getcwd(),
            "saved_models",
            model_filename)
        torch.save(checkpoint, model_path)
        print(f"Saved checkpoint for epoch {epoch}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        # Metrics
        running_loss = 0.0

        # Get  Batch Size from dataloader
        b_sz = self.train_data[0].batch_size
        print(
            f"[GPU {self.gpu_id}] Epoch {epoch} | Batch Size {b_sz} | Steps {len(self.train_data[0])}")
        # .sampler is the DistributedSampler for shuffling the data
        self.train_data[0].sampler.set_epoch(epoch)
        for source, targets in self.train_data[0]:
            loss = self._run_batch(
                source.to(self.gpu_id), targets.to(self.gpu_id))

            # Log loss to tensorboard
            if self.gpu_id == 0:
                WRITER.add_scalar("Loss/Train", loss, epoch)

        WRITER.flush()

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self.validate(epoch)

            # Save only from master process since all other nodes will have the same model
            if epoch % self.save_every == 0 and self.gpu_id == 0:
                self._save_checkpoint(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_correct = 0
        total_items = 0

        # Gradient computation is not needed
        with torch.no_grad():
            for source, targets in self.val_data[0]:
                # Send data to GPU
                source, targets = source.to(
                    self.gpu_id), targets.to(self.gpu_id)

                output = self.model(source)
                _, preds = torch.max(output, 1)
                total_correct += torch.sum(preds == targets)
                total_items += len(targets)

        print(
            f"Validation Accuracy: {100.0*(total_correct / total_items):.2f}%")
        print(f"Total Correct: {total_correct}")
        print(f"Total Items: {total_items}")
        WRITER.add_scalar("Accuracy/Validation", 100.0 *
                          (total_correct / total_items), epoch)
        self.model.train()

        return total_correct / total_items

    def write_images_to_tensorboard(self):
        for source, targets in self.val_data[0]:
            output = self.model(source.to(self.gpu_id))
            _, preds = torch.max(output, 1)
            for i in range(len(source)):
                WRITER.add_image(
                    "Validation Images",
                    source[i],
                    targets[i],
                    dataformats="CHW")
                WRITER.add_image(
                    "Validation Images",
                    source[i],
                    preds[i],
                    dataformats="CHW")
            break


# Load Train Objects
# The ingredients for training
def load_train_objs():
    VALIDATION_SPLIT = 0.1
    TRAIN_SPLIT = 0.7
    TEST_SPLIT = 1 - TRAIN_SPLIT - VALIDATION_SPLIT

    # Load YOUR dataset
    gtzan = GtzanDataset(ANNOTATIONS_FILE_CLOUD, GENRES_DIR_CLOUD)

    train_count = int(len(gtzan) * TRAIN_SPLIT)
    val_count = int(len(gtzan) * VALIDATION_SPLIT)
    test_count = len(gtzan) - train_count - val_count

    # Split into train and validation
    train_set, val_set, test_set = torch.utils.data.random_split(
        gtzan, [train_count, val_count, test_count])
    # load YOUR model
    # Types of VGG available: VGG11, VGG13, VGG16, VGG19
    model = VGG_Net(architecture="VGG16")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    return train_set, val_set, test_set, model, optimizer, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:

    # Create a DistributedSampler to handle distributing the dataset across nodes
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset),
    )


def main(total_epochs, save_every, snapshot_path: str = os.path.join(os.getcwd(), "snapshots", "snapshot.pth")):
    ddp_setup()
    train_set, val_set, test_set, model, optimizer, criterion = load_train_objs()
    train_data = prepare_dataloader(train_set, 45)
    val_data = prepare_dataloader(val_set, 45)
    test_data = prepare_dataloader(test_set, 45)

    trainer = Trainer(model, train_data, val_data, optimizer,
                      criterion, save_every, snapshot_path)

    trainer.train(total_epochs)
    # trainer.validate()
    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(total_epochs, save_every)
