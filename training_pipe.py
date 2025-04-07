import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm_3d
import wandb
from tqdm import tqdm
from data_preparation_tim import STL3DDataset

                    
                    
class TrainingPipeline:
    def __init__(self, root_dir, voxel_dim, batch_size, num_epochs, learning_rate, model_type, checkpoint_path,final_model_path):
        self.root_dir = root_dir
        self.voxel_dim = voxel_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.final_model_path = final_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.class_names = torch.load('class_names.pth')
        self.num_classes = len(self.class_names)
        self.dataset = STL3DDataset(root_dir=self.root_dir, voxel_dim=self.voxel_dim)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0
        
        # Initialize W&B
        wandb.init(
            project="3D-Teeth-Model-Training",
            config={
                "root_dir": root_dir,
                "voxel_dim": voxel_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "model_type": model_type,
                "Number of classes": self.num_classes,
                "Name of classes": self.class_names
            }
        )
        wandb.watch(self.model, log="all", log_freq=10)

    def _initialize_model(self):
        """
        Initialize the model, modify the input layer to handle single-channel input.
        """
        model = timm_3d.create_model(
            self.model_type,
            pretrained=True,
            num_classes=self.num_classes,
            global_pool='avg'
        )
        # Modify the first convolution layer to accept a single channel
        if 'efficientnet' in self.model_type:
            model.conv_stem = nn.Conv3d(
                in_channels=1,
                out_channels=model.conv_stem.out_channels,
                kernel_size=model.conv_stem.kernel_size,
                stride=model.conv_stem.stride,
                padding=model.conv_stem.padding,
                bias=model.conv_stem.bias
            )
        elif 'convnext' in self.model_type and hasattr(model, 'stem') and isinstance(model.stem[0], nn.Conv3d):
            original_bias = model.stem[0].bias is not None
            model.stem[0] = nn.Conv3d(
                in_channels=1,
                out_channels=model.stem[0].out_channels,
                kernel_size=model.stem[0].kernel_size,
                stride=model.stem[0].stride,
                padding=model.stem[0].padding,
                bias=original_bias
            )
        else:
            raise ValueError(f"Unsupported model type or architecture: {self.model_type}")

        model.to(self.device)
        return model

    def train_epoch(self, epoch):
        """
        Train the model for number of epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(self.dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{epoch + 1}/{self.num_epochs}]")

            for voxel_data, labels in tepoch:
                voxel_data, labels = voxel_data.to(self.device), labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(voxel_data)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Update metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log batch metrics to W&B
                wandb.log({
                    "Batch Loss": loss.item(),
                    "Batch Accuracy": 100 * correct / total
                })

                # Display batch metrics
                tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        epoch_loss = running_loss / len(self.dataloader)
        epoch_accuracy = 100 * correct / total

        return epoch_loss, epoch_accuracy

    def save_checkpoint(self, epoch, accuracy, loss):
        """
        Save a checkpoint of the model.
        """
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': loss,
        }, self.checkpoint_path)
        print(f"Checkpoint saved with accuracy: {accuracy:.2f}%")

    def train(self):
        """
        Train the model for the specified number of epochs.
        """
        for epoch in range(self.num_epochs):
            epoch_loss, epoch_accuracy = self.train_epoch(epoch)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

            # Log epoch metrics to W&B
            wandb.log({
                "Epoch": epoch + 1,
                "Epoch Loss": epoch_loss,
                "Epoch Accuracy": epoch_accuracy
            })
            
            # Save checkpoint if accuracy improves
            if epoch_accuracy > self.best_accuracy:
                self.best_accuracy = epoch_accuracy
                self.save_checkpoint(epoch, epoch_accuracy, epoch_loss)

        # Save the final trained model
        torch.save(self.model.state_dict(), self.final_model_path)
        wandb.log_artifact(f"{self.final_model_path}", type="model")
        print("Model training completed and saved.")

    def evaluate(self, dataloader):
        """
        Evaluate the model on a given DataLoader.
        """
        self.model.eval()
        correct = 0
        total = 0                                                                                                  

        with torch.no_grad():             
            for voxel_data, labels in dataloader:
                voxel_data, labels = voxel_data.to(self.device), labels.to(self.device)
                outputs = self.model(voxel_data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Evaluation Accuracy: {accuracy:.2f}%")
        
        # Log evaluation metrics to W&B
        wandb.log({
            "Evaluation Accuracy": accuracy
        })

        return accuracy


# Usage
if __name__ == "__main__":
    # Define parameters
    root_dir = '/root/font_recognition/Fardin/teeth3d/datasets/teeth_train_data/'
    voxel_dim = 128
    batch_size = 10
    num_epochs = 2
    learning_rate = 0.001
    model_type = 'tf_efficientnet_b0.in1k'  # or 'convnext_small'
    checkpoint_path = 'models/best_model_checkpoint_efficientnet.pth'
    final_model_path = "models/final_model_checkpoint_efficientnet_2.pth"

    # Initialize and run the pipeline
    pipeline = TrainingPipeline(
        root_dir=root_dir,
        voxel_dim=voxel_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        final_model_path=final_model_path
    )

    pipeline.train()
