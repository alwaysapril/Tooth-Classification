import os
import torch
from torch.utils.data import Dataset, DataLoader
from stl import mesh
import numpy as np
from tqdm import tqdm

def stl_to_voxel(file_path, voxel_dim=128):
    """
    Convert an STL file to a voxel grid.
    """
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # Calculate the bounding box dimensions and scale factor
    min_bound = np.min(stl_mesh.points.reshape(-1, 3), axis=0)
    max_bound = np.max(stl_mesh.points.reshape(-1, 3), axis=0)
    scale_factor = voxel_dim / (max_bound - min_bound).max()
    
    # Scale and center the mesh points
    scaled_points = (stl_mesh.points.reshape(-1, 3) - min_bound) * scale_factor
    scaled_points = scaled_points.astype(int)
    
    # Initialize the voxel grid and populate it
    voxel_grid = np.zeros((voxel_dim, voxel_dim, voxel_dim), dtype=bool)
    valid_points = (scaled_points < voxel_dim) & (scaled_points >= 0)
    valid_indices = valid_points.all(axis=1)
    voxel_grid[scaled_points[valid_indices, 0],
               scaled_points[valid_indices, 1],
               scaled_points[valid_indices, 2]] = True
    
    return voxel_grid.astype(float)

class STL3DDataset(Dataset):
    def __init__(self, root_dir, voxel_dim=128, transform=None):
        self.root_dir = root_dir
        self.voxel_dim = voxel_dim
        self.transform = transform
        self.file_paths = []
        self.labels = []
        
        # Traverse directories to gather file paths and labels
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in tqdm(os.listdir(class_dir),desc="preparing dataset for training......."):
                    if file_name.endswith('.stl'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(label)
                        
        self.class_names = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and convert the STL file to a voxel grid
        voxel_data = stl_to_voxel(file_path, self.voxel_dim)
        
        # Apply any additional transformations
        if self.transform:
            voxel_data = self.transform(voxel_data)
        
        # Convert voxel_data to a PyTorch tensor and expand dimensions to match model input
        voxel_data = torch.tensor(voxel_data, dtype=torch.float32).unsqueeze(0)
        
        return voxel_data, label

if __name__ == "__main__":
    # Set the dataset root directory and voxel dimensions
    root_dir = '/home/lenovo/Downloads/teeth/teeth_new_data/train_new_data/'  # Replace with the path to your dataset folder
    voxel_dim = 256  # Adjust as needed

    # Create dataset and dataloader
    dataset = STL3DDataset(root_dir=root_dir, voxel_dim=voxel_dim)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Save class names and print dataset information
    torch.save(dataset.class_names, 'class_names.pth')
    print(f"Number of classes: {len(dataset.class_names)}")
    print(f"Classes: {dataset.class_names}")
    print(f"Number of samples: {len(dataset)}")
