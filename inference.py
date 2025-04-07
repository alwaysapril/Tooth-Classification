import torch
import torch.nn as nn
import timm_3d
from icecream.icecream import ic
from data_preparation import stl_to_voxel
import os


class STL3DModelInference:
    def __init__(self, checkpoint_path, class_names_path, model_name='tf_efficientnet_b0.in1k', voxel_dim=256):
        """
        Initializes the inference pipeline for the 3D STL model.
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint.
            class_names_path (str): Path to the file containing class names.
            model_name (str): Name of the model to be loaded from timm_3d.
            voxel_dim (int): Dimension for voxel grid processing.
        """
        self.checkpoint_path = checkpoint_path
        self.class_names = torch.load(class_names_path)
        self.num_classes = len(self.class_names)
        self.model_name = model_name
        self.voxel_dim = voxel_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads and initializes the trained model from a checkpoint.
        """
        model = timm_3d.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes,
            global_pool='avg'
        )

        # Modify the first convolutional layer to accept single-channel input
        model.conv_stem = nn.Conv3d(
            in_channels = 1,
            out_channels = model.conv_stem.out_channels,
            kernel_size = model.conv_stem.kernel_size,
            stride = model.conv_stem.stride,
            padding = model.conv_stem.padding,
            bias = model.conv_stem.bias
        )

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(self.device)
        model.eval()
        
        return model

    def preprocess_stl(self, file_path):
        """
        Converts an STL file into a voxel grid for inference.
        
        Args:
            file_path (str): Path to the STL file.
            
        Returns:
            torch.Tensor: Preprocessed voxel data.
        """
        voxel_data = stl_to_voxel(file_path, self.voxel_dim)
        voxel_data = torch.tensor(voxel_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return voxel_data.to(self.device)
    
    def make_possible_combinations(self,top5_classes,top5_probability):
        # Extract the main class,
        main_class = []
        sub_class= []
        main_class_name = ["Class I","Class II div I","Class II div II","Class III"]
        final_list = []

        # Show top-3 predicted classes and their probabilities,
        for i in range(5):
            print(f"Rank {i+1}: {top5_classes[i]} with confidence(probability) {top5_probability[i]}")

            if top5_probability[i] > 0:
                if top5_classes[i] not in main_class and top5_classes[i] in main_class_name:
                    main_class.append(top5_classes[i])
                elif top5_classes[i] not in sub_class and top5_classes[i] not in main_class_name:
                    sub_class.append(top5_classes[i])
        
        if len(main_class) == 0:
            print("Main class not found")
            return None
        else:
            if len(main_class) == 1:
                if len(sub_class) == 0:
                    return final_list.append(main_class[0])
                elif len(sub_class) == 1:
                    final_list.append([main_class[0],sub_class[0]])
                elif len(sub_class) == 2:
                    final_list.append([main_class[0],sub_class[0]])
                    final_list.append([main_class[0],sub_class[1]])
                elif len(sub_class) == 3:
                    final_list.append([main_class[0],sub_class[0]])
                    final_list.append([main_class[0],sub_class[1]])
                    final_list.append([main_class[0],sub_class[2]])
                else:
                    final_list.append([main_class[0],sub_class[0]])
                    final_list.append([main_class[0],sub_class[1]])
                    final_list.append([main_class[0],sub_class[2]])
                    final_list.append([main_class[0],sub_class[3]])
            elif len(main_class) == 2:
                if len(sub_class) == 0:
                    final_list.append(main_class[0])
                    final_list.append(main_class[1])
                elif len(sub_class) == 1:
                    final_list.append([main_class[0],sub_class[0]])
                    final_list.append([main_class[1],sub_class[0]])
                elif len(sub_class) == 2:
                    final_list.append([main_class[0],sub_class[0]])
                    final_list.append([main_class[0],sub_class[1]])
                    final_list.append([main_class[1],sub_class[0]])
                    final_list.append([main_class[1],sub_class[1]])
        
        return final_list


        
        
        

    def predict(self, file_path):
        """
        Predicts the class of an input STL file.
        
        Args:
            file_path (str): Path to the STL file.
            
        Returns:
            str: Predicted class name.
        """
        voxel_data = self.preprocess_stl(file_path)
        
        with torch.no_grad():
            output = self.model(voxel_data)
            _, predicted = torch.max(output, 1)
            predicted_class = self.class_names[predicted.item()]

            # Get the top 3 predictions and their indices
            top3_probs, top3_indices = torch.topk(output, 3, dim=1)  

            # Convert indices to class names
            top3_classes = [self.class_names[idx.item()] for idx in top3_indices[0]]

            # Convert probabilities to a list (optional, useful for debugging)
            top3_probs = top3_probs[0].tolist()

            # Get the top 3 predictions and their indices
            top5_probs, top5_indices = torch.topk(output, 5, dim=1)  
            # Convert indices to class names
            top5_classes = [self.class_names[idx.item()] for idx in top5_indices[0]]
            # Convert probabilities to a list (optional, useful for debugging)
            top5_probs = top5_probs[0].tolist()

            # Make possible combinations,
            final_list = self.make_possible_combinations(top5_classes,top5_probs)

        
        # return predicted_class, top3_classes, top3_probs
        return final_list



if __name__ == "__main__":
    # Define paths
    class_names_path = 'class_names.pth'
    model_name = 'best_model_checkpoint_efficientnet.pth'
    model_checkpoint_path = os.path.join("models",model_name)
    stl_file_path = input("Enter File path of STL File: ")

    # Initialize the inference pipeline
    inference_pipeline = STL3DModelInference(
        checkpoint_path=model_checkpoint_path,
        class_names_path=class_names_path
    )

    # Predict the class
    # predicted_class,top3_classes, top3_probability = inference_pipeline.predict(stl_file_path)
    predicted_classes = inference_pipeline.predict(stl_file_path)


    # Show top-3 predicted classes and their probabilities,
    for i in range(len(predicted_classes)):
        # print(f"Rank {i+1}: {top3_classes[i]} with confidence(probability) {top3_probability[i]}")
        print(f"Rank {i+1}: {predicted_classes[i]}")
