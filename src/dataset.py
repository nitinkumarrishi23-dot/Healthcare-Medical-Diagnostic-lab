from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
class medicalimagedataset(Dataset):   #finds image,loads the image, preprocesses them and returns labels.
    def __init__(self,root):#stores the image folder path so it can be used later
        self.folder_path=Path(root)
        self.folders=[
            folder
            for folder in self.folder_path.iterdir()
            if folder.is_dir() 
        ]       #find all class folders inside the dataset folder


    #to find folder name of each class folders
        self.classes=[
            folder.name
            for folder in self.folders

        ]
    #to assign index to each class and store it as dict on class_to_index
        self.class_to_index={
            class_name:index
            for index,class_name in enumerate(self.classes)
        }
        self.image_path=[] #store image paths
        self.image_label=[] #store image labels
        #both image_path and image_label refers to same image on same index , such as image_path[0]and image_label[0] refers to same image
        
        img_extensions={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} 
        for folder in self.folders:
            class_name=folder.name
            label=self.class_to_index[class_name]
            for img_path in folder.iterdir():
                if img_path.is_file() and img_path.suffix.lower()in img_extensions: #checks whether the img_path is a file and if its a file is it a img file
                    self.image_path.append(img_path)
                    self.image_label.append(label)
