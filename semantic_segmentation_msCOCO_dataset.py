

#imports
import os
import torch
import torchvision
import copy
import time
import random
import numpy as np
import requests
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import cv2
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms





#Dataset generation code same as Homework 6 code with slight modifications for mask area contraints
class DatasetGenerator():
    def __init__(
        self, root_dir, annotation_path, classes, min_area=40000  # 200x200 = 40000 pixels
    ):
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.classes = classes
        self.min_area = min_area

        self.coco = COCO(annotation_path)
        self.catIds = self.coco.getCatIds(catNms=classes)
        self.categories = self.coco.loadCats(self.catIds)
        self.categories.sort(key=lambda x: x["id"])
        self.class_dir = {}

        self.coco_labels_inverse = {
            c["id"]: idx for idx, c in enumerate(self.categories)
        }

    def create_dir(self):
        for c in self.classes:
            dir_path = os.path.join(self.root_dir, c)
            self.class_dir[c] = dir_path
            os.makedirs(dir_path, exist_ok=True)

    def download_images(self, download=True, val=False):
        img_paths = {c: [] for c in self.classes}
        img_masks = {c: [] for c in self.classes}

        for c in tqdm(self.classes):
            class_id = self.coco.getCatIds(c)
            img_ids = self.coco.getImgIds(catIds=class_id)
            images = self.coco.loadImgs(img_ids)

            for image in images:
                annIds = self.coco.getAnnIds(
                    imgIds=image["id"], catIds=class_id, iscrowd=False
                )
                annotations = self.coco.loadAnns(annIds)

                valid_annotations = [
                    ann
                    for ann in annotations
                    if ann["area"] >= self.min_area
                ]

                if len(valid_annotations) == 1:
                    ann = valid_annotations[0]
                    mask = self.coco.annToMask(ann)
                    mask_path = os.path.join(
                        self.root_dir, c, image["file_name"].replace(".jpg", "_mask.png")
                    )
                    # Convert mask array to binary mask with 0 for background and 1 for mask
                    binary_mask = (mask > 0).astype(np.uint8) * 255

                    # Create PIL image from the binary mask and save it
                    Image.fromarray(binary_mask, mode='L').save(mask_path)

                    img_path = os.path.join(self.root_dir, c, image["file_name"])
                    if download:
                        if self.download_image(img_path, image["coco_url"]):
                            self.convert_image(img_path)
                            img_paths[c].append(img_path)
                            img_masks[c].append(mask_path)
                    else:
                        img_paths[c].append(img_path)
                        img_masks[c].append(mask_path)

        return img_paths, img_masks

    # Download image from URL using requests
    def download_image(self, path, url):
        try:
            img_data = requests.get(url).content
            with open(path, "wb") as f:
                f.write(img_data)
            return True
        except Exception as e:
            print(f"Caught exception: {e}")
        return False

    # Resize image
    def convert_image(self, path):
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert(mode="RGB")
            im = im.resize((256,256)) # resize the image to 256x256 size before downloading
        im.save(path)





classes = ['cake', 'dog', 'motorcycle']

# Download training images
train_downloader = DatasetGenerator('/Users/skose/Downloads/CocoDatasetTrain',
            '/Users/skose/Downloads/CocoDataset/annotations/instances_train2017.json',
            classes)

train_downloader.create_dir()
train_img_paths, train_img_masks = train_downloader.download_images(download=True)





# Download validation images
test_downloader = DatasetGenerator('/Users/skose/Downloads/CocoDatasetTest',
            '/Users/skose/Downloads/CocoDataset/annotations/instances_val2017.json',
            classes)

test_downloader.create_dir()
test_img_paths, test_img_masks = test_downloader.download_images(download=True, val=True)




#Custom Dataset Class
class CocoDatasetSubset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'mask')
        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.split('.')[0] + '_mask.png')

        # Open image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        sample = {'image' : image,
                    'mask'  : mask}

        return sample





#mUnet architecture class inherited from Dr. Kak's DL studio module mUnet class code
class mUnet(nn.Module):
   
    def __init__(self, skip_connections=True, depth=16):
        super().__init__()
        self.depth = depth // 2
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        ##  For the DN arm of the U:
        self.bn1DN  = nn.BatchNorm2d(64)
        self.bn2DN  = nn.BatchNorm2d(128)
        self.skip64DN_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip64DN_arr.append(SkipBlockDN(64, 64, skip_connections=skip_connections))
        self.skip64dsDN = SkipBlockDN(64, 64,   downsample=True, skip_connections=skip_connections)
        self.skip64to128DN = SkipBlockDN(64, 128, skip_connections=skip_connections )
        self.skip128DN_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip128DN_arr.append(SkipBlockDN(128, 128, skip_connections=skip_connections))
        self.skip128dsDN = SkipBlockDN(128,128, downsample=True, skip_connections=skip_connections)
        ##  For the UP arm of the U:
        self.bn1UP  = nn.BatchNorm2d(128)
        self.bn2UP  = nn.BatchNorm2d(64)
        self.skip64UP_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip64UP_arr.append(SkipBlockUP(64, 64, skip_connections=skip_connections))
        self.skip64usUP = SkipBlockUP(64, 64, upsample=True, skip_connections=skip_connections)
        self.skip128to64UP = SkipBlockUP(128, 64, skip_connections=skip_connections )
        self.skip128UP_arr = nn.ModuleList()
        for i in range(self.depth):
            self.skip128UP_arr.append(SkipBlockUP(128, 128, skip_connections=skip_connections))
        self.skip128usUP = SkipBlockUP(128,128, upsample=True, skip_connections=skip_connections)
        self.conv_out = nn.ConvTranspose2d(64, 1, 3, stride=2,dilation=2,output_padding=1,padding=2)

    def forward(self, x):
        ##  Going down to the bottom of the U:
        x = nn.MaxPool2d(2,2)(nn.functional.relu(self.conv_in(x)))          
        for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
            x = skip64(x)                

        num_channels_to_save1 = x.shape[1] // 2
        save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
        x = self.skip64dsDN(x)
        for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
            x = skip64(x)                
        x = self.bn1DN(x)
        num_channels_to_save2 = x.shape[1] // 2
        save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
        x = self.skip64to128DN(x)
        for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
            x = skip128(x)                

        x = self.bn2DN(x)
        num_channels_to_save3 = x.shape[1] // 2
        save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
        for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
            x = skip128(x)                
        x = self.skip128dsDN(x)
        ## Coming up from the bottom of U on the other side:
        x = self.skip128usUP(x)          
        for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
            x = skip128(x)                
        x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
        x = self.bn1UP(x)
        for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
            x = skip128(x)                
        x = self.skip128to64UP(x)
        for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
            x = skip64(x)                
        x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
        x = self.bn2UP(x)
        x = self.skip64usUP(x)
        for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
            x = skip64(x)                
        x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
        x = self.conv_out(x)
        return x
    
    
    
class SkipBlockDN(nn.Module):
   
    def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
        super().__init__()
        self.downsample = downsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if downsample:
            self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
    def forward(self, x):
        identity = x                                     
        out = self.convo1(x)                              
        out = self.bn1(out)                              
        out = nn.functional.relu(out)
        if self.in_ch == self.out_ch:
            out = self.convo2(out)                              
            out = self.bn2(out)                              
            out = nn.functional.relu(out)
        if self.downsample:
            out = self.downsampler(out)
            identity = self.downsampler(identity)
        if self.skip_connections:
            if self.in_ch == self.out_ch:
                out = out + identity
            else:
                out = out + torch.cat((identity, identity), dim=1) 
        return out


class SkipBlockUP(nn.Module):

    def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
        super().__init__()
        self.upsample = upsample
        self.skip_connections = skip_connections
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
        self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if upsample:
            self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
    def forward(self, x):
        identity = x                                     
        out = self.convoT1(x)                              
        out = self.bn1(out)                              
        out = nn.functional.relu(out)
        out  =  nn.ReLU(inplace=False)(out)            
        if self.in_ch == self.out_ch:
            out = self.convoT2(out)                              
            out = self.bn2(out)                              
            out = nn.functional.relu(out)
        if self.upsample:
            out = self.upsampler(out)
            identity = self.upsampler(identity)
        if self.skip_connections:
            if self.in_ch == self.out_ch:
                out = out + identity                              
            else:
                out = out + identity[:,self.out_ch:,:,:]
        return out




# Define transformations
transform = transforms.Compose([
  transforms.Resize((256, 256)),
 transforms.ToTensor()
])

# Create dataset instance
train_dataset = CocoDatasetSubset(root_dir='/home/skose/CocoTrainDataset2', transform=transform)

test_dataset = CocoDatasetSubset(root_dir='/home/skose/CocoDatasetVal2', transform=transform)
#Create Dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)





model = mUnet(skip_connections=True, depth=16)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n\nThe number of learnable parameters in the model: %d\n" % number_of_learnable_params)

num_layers = len(list(model.parameters()))
print("\nThe number of layers in the model: %d\n\n" % num_layers)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device





#Training loop similar to Dr. Kak's DL studio code
def run_code_for_training_for_semantic_segmentation_MSE(net, data_loader, device, epochs=1):        
    filename_for_out1 = "performance_numbers_" + str(epochs) + ".txt"
    FILE1 = open(filename_for_out1, 'w')
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion1 = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), 
                 lr=1e-4, momentum=0.9)
    start_time = time.perf_counter()
    loss_values = []
    for epoch in range(epochs):  
        print("")
        running_loss_segmentation = 0.0
        for i, data in enumerate(data_loader):    
            im_tensor,mask_tensor =data['image'],data['mask']
            im_tensor   = im_tensor.to(device)
            mask_tensor = mask_tensor.type(torch.FloatTensor)
            mask_tensor = mask_tensor.to(device)      
            
            optimizer.zero_grad()
            output = net(im_tensor) 
            
            segmentation_loss = criterion1(output, mask_tensor)  
            segmentation_loss.backward()
            optimizer.step()
            running_loss_segmentation += segmentation_loss.item()
           
            if i%100==99:    
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                avg_loss_segmentation = (running_loss_segmentation / float(100))
                print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   MSE loss: %.3f" % (epoch+1,epochs, i+1, elapsed_time, avg_loss_segmentation))
                FILE1.write("%.3f\n" % avg_loss_segmentation)
                FILE1.flush()
                running_loss_segmentation = 0.0
                loss_values.append(avg_loss_segmentation)
                
    FILE1.close()
    print("\nFinished Training\n")
    torch.save(net.state_dict(), "./saveModel0")
    return loss_values





mse_loss = run_code_for_training_for_semantic_segmentation_MSE(net=model,data_loader=train_dataloader, 
                                                           device=device,epochs= 6)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

#Testing loop similar to Dr. Kak's DL studio code
def run_code_for_testing_semantic_segmentation_MSE(net, test_dataloader):
    # Load the trained model
    net.load_state_dict(torch.load("/home/skose/saveModel0"))
    
    # Define batch size
    batch_size = 4
    
    # Loop over the entire dataloader
    for j, data in enumerate(test_dataloader):
        # Get images and masks from the data
        images, masks = data['image'], data['mask']
        
        # Perform inference using the model
        outputs = net(images)
        
        # Print output every 10 iterations
        if j % 10 == 0:
            print("\n\n\n\nShowing output new for test batch %d: " % (j+1))
        
        # Plot images, masks, and network outputs for each sample in the batch
        for i in range(batch_size):
            image_np = images[i].permute(1, 2, 0).numpy()
            mask_np = masks[i].squeeze().numpy()
            output_np = outputs[i].permute(1, 2, 0).detach().numpy().squeeze()

            plt.figure(figsize=(15, 5))

            # Plot the original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')

            # Plot the ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Plot the network output
            plt.subplot(1, 3, 3)
            # Normalize the output_np to the range [0, 1]
            output_np_normalized = (output_np - np.min(output_np)) / (np.max(output_np) - np.min(output_np))
            # Threshold value
            threshold = 0.5  
            # Apply thresholding
            output_mask_bw = np.where(output_np_normalized >= threshold, 1.0, 0.0)

            plt.imshow(output_mask_bw, cmap='gray')  
            plt.title('Model Output')
            plt.axis('off')

            plt.show()

# Call the function with your model and test dataloader
run_code_for_testing_semantic_segmentation_MSE(model, test_dataloader)



plt.plot(mse_loss, label='MSE Loss')
plt.title('MSE Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()





#For Dice Loss
def run_code_for_training_for_semantic_segmentation_dice(net, data_loader, device, epochs=1):        
    filename_for_out1 = "performance_numbers_" + str(epochs) + ".txt"
    FILE1 = open(filename_for_out1, 'w')
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion1 = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), 
                 lr=1e-4, momentum=0.9)
    start_time = time.perf_counter()
    loss_values = []
    for epoch in range(epochs):  
        print("")
        running_loss_segmentation = 0.0
        for i, data in enumerate(data_loader):    
            im_tensor,mask_tensor =data['image'],data['mask']
            im_tensor   = im_tensor.to(device)
            mask_tensor = mask_tensor.type(torch.FloatTensor)
            mask_tensor = mask_tensor.to(device)                 
            optimizer.zero_grad()
            output = net(im_tensor) 
            
            numerator = torch.sum(output * mask_tensor)
            denominator = torch.sum(output * output) + torch.sum(mask_tensor * mask_tensor)
            dice_coefficient = 2 * numerator / (denominator + (1e-6))
            segmentation_loss_dice = 1 - dice_coefficient

            running_loss_segmentation += segmentation_loss_dice.item()
            segmentation_loss_dice.backward()
            optimizer.step()
           
            if i%100==99:    
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                avg_loss_segmentation = (running_loss_segmentation / float(100))
                print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   Dice loss: %.3f" % (epoch+1,epochs, i+1, elapsed_time, avg_loss_segmentation))
                FILE1.write("%.3f\n" % avg_loss_segmentation)
                FILE1.flush()
                running_loss_segmentation = 0.0
                loss_values.append(avg_loss_segmentation)
                
    FILE1.close()
    print("\nFinished Training\n")
    torch.save(net.state_dict(), "./saveModel1")
    return loss_values





dice_loss = run_code_for_training_for_semantic_segmentation_dice(net=model,data_loader=train_dataloader, 
                                                           device=device,epochs= 6)





def run_code_for_testing_semantic_segmentation_dice(net, test_dataloader):
    # Load the trained model
    net.load_state_dict(torch.load("/home/skose/saveModel1"))
    
    # Define batch size
    batch_size = 4
    
    # Loop over the entire dataloader
    for j, data in enumerate(test_dataloader):
        # Get images and masks from the data
        images, masks = data['image'], data['mask']
        
        # Perform inference using the model
        outputs = net(images)
        
        # Print output every 10 iterations
        if j % 10 == 0:
            print("\n\n\n\nShowing output new for test batch %d: " % (j+1))
        
        # Plot images, masks, and network outputs for each sample in the batch
        for i in range(batch_size):
            image_np = images[i].permute(1, 2, 0).numpy()
            mask_np = masks[i].squeeze().numpy()
            output_np = outputs[i].permute(1, 2, 0).detach().numpy().squeeze()

            plt.figure(figsize=(15, 5))

            # Plot the original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')

            # Plot the ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Plot the network output
            plt.subplot(1, 3, 3)
            # Normalize the output_np to the range [0, 1]
            output_np_normalized = (output_np - np.min(output_np)) / (np.max(output_np) - np.min(output_np))
            # Threshold value
            threshold = 0.5  
            # Apply thresholding
            output_mask_bw = np.where(output_np_normalized >= threshold, 1.0, 0.0)

            plt.imshow(output_mask_bw, cmap='gray')  
            plt.title('Model Output')
            plt.axis('off')

            plt.show()

# Call the function with your model and test dataloader
run_code_for_testing_semantic_segmentation_dice(model, test_dataloader)





plt.plot(dice_loss, label='Dice Loss')
plt.title('Dice Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()





#For Combined loss
def run_code_for_training_for_semantic_segmentation_combined(net, data_loader, device, epochs=1):        
    filename_for_out1 = "performance_numbers_" + str(epochs) + ".txt"
    FILE1 = open(filename_for_out1, 'w')
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion1 = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), 
                 lr=1e-4, momentum=0.9)
    start_time = time.perf_counter()
    loss_values = []
    for epoch in range(epochs):  
        print("")
        running_loss_segmentation_mse = 0.0
        running_loss_segmentation_dice = 0.0
        running_loss_segmentation_combined = 0.0
        for i, data in enumerate(data_loader):    
            im_tensor,mask_tensor =data['image'],data['mask']
            im_tensor   = im_tensor.to(device)
            mask_tensor = mask_tensor.type(torch.FloatTensor)
            mask_tensor = mask_tensor.to(device)                 
            optimizer.zero_grad()
            output = net(im_tensor) 
            
            # Calculate MSE loss

            segmentation_loss_mse = criterion1(output, mask_tensor)  
            running_loss_segmentation_mse += segmentation_loss_mse.item()

            # Calculate Dice loss
            numerator = torch.sum(output * mask_tensor)
            denominator = torch.sum(output * output) + torch.sum(mask_tensor * mask_tensor)
            dice_coefficient = 2 * numerator / (denominator + (1e-6))
            segmentation_loss_dice = 1 - dice_coefficient
            running_loss_segmentation_dice += segmentation_loss_dice.item()

            # Combine MSE and Dice losses with weights
            combined_loss =  segmentation_loss_mse  + 20 * segmentation_loss_dice

            #combined_loss.backward()
            combined_loss.backward()
            optimizer.step()

            running_loss_segmentation_combined += combined_loss.item()

            if i%100==99:    
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                avg_loss_segmentation = running_loss_segmentation_combined / float(100)
                print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   Combined loss: %.3f" % (epoch+1, epochs, i+1, elapsed_time, avg_loss_segmentation))
                FILE1.write("%.3f\n" % avg_loss_segmentation)
                FILE1.flush()

                running_loss_segmentation_combined = 0.0   
                running_loss_segmentation_mse = 0.0
                running_loss_segmentation_dice = 0.0

                loss_values.append(avg_loss_segmentation)
                    
                
    FILE1.close()
    print("\nFinished Training\n")
    torch.save(net.state_dict(), "./saveModel2")
    return loss_values
 





combined_loss = run_code_for_training_for_semantic_segmentation_combined(net=model,data_loader=train_dataloader, 
                                                           device=device,epochs= 6)





plt.plot(combined_loss, label='Combined Loss')
plt.title('Combined Loss vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()



def run_code_for_testing_semantic_segmentation(net, test_dataloader):
    # Load the trained model
    net.load_state_dict(torch.load("/home/skose/saveModel2"))
    
    # Define batch size
    batch_size = 4
    
    # Loop over the entire dataloader
    for j, data in enumerate(test_dataloader):
        # Get images and masks from the data
        images, masks = data['image'], data['mask']
        
        # Perform inference using the model
        outputs = net(images)
        
        # Print output every 10 iterations
        if j % 10 == 0:
            print("\n\n\n\nShowing output new for test batch %d: " % (j+1))
        
        # Plot images, masks, and network outputs for each sample in the batch
        for i in range(batch_size):
            image_np = images[i].permute(1, 2, 0).numpy()
            mask_np = masks[i].squeeze().numpy()
            output_np = outputs[i].permute(1, 2, 0).detach().numpy().squeeze()

            plt.figure(figsize=(15, 5))

            # Plot the original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')

            # Plot the ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            # Plot the network output
            plt.subplot(1, 3, 3)
            # Normalize the output_np to the range [0, 1]
            output_np_normalized = (output_np - np.min(output_np)) / (np.max(output_np) - np.min(output_np))
            # Threshold value
            threshold = 0.5  
            # Apply thresholding
            output_mask_bw = np.where(output_np_normalized >= threshold, 1.0, 0.0)

            plt.imshow(output_mask_bw, cmap='gray')  
            plt.title('Model Output')
            plt.axis('off')

            plt.show()

# Call the function with your model and test dataloader
run_code_for_testing_semantic_segmentation(model, test_dataloader)




plt.plot(combined_loss, label='Combined Loss')
plt.plot(mse_loss, label='MSE Loss')
plt.plot(dice_loss, label='Dice Loss')
plt.title('Combined Loss vs Iteration')
plt.xlabel('Iteration')
plt.legend()
plt.ylabel('Loss')
plt.grid(True)
plt.show()







