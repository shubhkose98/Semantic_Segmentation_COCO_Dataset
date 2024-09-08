# Semantic Segmentation using Unet on MS COCO Dataset
Implementation of Semantic segmentation using a modified U-Net architecture - that can perform multi-object image segmentation - and Dice Loss for MS COCO dataset. Mean Squared Error (MSE) Loss, Dice Loss, and a combined Dice-MSE Loss are used in the evaluation of the model. Using PyTorch, the implementation is done in Python with the model architecture inspired from Professor Avinash Kak's [DLStudio](https://engineering.purdue.edu/kak/distDLS/#106) library.

## Architecture
The foundation of U-Net is an increasingly popular encoder-decoder architecture, in which the encoder path extracts object features and the decoder path returns those objects to the pixel space. The concept was initially presented as Fully Connected Networks, but U-Net builds on this architecture by enabling data to flow between the encoder and the decoder via a number of skip connections, which enhance the decoder's performance by maintaining fine-grained pixel relationships. Additionally, the decoder path makes use of transpose convolutions, which significantly increase decoding efficiency by transforming convolutions into matrix-vector products—a task that GPUs are specifically designed to perform. This project uses the modified unet architecture to perform semantic segmentation on the MS COCO dataset. The model is evaluated using three different loss functions:
1. Mean Squared Error (MSE) Loss
2. Dice Loss
3. Combined Dice-MSE Loss

## Dataset

The MS COCO dataset and [PurdueShapes dataset](https://engineering.purdue.edu/kak/distDLS/) is used for training and evaluation for the semantic segmentation. You can download the COCO dataset from the official [COCO website](https://cocodataset.org/#download) or use the DatasetGenerator class as provided in the code attached to download COCO filtered subset images.

Inintially we have also used the PurdueShapes dataset for testing the model architecture.

The directory structure should look something similar to this:


```
CocoFilteredDataset/ 
└── train/
|   ├── motorcycle/
|   ├── dog/
|   └── cake/
└── test/
    ├── motorcycle/
    ├── dog/
    └── cake/
```

## Implementation:

1. Initial test on the PurdueShapes dataset for checking the operation for the Combined Dice-MSE Loss function was carried out using a pre existing script(from [DLStudio Library](https://engineering.purdue.edu/kak/distDLS/#106) by Prof. Avinash Kak's) called sematic_segmentation.
2. Once the proper operation was confirmed the same was implmenated for the Coco Dataset referenceing the implementation of semantic_segmentation.py script.
3. Complete implemntation documentation can be found in the attached PDF file, with mathematical explanation of dice loss and details on the model architecture and code.
1. To conduct an initial test on the PurdueShapes dataset to verify the operation for the Combined Dice-MSE Loss function, Dr. Avinash Kak's sematic segmentation script, which can be found in the [DLStudio Library](https://engineering.purdue.edu/kak/distDLS/#106), was used.
2. After the model gave pretty accurate results for PurdueShapes dataset, the semantic segmentation implementation was implemented for the Coco Dataset.
3. The PDF file that is attached contains all of the implementation documentation, including a mathematical justification for dice loss as well as information on the model's architecture and code.










    
