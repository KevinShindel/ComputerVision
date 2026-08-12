# Annotations

This project focuses on creating and managing annotations for computer vision tasks.

## Option 1. Using ultralytics Annotator
- See notebook `Ultralytics.ipynb` for more details

`from ultralytics.data.annotator import auto_annotate` <br/> - Automatically annotate images in a dataset
`from ultralytics.data.utils import visualize_image_annotations` - Visualize annotations on images

Cons: 
- Limited to the capabilities of the ultralytics library
- Need visualize all images to see the annotations ( consuming time and resources )

Pros: 
- Easy to use and integrate with existing ultralytics workflows
- Supports various annotation formats and types
- Fast and efficient for large datasets

## Option 2. Using CVAT (Computer Vision Annotation Tool)
TODO: Investigate this tool

## Option 3. Using LabelStudio + ML-Backend
- See notebook `LabelStudio.ipynb` for more details

Cons:
- Requires setting up a separate server for LabelStudio and ML-Backend
- More complex setup and configuration compared to ultralytics Annotator
- May require additional resources for running the server and managing the backend
- Template in LabelStudio must be the same as the model output format, otherwise it will not work

Pros:
- Provides a web-based interface for annotation, making it easier for multiple users to collaborate
- Supports various annotation formats and types
- Auto-Annotation can be integrated with ML-Backend for improved efficiency ( batch annotation )
- Easy to manage and organize annotations, especially for large datasets

## Option 4. Using FiftyOne
TODO: Investigate this tool

## Option 5. Using Supervisely
TODO: Investigate this tool