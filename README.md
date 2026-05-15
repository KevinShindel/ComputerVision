# 🎯 ComputerVision

A comprehensive collection of computer vision projects, bootcamps, and learning materials covering OpenCV, deep learning, object detection, image processing, and advanced neural networks.

---

## 📚 Repository Structure

### 🔷 [OpenCV-Bootcamp](./OpenCV-Bootcamp)

#### 🎬 Intro
A complete interactive bootcamp for learning OpenCV fundamentals, covering image and video manipulation, object detection, face recognition, and real-time computer vision applications.

#### 📝 Description
This directory contains 13 comprehensive Jupyter notebooks that guide you through:
- Image manipulation and filtering techniques
- Edge detection and image features
- Object tracking and face detection
- Advanced topics like panorama stitching, HDR processing, and human pose estimation
- TensorFlow-based object detection workflows
- Real-time camera access and processing

#### 📦 Dependencies
```
opencv-python==4.13.0.92
numpy==2.4.4
matplotlib==3.11.0rc1
```

#### 🎯 Business Goals
- Master fundamental and advanced image processing techniques
- Develop skills in real-time computer vision applications
- Build practical solutions for object detection and face recognition
- Understand deep learning integration for visual AI tasks
- Create real-world applications using camera input and video processing

---

### 🔷 [OpenCVPython](./OpenCVPython)

#### 🎬 Intro
A structured, chapter-by-chapter progression through OpenCV fundamentals using Python, ideal for developers transitioning from basics to intermediate OpenCV programming.

#### 📝 Description
This directory contains 11 Python scripts organized as chapters covering:
- Image I/O and basic operations
- Drawing and annotation
- Image transformations and geometric operations
- Color space handling and filtering
- Contour detection and shape analysis
- Video processing and essential utilities
- Progressive complexity from chapter 1 to chapter 11

#### 📦 Dependencies
```
numpy==1.24.2
opencv-python==4.7.0.68
```

#### 🎯 Business Goals
- Provide a structured learning path for OpenCV programming
- Build proficiency in essential image processing operations
- Develop scriptable, reproducible computer vision solutions
- Create a reference library for common CV tasks
- Enable rapid prototyping of image processing algorithms

---

### 🔷 [RobotDreams](./RobotDreams)

#### 🎬 Intro
An advanced, comprehensive computer vision course structured in 21 lessons with real-world applications, advanced techniques, and cutting-edge deep learning frameworks.

#### 📝 Description
This directory contains extensive learning materials with:
- 20+ structured lessons covering beginner to advanced topics
- Workshop materials for hands-on practice
- Integration with TensorFlow, Keras, and state-of-the-art models (YOLOv8)
- Advanced techniques including optimization (Optuna), hyperparameter tuning
- Real datasets from Kaggle and TensorFlow Datasets
- Video processing and computer vision pipelines
- Utility functions for common CV tasks across lessons

#### 📦 Dependencies
```
datasets==4.8.5
opencv-python==4.13.0.92
numpy==2.4.4
ipywidgets==8.1.8
matplotlib==3.10.9
kagglehub==1.0.1
pandas==3.0.2
optuna==4.8.0
pillow==12.2.0
pydot==4.0.1
scikit-learn==1.8.0
seaborn==0.13.2
tensorflow==2.21.0
tensorflow-datasets==4.9.9
tensorflow-hub==0.16.1
tqdm==4.67.3
ultralytics==8.4.43
```

#### 🎯 Business Goals
- Develop production-ready computer vision solutions
- Master advanced deep learning techniques and modern architectures (YOLOv8, etc.)
- Learn hyperparameter optimization and model tuning strategies
- Implement scalable CV pipelines with real-world datasets
- Build expertise in transfer learning and pre-trained models
- Create end-to-end CV applications from data collection to deployment

---

### 🔷 [TF-Keras-Bootcamp](./TF-Keras-Bootcamp)

#### 🎬 Intro
A focused bootcamp on neural networks and deep learning using TensorFlow and Keras, covering everything from fundamentals to advanced CNN and object detection applications.

#### 📝 Description
This directory contains 9 comprehensive Jupyter notebooks covering:
- Fundamentals of neural network training and optimization
- Linear regression modeling
- Multi-layer perceptron (MLP) for digit classification
- Convolutional neural network (CNN) fundamentals
- CIFAR-10 image classification with CNNs
- Transfer learning using pre-trained ImageNet models
- Fine-tuning strategies for custom datasets
- Semantic segmentation techniques
- Object detection frameworks and applications

#### 📦 Dependencies
```
opencv-python==4.13.0.92
tensorflow==2.20.0
matplotlib==3.10.8
numpy==2.4.2
pandas==3.0.1
seaborn==0.13.2
```

#### 🎯 Business Goals
- Build strong foundation in neural network theory and practice
- Master CNN architectures for image classification
- Leverage transfer learning for efficient model development
- Implement advanced techniques: semantic segmentation and object detection
- Create production-quality deep learning models
- Understand model training pipelines, optimization, and evaluation metrics

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Installation

Clone the repository:
```bash
git clone <repository-url>
cd ComputerVision
```

Install dependencies for your chosen course:
```bash
# For OpenCV-Bootcamp
pip install -r OpenCV-Bootcamp/requirements.txt

# For OpenCVPython
pip install -r OpenCVPython/requirements.txt

# For RobotDreams
pip install -r RobotDreams/requirements.txt

# For TF-Keras-Bootcamp
pip install -r TF-Keras-Bootcamp/requirements.txt
```

### Running Jupyter Notebooks
```bash
jupyter notebook
```
Then navigate to your desired directory and open the notebooks.

---

## 📊 Learning Path Recommendations

**Beginner**: Start with `OpenCV-Bootcamp` for interactive learning
↓
**Intermediate**: Progress to `OpenCVPython` for deeper technical knowledge
↓
**Advanced**: Explore `TF-Keras-Bootcamp` and `RobotDreams` for modern AI applications

---

## 📄 License

See the [LICENSE](./LICENSE) file for details.

---

## 📧 Author

**Tymur Hilfatullin**

For questions and discussions, feel free to open an issue or contribute to the repository.
