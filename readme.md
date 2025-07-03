# Maximizing ROI through Early Crop Disease Detection for an Agricultural Farm

## Project Overview

This project proposes an innovative solution leveraging **computer vision and machine learning technologies for the early detection of crop diseases**. It addresses the substantial financial losses faced by agricultural enterprises, such as Sasya’s Farm in Kottayam, Kerala, due to decreased yields and increased production costs caused by crop diseases. By analyzing image data of crops, advanced machine learning models can identify disease symptoms at their initial stages, enabling timely intervention to prevent disease spread and minimize damage. This approach aims to optimize resource allocation, enhance overall productivity, and promote sustainable, technology-driven agriculture.

## Key Features

*   **Early Disease Detection:** Identifies symptoms of crop diseases at their initial stages through image analysis.
*   **Machine Learning Models:** Utilizes **transfer learning with pre-trained deep learning models** (MobileNetV2, VGG19, ResNet101) for robust feature extraction and classification.
*   **Diverse Crop and Disease Coverage:** Trained on a comprehensive dataset comprising **93 distinct classes**, covering multiple crop types (e.g., Banana, Chilli, Cucumber, tomato, potato, maize) and various diseases (e.g., Leaf blight, Mosaic virus, leaf rust, powdery mildew), alongside healthy samples.
*   **Robustness through Data Augmentation:** Employs advanced techniques such as **rotation, flipping, zooming, shearing, brightness adjustment, and Gaussian noise addition** to artificially expand the dataset, enhance model generalization, and address class imbalances.
*   **Interactive Web Application:** Deployed via **Streamlit**, providing a user-friendly interface for farmers to upload crop images and receive instant predictions regarding disease presence and type.
*   **High Performance:** The fine-tuned MobileNetV2 model achieved an **overall accuracy of 88.52%** on validation and testing datasets.


## Technologies Used

*   **Programming Language:** Python 3.12
*   **Deep Learning & Computer Vision Libraries:**
    *   OpenCV 4.10
    *   TensorFlow/Keras
*   **Machine Learning Models:**
    *   **MobileNetV2:** A lightweight Convolutional Neural Network (CNN) known for its efficiency and suitability for mobile devices, comprising 19 residual bottleneck layers. It was chosen for its **small memory footprint and fast inference times**, making it ideal for real-time applications.
    *   **VGG19:** A deep CNN with 19 weight layers, valued for its straightforward architecture and effectiveness in feature extraction.
    *   **ResNet101:** A very deep CNN featuring **skip connections** to overcome the vanishing gradient problem, trained on over a million images from the ImageNet database.
*   **Development Environment:** Google Colab, utilizing a T4 GPU for accelerated computations.
*   **Deployment Framework:** Streamlit, for building interactive web applications.

## Dataset Details

*   **Sources:** The dataset was compiled from **publicly available Kaggle datasets**, primarily the Plant Village dataset (`https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset`), and augmented with data and insights from the regional 'Krishi Bhavan' in Kerala.
*   **Total Images:** **66,807 images**.
*   **File Types:** Images are stored in JPG and PNG formats.
*   **Classes:** Comprises **93 distinct classes**, covering multiple crop types (e.g., tomato, potato, maize) and a wide array of diseases (e.g., leaf rust, powdery mildew), alongside a "healthy" class for unaffected plants.
    *   **Top 3 Classes by Count:** Tomato Late blight (3113), Tomato healthy (3051), Tomato Septoria leaf_spot (2882).
    *   **Bottom 3 Classes by Count:** Lettuce Wilt and leafblight on lettuce (6), Cauliflower Black rot (8), Cauliflower Mosaic (11).
*   **Preprocessing:** All images were preprocessed and resized to a **standard dimension of 224x224 pixels** for model compatibility. The dataset was systematically split into training (70%), validation (15%), and testing (15%) subsets to ensure unbiased model evaluation.

## Model Training and Performance

*   **Experimental Setup:** Experiments were performed on a Windows 10 OS with 16 GB RAM and an Intel Core i5 processor. Fine-tuning of the CNN models was carried out on **Google Colab using a T4 GPU**.
*   **Training Process:** Models were fine-tuned on the training dataset for **5 epochs**, utilizing **categorical cross-entropy as the loss function** and the **Adam optimizer** for gradient-based optimization. A batch size of 128 was used, with early stopping and learning rate scheduling implemented to prevent overfitting and ensure stable convergence.
*   **Model Comparison (Performance on ImageNet):**
    | Feature         | MobileNetV2   | VGG19          | ResNet101      |
    | :-------------- | :------------ | :------------- | :------------- |
    | Input Image Size | 224x224       | 224x224        | 224x224        |
    | Parameters      | ~4.2 million  | ~143 million   | ~44.5 million  |
    | Top-1 Accuracy (ImageNet) | ~70%         | ~71%           | ~76%           |
    | Memory          | Low           | High           | Moderate       |
   

*   **Overall Performance (Fine-tuned Models on Project Dataset):**
    | Metric    | MobileNetV2 | VGG19 | ResNet101 |
    | :-------- | :---------- | :---- | :-------- |
    | Accuracy  | **88.52%**  | 81.91% | 86.71%    |
    | Precision | 88.52%      | 82.10% | 87.07%    |
    | Recall    | 88.52%      | 81.91% | 86.71%    |
    | F1 Score  | 88.38%      | 81.20% | 86.33%    |
   

*   **MobileNetV2 emerged as the best-performing model** for this specific task, demonstrating superior accuracy and computational efficiency compared to ResNet101 and VGG19. Data augmentation played a pivotal role, leading to a **6% improvement in accuracy** and significantly enhancing the model's robustness. Common diseases like leaf spots and rust were consistently detected with high confidence.

## Deployment & Usage

The trained crop disease detection model has been **successfully deployed as an interactive web application using Streamlit**.

**Access the Web Application here:** `https://farm-disease-classification.streamlit.app`

Users can upload images of their crops to receive **instant predictions** regarding the presence and specific type of disease. The model has been successfully tested with farm crops, including banana plants, and accurately diagnosed common banana diseases like Black Sigatoka and Banana Panama disease, demonstrating its practical utility in real agricultural settings.

## Challenges & Future Work

While the application of CNNs in disease detection is promising, real-world implementation presents several challenges and opportunities for future improvement:

*   **Real-world Variability:** Variations in lighting conditions, weather, and plant growth stages in natural environments can impact model accuracy. Fine-tuning and adaptation of models to diverse environmental conditions will be crucial for robust deployment.
*   **Small Dataset Problem:** Addressing the limited dataset problem in deep learning for plant pest identification remains a significant obstacle.
*   **Connectivity in Remote Areas:** The availability of reliable and consistent internet connectivity in remote farming areas can affect the feasibility of using mobile applications for disease detection.

### Recommendations for Continual Improvement

*   **Scalability:** The model can be expanded to include more crop types and diseases by collecting additional data.
*   **Integration:** Further integration into mobile applications is recommended, allowing farmers to receive **real-time diagnoses** directly on their devices.
*   **Localized Training:** Incorporating region-specific data, such as information gathered from local agricultural offices (e.g., Krishi Bhavan Kerala), can significantly enhance the model's relevance and accuracy for targeted areas.
*   **Continuous Learning:** Regular updates to the dataset with new images depicting various disease stages, different crops, and changing environmental conditions are essential for building a more comprehensive and accurate model.
*   **Technology Adoption:** Encouraging farms to invest in advanced technologies like drones for remote monitoring and IoT devices for real-time data collection can improve efficiency and reduce crop losses.
*   **Farmer Empowerment:** Organizing workshops and training sessions for farmers will familiarize them with the deployed system, enabling effective technology usage.
*   **Ongoing Data Collection:** Continuously gathering data from farms is vital to improve the model’s accuracy and reliability over time.

