# Wheat Disease Analysis App

![GitHub repo size](https://img.shields.io/github/repo-size/your-username/your-repo)
![GitHub contributors](https://img.shields.io/github/contributors/your-username/your-repo)
![GitHub stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)

This web application provides a powerful tool for the early detection of diseases in wheat plants. By leveraging a deep learning model, it can accurately classify wheat plant images into different disease categories, helping farmers and researchers to take timely action and prevent crop losses.

## The Problem

Wheat is a staple food for a large part of the world's population. However, wheat crops are susceptible to various diseases that can significantly reduce yield and quality. Early and accurate detection of these diseases is crucial for effective management and control. Traditional methods of disease detection can be time-consuming, labor-intensive, and require expertise. This project aims to automate the process of wheat disease detection using computer vision and deep learning.

## Features

*   **Accurate Disease Classification:** The application uses a state-of-the-art deep learning model to classify wheat diseases with high accuracy. It can identify the following conditions:
    *   **Crown and Root Rot:** A fungal disease that affects the crown and roots of the wheat plant.
    *   **Healthy Wheat:** A healthy wheat plant with no visible signs of disease.
    *   **Leaf Rust:** A fungal disease that appears as small, circular, orange-to-brown pustules on the leaves.
    *   **Wheat Loose Smut:** A fungal disease that replaces the grains with black, powdery spores.
*   **User-Friendly Web Interface:** The application provides a simple and intuitive web interface that allows users to easily upload an image and get a prediction.
*   **Fast and Responsive:** The application is designed to be fast and responsive, providing predictions in a matter of seconds.

## Tech Stack

This project is built using a combination of powerful and popular technologies:

*   **Backend:**
    *   **Python:** A versatile and powerful programming language that is widely used in web development and machine learning.
    *   **Flask:** A lightweight and flexible web framework for Python.
*   **Machine Learning:**
    *   **TensorFlow:** An open-source machine learning framework developed by Google.
    *   **Keras:** A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
    *   **Scikit-learn:** A machine learning library for Python that provides simple and efficient tools for data mining and data analysis.
*   **Image Processing:**
    *   **OpenCV:** A library of programming functions mainly aimed at real-time computer vision.
    *   **Pillow:** A powerful image processing library for Python.
*   **Frontend:**
    *   **HTML:** The standard markup language for creating web pages.
    *   **CSS:** A stylesheet language used for describing the presentation of a document written in a markup language like HTML.
    *   **Bootstrap:** A popular CSS framework for developing responsive and mobile-first websites.

## Workflow

The application follows a simple and efficient workflow:

1.  **Image Upload:** The user selects an image of a wheat plant through the web interface. The image is sent to the Flask backend.

2.  **Image Preprocessing:** The backend receives the image and preprocesses it to make it suitable for the model. This involves:
    *   **Resizing:** The image is resized to 224x224 pixels, which is the input size of the model.
    *   **Normalization:** The pixel values are normalized to be in a range that is suitable for the model.

3.  **Prediction:** The preprocessed image is passed to the trained classification model. The model predicts the probability of the image belonging to each of the four classes.

4.  **Display Result:** The application identifies the class with the highest probability and displays the corresponding disease label to the user, along with the uploaded image.

```mermaid
graph TD;
    A[User opens the web application] --> B{index.html};
    B --> C[User selects an image];
    C --> D[User clicks "Detect Disease"];
    D --> E{Flask Backend /predict};
    E --> F[Image Preprocessing];
    F --> G[Classification Model];
    G --> H[Get Prediction];
    H --> I{result.html};
    I --> J[Display result to the user];
```

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

*   Python 3.7 or higher
*   pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Flask application:**
    ```bash
    python src/app.py
    ```
2.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

1.  Open the web application in your browser.

    ![App Screenshot 1](<image_placeholder_for_screenshot_1>)

2.  Click on the "Choose File" button to select an image of a wheat plant.

3.  Click on the "Detect Disease" button to upload the image and get the prediction.

4.  The application will display the uploaded image and the predicted disease label.

    ![App Screenshot 2](<image_placeholder_for_screenshot_2>)

## Model

### Classification Model

The application uses a classification model to identify wheat diseases.

*   **Architecture:** The model is based on the **VGG19** architecture, a convolutional neural network that is 19 layers deep. The VGG19 model, pre-trained on the ImageNet dataset, is used as a feature extractor. A custom classification head is added on top of the VGG19 base. The custom head consists of:
    *   An `AveragePooling2D` layer to reduce the spatial dimensions.
    *   A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    *   A `Dense` layer with 128 neurons and a `relu` activation function.
    *   A `Dropout` layer with a rate of 0.5 to prevent overfitting.
    *   A final `Dense` layer with 4 neurons (one for each class) and a `softmax` activation function to output the class probabilities.

*   **Training:** The model was trained on the images in the `data/classification/Images` directory. The training process involves:
    *   Loading the images and their corresponding labels.
    *   Splitting the data into training and validation sets.
    *   Using the `Adam` optimizer with a learning rate of `1e-4`.
    *   Training the model for 20 epochs with a batch size of 32.

*   **Retraining:** To retrain the model, you can run the `classification_model_training.py` script:
    ```bash
    python scripts/classification_model_training.py
    ```

*   **Model file:** The trained model is saved at `src/models/classification_model.h5`.

## Dataset

The classification dataset is located in the `data/classification/Images` directory. It is organized into subdirectories, where each subdirectory name corresponds to a different class of wheat disease or healthy wheat.

| Class                | Number of Images |
| -------------------- | ---------------- |
| Crown and Root Rot   | 159              |
| Healthy Wheat        | 150              |
| Leaf Rust            | 150              |
| Wheat Loose Smut     | 150              |

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/your-username/your-repo](https://github.com/your-username/your-repo)
