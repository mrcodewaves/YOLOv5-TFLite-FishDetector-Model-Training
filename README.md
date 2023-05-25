# YOLOv5-TFLite-FishDetector-Model-Training

This repository contains the code and resources used for training a custom YOLOv5 model to detect different species of salmonid fish. The model was trained using a dataset of images collected from various sources, including Google, Bing, and private fishing Facebook groups. The images were processed and annotated to create a robust dataset for training the model.

## Project Structure

The project is structured as follows:

- `main.py`: this script contains CLI commands for cloning YOLOv5 repo, installing requirements and dependencies, and training, testing, and exporting the YOLOv5 model to TFLite format. 
- `train.py`: This script is used to train the YOLOv5 model with custom parameters including image size, batch size, and number of epochs.
- `detect.py`: This script is used to test the trained model on a set of test images.
- `export.py`: This script is used to convert the trained model into TFLite format, which can be used for deployment on mobile devices.

## Training the Model

The model was trained using the YOLOv5 architecture, which is known for its balance between speed and accuracy. The training was performed on a dataset of images of different species of salmonid fish. The images were resized to 416x416 pixels and the model was trained for 50 epochs.

## Testing the Model

The trained model was tested on a separate set of images to evaluate its performance. The model was able to accurately detect and classify the different species of salmonid fish in the test images.

## Exporting the Model

The trained model was exported to TFLite format using the `export.py` script. This allows the model to be used in mobile applications for real-time object detection.

## Usage

To train your own model, you can clone this repository and use the provided scripts. You will need to replace the dataset file with your own and adjust the parameters in the scripts as needed.

## Future Work

Future versions of the project may include expanding the recognition of species and improving the accuracy of the model. Contributions are welcome.

## License

This project is licensed under the MIT License. Please see the [LICENSE](LICENSE) file for details.
