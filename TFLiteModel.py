import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

class TFLiteModel:
    def __init__(self, tflite_model_path):
        self.interpreter = tflite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Correct assignment of input indices
        self.input_index_color = self.input_details[0]['index']  # Color index input should be the first one
        self.input_index_image = self.input_details[1]['index']  # Image input should be the second one
        self.output_index = self.output_details[0]['index']

    def predict(self, color_input, image_input):
        self.interpreter.set_tensor(self.input_index_color, color_input)
        self.interpreter.set_tensor(self.input_index_image, image_input)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)
        return output