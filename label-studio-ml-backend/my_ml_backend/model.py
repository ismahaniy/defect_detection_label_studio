from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_local_path
from ultralytics import YOLO
import os
import logging
import threading
import time

# Configure logger
logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        """
        Loads the YOLO model once when the server starts.
        """
        # ---------------- CONFIGURATION ---------------- #
        # 1. Update this if your model has a different name
        MODEL_FILE = "best.pt" 
        
        # 2. Update these to match your Label Studio XML Config
        self.from_name = "label"  # Name of <RectangleLabels> tag
        self.to_name = "image"    # Name of <Image> tag
        self.score_threshold = 0.30 # Ignore predictions lower than 30% confidence
        # ----------------------------------------------- #

        # Load the model
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find model at {model_path}. Did you copy best.pt here?")
            
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        """
        Run inference on the images sent by Label Studio.
        """
        predictions = []

        for task in tasks:
            # 1. Get the local path to the image
            image_url = task['data']['image']
            image_path = get_image_local_path(image_url)
            
            # 2. Run the Model
            # imgsz can be adjusted if your images are huge
            results = self.model.predict(image_path, conf=self.score_threshold)
            
            # 3. Process Results
            formatted_results = []
            
            # YOLO returns a list of results (one per image). We only processed one.
            result = results[0] 
            
            # Get image dimensions to calculate percentages
            # orig_shape is (height, width)
            img_height, img_width = result.orig_shape

            for box in result.boxes:
                # Get coordinates (Pixels)
                # xyxy format: x_min, y_min, x_max, y_max
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                
                # Get confidence and class name
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                label_name = result.names[class_id] # e.g., 'scratch', 'dent'

                # CONVERT TO LABEL STUDIO FORMAT (Percentages 0-100)
                x = (x_min / img_width) * 100
                y = (y_min / img_height) * 100
                width = ((x_max - x_min) / img_width) * 100
                height = ((y_max - y_min) / img_height) * 100

                formatted_results.append({
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [label_name], # The class name
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    },
                    "score": score
                })

            predictions.append({
                "result": formatted_results,
                "score": 1.0 # Optional: task score
            })

        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])

        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        
        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
        """
        print(f"[ML Backend] Fit event received: {event}")
        print("[ML Backend] No training performed here")
        



