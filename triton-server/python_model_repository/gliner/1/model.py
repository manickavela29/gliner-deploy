# import app
import json
import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline
from gliner import GLiNER

# inferless_model = app.InferlessPythonModel()


class TritonPythonModel:
    def initialize(self, args):
        self.model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

    def execute(self, requests):
        responses = []
        for request in requests:
		    # Decode the Byte Tensor into Text 
            input = pb_utils.get_input_tensor_by_name(request, "text")
            labels = pb_utils.get_input_tensor_by_name(request, "labels")
            
            input_string = input.as_numpy()[0].decode()	       
            
            # Labels for entity prediction           
            labels_string = labels.as_numpy()[0].decode()
            labels_list = [label for label in labels_string.split(',')]

            # Perform entity prediction
            entities = self.model.predict_entities(input_string, labels_list, threshold=0.5)

            # Encode the text to byte tensor to send back
            inference_response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "entities",
                    np.array(entities),
                    )
            ])
            responses.append(inference_response)
	    
        return responses

    def finalize(self, args):
         self.generator = None