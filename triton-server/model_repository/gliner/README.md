# Gliner Model Deployment on Triton Server (Python Backend)

This guide provides steps to deploy the **Gliner NER model** on Triton Server using the Python backend.

## 1. Set Up Model Repository

First, move the pre-made model repository into place:
```bash
mv python_model_repository model_repository
```
## 2. Pull and Run the Triton Container

Replace yy.mm with the appropriate year and month of release (e.g., 23.05).

```bash
docker run --gpus='"device=0"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.10-py3 bash
```
## 3. Install Dependencies

Inside the container, install the necessary Python packages:

```bash
Copy code
pip install torch torchvision
pip install transformers
pip install gliner
```
## 4. Launch the Triton Server

Start the Triton server with the model repository:

```bash
tritonserver --model-repository=/models
```
## Client Example

You can test the server with a simple curl command to send an inference request:

```bash
curl --location --request POST 'http://localhost:7000/v2/models/gliner/infer' \
--header 'Content-Type: application/json' \
--data-raw '{
   "inputs":[
       {    
           "name": "text",
           "shape": [1],
           "datatype": "BYTES",
           "data": ["I really enjoyed this, born on date 26th Oct"]
       },
       {
           "name": "labels",
           "shape": [1],
           "datatype": "BYTES",
           "data": ["Person, Award, Date, Competitions, Teams"]
       }
   ]
}'
```
## Sample Text for Entity Prediction

To use the model with sample text, try the following:

```python
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a 
Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr 
and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five 
Ballon d'Or awards, a record three UEFA Men's Player of the Year Awards
"""
```