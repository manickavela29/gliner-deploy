# Gliner Model Deployment on Triton Server (Python Backend)

This guide provides steps to deploy the **Gliner NER model** on Triton Server using the Python backend.

# Deploy with Dockerfile

## Build the Docker image:

```bash
docker build -t gliner-triton-server .
```
## Run the Docker container:

```bash
docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 gliner-triton-server
```

The server will automatically start on container launch, exposing the standard Triton ports (8000, 8001, 8002) for HTTP, gRPC, and metrics access. This setup enables quick deployment and testing of the Gliner model with Triton Inference Server in a fully containerized environment. Let me know if further customization is needed!


# Deploy in terminal locally

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

labels = ["Person", "Award", "Date", "Competitions", "Teams"]
```