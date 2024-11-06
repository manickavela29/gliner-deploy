Gliner model deployed on Triton server with python backend

# Pick the pre-made model repository
mv python_model_repository model_repository

# Pull and run the Triton container & replace yy.mm
# with year and month of release. Eg. 23.05
docker run --gpus='"device=0"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.10-py3 bash

# Install dependencies
pip install torch torchvision
pip install transformers
pip install gliner

# Launch the server
tritonserver --model-repository=/models

Client example
--------------

curl --location --request POST 'http://localhost:7000/v2/models/gliner/infer' \
--header 'Content-Type: application/json' \
--data-raw '{
   "inputs":[
   {    
    "name": "text",
    "shape": [1],
    "datatype": "BYTES",
    "data":  ["I really enjoyed this, born on date 26th oct"]
   },
   {
    "name": "labels",
    "shape": [1],
    "datatype": "BYTES",
    "data":  ["Person, Award, Date, Competitions, Teams"] 
   }
  ]
}
'

# Sample text for entity prediction
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

# Labels for entity prediction
labels = ["Person", "Award", "Date", "Competitions", "Teams"]