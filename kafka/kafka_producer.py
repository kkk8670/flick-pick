import os
import json, csv
import time
import logging
from kafka import KafkaProducer
 
from flick_pick import ROOT_DIR



# data_path = get_project_root() / "data/test/movies.csv"
topic = 'flick-pick'
bootstrap_server = ['localhost:9093'] 

# producer = KafkaProducer(bootstrap_servers=bootstrap_server,
#                          value_serializer=lambda v: json.dumps(v).encode('utf-8'))


# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

batch_size = 10
 
# while True:
#     with open(file_path, "r") as f:
#         csv_reader = csv.DictReader(f)
#         for i, row in enumerate(csv_reader):
#             producer.send(topic, row)
#             if not i % batch_size:
#                 producer.flush()
#             logging.info(f"Sent row {count}: {row}")
#             time.sleep(0.5)  #   
    
#     logging.info("Finished one round, restarting...")

if __name__ == '__main__':
    print(project_root)