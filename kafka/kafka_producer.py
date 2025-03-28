from kafka import KafkaProducer
import json, csv
import time
import logging
from kafka_consumer import topic, bootstrap_server 


file_path = "../data/test/movies.csv"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
 
batch_size = 10
 
while True:
    with open(file_path, "r") as f:
        csv_reader = csv.DictReader(f)
        for i, row in enumerate(csv_reader):
            producer.send(topic, row)
            if not i % batch_size:
            	producer.flush()
        	logging.info(f"Sent row {count}: {row}")
            time.sleep(0.5)  #   
    
    logging.info("Finished one round, restarting...")