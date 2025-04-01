import os
import json, csv
import time
import logging
from kafka import KafkaProducer
from pathlib import Path
from dotenv import load_dotenv

# env
load_dotenv()
file_path = Path(os.getenv('ROOT_DIR')) / "data/test/movies.csv"

# log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Producer')

# kafka
topic = 'flick-pick'
bootstrap_server = ['localhost:9092'] 

 
def create_producer():
    return KafkaProducer(
        bootstrap_servers=bootstrap_server,
        api_version=(3, 5),
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

 
def send_csv_data_to_kafka(producer, file_path, batch_size=10):
    with open(file_path, "r") as f:
        csv_reader = csv.DictReader(f)
        for i, row in enumerate(csv_reader):
            producer.send(topic, row)
            if not i % batch_size:
                producer.flush()
            logger.info(f"Sent row {i}: {row}")
            time.sleep(0.5)  

 
def run_producer():
    logger.info("Start kafka_producer!")
    producer = create_producer()

    while True:
        send_csv_data_to_kafka(producer, file_path)
        logger.info("Finished one round, restarting...")


if __name__ == "__main__":
    # producer.send('test_topic', b'Hello Kafka!')
    # producer.flush()
    # print("Message sent successfully!")
    # print(file_path)

    run_producer()