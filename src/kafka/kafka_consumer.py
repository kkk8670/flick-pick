#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/01

import json
import logging
from kafka import KafkaConsumer
from kafka_producer import topic, bootstrap_server 
 

# log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Consumer')


def create_consumer():
    return KafkaConsumer(topic,   
            bootstrap_servers=bootstrap_server,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='movie_group')


def run_consumer():
    logger.info("Start kafka_consumer!")

    consumer = create_consumer()
    for message in consumer:
        data = message.value
        logger.info(f"Consumed data: {data}")
        f"Offset: {message.offset}, Partition: {message.partition}, Key: {message.key}, Value: {message.value}"
    

if __name__ == "__main__":
    run_consumer()
    