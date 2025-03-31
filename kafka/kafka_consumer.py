from kafka import KafkaConsumer
import json
from kafka_producer import topic, bootstrap_server 
 
  
 
consumer = KafkaConsumer(topic,   
                         bootstrap_servers=bootstrap_server,
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                         group_id='movie_group')
 
for message in consumer:
    data = message.value
    print(f"Consumed data: {data}")
    
    