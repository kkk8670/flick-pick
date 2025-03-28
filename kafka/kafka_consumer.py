from kafka import KafkaConsumer
import json

 
topic = 'flick-pick'
bootstrap_server = ['localhost:9093']   
 
consumer = KafkaConsumer(topic,   
                         bootstrap_servers=bootstrap_server,
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                         group_id='movie_group')
 
for message in consumer:
    data = message.value
    print(f"Consumed data: {data}")
    
    