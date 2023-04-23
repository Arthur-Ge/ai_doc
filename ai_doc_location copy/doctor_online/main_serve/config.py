REDIS_CONFIG = {
     "host": "0.0.0.0",
     "port": 6379
}


NEO4J_CONFIG = {
    "uri": "bolt://0.0.0.0:7687",
    "auth": ("neo4j", "12345678"),
    "encrypted": False
}

model_serve_url = "http://0.0.0.0:9999/v1/recognition/"

TIMEOUT = 2

reply_path = "./reply.json"

ex_time = 60
