import pprint
import requests
import os
from dotenv import load_dotenv

load_dotenv()

pprint.pprint(requests.get(os.getenv("TEST_JSON_URL")).json())