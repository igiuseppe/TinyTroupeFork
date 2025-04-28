from dotenv import load_dotenv
load_dotenv()

import json
import sys
sys.path.insert(0, '..')

import tinytroupe
from tinytroupe.agent import TinyPerson

person=TinyPerson.load_specification("my_tests/vacca1.json")

ans=person.listen_and_act("which companies and roles could be interested in synthetic interviews?",return_actions=True)

specifications=person.save_specification(path=None,include_memory=True)

with open("my_tests/vacca2.json", "w") as f:
    json.dump(specifications, f, indent=4)