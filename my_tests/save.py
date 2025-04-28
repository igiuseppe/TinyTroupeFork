from tinytroupe.factory import TinyPersonFactory
from tinytroupe import control
from tinytroupe.agent.grounding import LocalFilesGroundingConnector
from tinytroupe.agent import RecallFaculty
import json
from dotenv import load_dotenv
load_dotenv()

control.begin("my_tests/vacca.cache.json")

general_context=f"""
You have to generate the digital twin of the person interviewed in the following text. Try to recall from the interview all the details you can about the person. Do not make up any details.
"""

factory = TinyPersonFactory(general_context)

with open('my_tests/interview_example.txt', 'r') as file:
    agent_particularities = file.read()

print("Generating the digital twin of the person interviewed in the text...")
person=factory.generate_person(agent_particularities=agent_particularities)
control.checkpoint()

with open('my_tests/ground.txt', 'r') as file:
    ground = file.read()
person.think(f"Here is a text that contains the summary of a previous interview that I had. Whenever I am asked a question, I will try to recall it and answer the question using it.\nThe text is:\n{ground}")

print("Asking the person questions...")
ans=person.listen_and_act("what is your favorite color? always recall your memories before answering",return_actions=True)

specifications=person.save_specification(path=None,include_memory=True)
control.checkpoint()
control.end()

with open("my_tests/vacca1.json", "w") as f:
    json.dump(specifications, f, indent=4)
