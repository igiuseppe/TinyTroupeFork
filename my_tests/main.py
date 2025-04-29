from dotenv import load_dotenv
load_dotenv()
from tinytroupe.factory import TinyPersonFactory
from tinytroupe import control
from tinytroupe.agent.grounding import LocalFilesGroundingConnector
from tinytroupe.agent import RecallFaculty
import json
import sys
sys.path.insert(0, '..')

import tinytroupe
from tinytroupe.agent import TinyPerson

def generate_factory(n):
    if n==1:
        general_context=f"""
You have to generate the digital twin of the person interviewed in the following text. 
Try to recall from the interview all the details you can about the person. 
Do not make up any details.
"""
    else:
        general_context=f"""
You have to generate different digital twins of the person interviewed in the following text. 
Try to recall from the interviews all the details you can about the person. Generate different digital twins modifying the name and other details of the person that are not specified in the text.
Do not make up any details.
"""
    factory = TinyPersonFactory(context_text=general_context)
    return factory

def generate_persons(agent_particularities,n):
    factory=generate_factory(n=n)
    temperature=1
    frequency_penalty=1
    presence_penalty=1
    persons=factory.generate_people(number_of_people=n,agent_particularities=agent_particularities,temperature=temperature,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty)
    return persons


def create_memory(person, memory):
    ground=f"""
Here is a text that contains the summary of a previous interview that I had. 
Whenever I am asked a question, I will try to recall it and answer the question using it.
The text is:
{memory}
"""
    person.think(ground)

def get_answer(actions):
    talk_content = []
    if actions:
        for item in actions:
            action_data = item.get('action', {})
            if action_data.get('type') == 'TALK':
                talk_content.append(action_data.get('content', ''))
    return " ".join(talk_content).strip()


def answer_question(person, question):
    actions = person.listen_and_act(speech=question,return_actions=True)
    return actions


def save_person(person):
    specifications=person.save_specification(path=None,include_memory=False,include_mental_faculties=False)
    return specifications

def save_memory(person):
    specifications=person.save_specification(path=None,include_memory=True,include_mental_faculties=False)
    memory=specifications.get("episodic_memory")
    return memory

def load_person(specifications,memory):
    specifications["episodic_memory"]=memory
    person=TinyPerson.load_specification(specifications)
    return person

def main():
    with open("my_tests/interview_example.txt", "r") as file:
        agent_particularities = file.read()
    with open("my_tests/ground.txt", "r") as file:
        ground = file.read()
    
    persons=generate_persons(n=1,agent_particularities=agent_particularities)
    person=persons[0]
    create_memory(person,ground)
    answer=answer_question(person,"what's your favorite color?")
    print(f"The answer is: {answer}")
    specs=save_person(person)
    specs["persona"]["name"]="John Big"
    memory=save_memory(person) 
    loaded_person=load_person(specs,memory)
    answer=answer_question(loaded_person,"and if you have to say your second favorite one?")
    print(f"The answer is: {get_answer(answer)}")
    print(loaded_person.retrieve_recent_memories(include_omission_info=False)[-len(answer)-1:])

if __name__ == "__main__":
    main()