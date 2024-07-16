from flask import Flask, g, request, after_this_request, abort
import json
import random
from pysondb import db
import time
from datetime import datetime
from virtual_skeleton import DialogueGraph, BM25Retriever, DenseRetriever, ResponseGenerator 

app = Flask(__name__)

with open("data/tasks_testdata.json", "r") as file:
    setup_data = json.load(file)

database = db.getDb("experiment/results.json")

chatbot = ResponseGenerator()
sparse_retriever = BM25Retriever()
dense_retriever = DenseRetriever()

#TASK_DATA = random.choice(setup_data)
#dialogue_graph = DialogueGraph()

restart = False
#UTT_IDX = 0

INTERACTIONS = dict()

def load_graph(data):
        
    event_names = dict()
    edges = list()
    for e in data["events"]:
        event_names[e["id"]] = e["name"]
        edges.append(tuple((e["id"],"name",e["name"])))
        edges.append(tuple((e["id"],"start_time",e["start_time"])))
        edges.append(tuple((e["id"],"end_time",e["end_time"])))
        edges.append(tuple((e["id"],"date",e["date"])))
        edges.append(tuple((e["id"],"location",e["location"])))
        edges.append(tuple((e["id"],"organizer",e["organizer"])))
        for a in e["attendees"]:
            edges.append(tuple((a,"attending",e["id"])))
    for p in data["people"]:
        edges.append(tuple((p["name"],"office",p["office"])))
        edges.append(tuple((p["name"],"phone",p["phone"])))
        edges.append(tuple((p["name"],"email",p["email"])))
        edges.append(tuple((p["name"],"member",p["group"])))

    today = datetime.today().strftime('%Y-%m-%d')
    edges.append(tuple(("today","is",today)))

    graph = DialogueGraph(triples=edges)
    graph.lookup = event_names
    return graph


# ---------
# Callbacks
# ---------

#@app.url_value_preprocessor
#def get_site(endpoint, values):
#    print(f"In url_value_preprocessor callback... endpoint: {endpoint}, values: {values}")
#    g.username = values.pop('username_slug', None)


@app.before_request
def before_request():
    print("In before_request callback...")


@app.after_request
def after_request(response):
    print("In after_request callback...")
    return response


@app.teardown_request
def teardown_request(error):
    print(f'In teardown_request callback...')


@app.teardown_appcontext
def teardown_appcontext(error):
    print(f'In teardown_appcontext callback...')


# ------
# Routes
# ------


@app.route('/llm', methods=['POST'])
def send_llm_response():
    #global TASK_DATA
    #global UTT_IDX
    #global dialogue_graph
    global INTERACTIONS
    global sparse_retriever
    global dense_retriever
    global chatbot
    global restart

    recv_json = request.get_json()
    response = {"message": ""}
   
    print(recv_json)

    if "signal" in recv_json:

        if recv_json["signal"] == "start":
            user_id = recv_json["user_id"]
            task_count = recv_json["task_count"]

            if restart == True:
                if user_id in INTERACTIONS and task_count in INTERACTIONS[user_id]:
                    INTERACTIONS[user_id][task_count]["task"]
                    response = {"task": INTERACTIONS[user_id][task_count]["task"]}
                else:
                    if not user_id in INTERACTIONS:
                        INTERACTIONS[user_id] = dict()
                        INTERACTIONS[user_id]["order"] = ["baseline_nonverbal","baseline_verbal", "non_enrich_dense", "enrich_dense"]
                        random.shuffle(INTERACTIONS[user_id]["order"])
                    INTERACTIONS[user_id][task_count] = dict()
                    interaction_data = random.choice(setup_data)
                    INTERACTIONS[user_id][task_count]["task"] = interaction_data["task"]
                    INTERACTIONS[user_id][task_count]["data"] = interaction_data["data"]
                    INTERACTIONS[user_id][task_count]["graph"] = load_graph(interaction_data["data"])
                    INTERACTIONS[user_id][task_count]["turns"] = list()
                    INTERACTIONS[user_id][task_count]["utt_idx"] = 0
                    INTERACTIONS[user_id][task_count]["model"] = INTERACTIONS[user_id]["order"][task_count]
                    response = {"task": interaction_data["task"]}
                    print("Response:", response)
                return response
            
            if not user_id in INTERACTIONS:
                INTERACTIONS[user_id] = dict()
                INTERACTIONS[user_id]["order"] = ["baseline_nonverbal","baseline_verbal", "bm25", "dense"]
                random.shuffle(INTERACTIONS[user_id]["order"])


            INTERACTIONS[user_id][task_count] = dict()
            interaction_data = random.choice(setup_data)
            INTERACTIONS[user_id][task_count]["task"] = interaction_data["task"]
            INTERACTIONS[user_id][task_count]["data"] = interaction_data["data"]
            INTERACTIONS[user_id][task_count]["graph"] = load_graph(interaction_data["data"])
            INTERACTIONS[user_id][task_count]["turns"] = list()
            INTERACTIONS[user_id][task_count]["utt_idx"] = 0
            INTERACTIONS[user_id][task_count]["model"] = INTERACTIONS[user_id]["order"][task_count]
        
            # Create interaction object to save in db
            #INTERACTION = dict() # Remove previous data
            #INTERACTION["user_id"] = user_id
            #INTERACTION["task"] = TASK_DATA["task"]
            #INTERACTION["turns"] = list()
            #INTERACTION["data"] = TASK_DATA["data"]
            
            #dialogue_graph = load_graph(TASK_DATA["data"]) 
            #print("Loaded graph with %d edges" %len(dialogue_graph.edges))
            print("Response:", response)
            response = {"task": interaction_data["task"]}
            return response
        
        elif recv_json["signal"] == "stop" and recv_json["was_completed"] == False:
            user_id = recv_json["user_id"]
            task_count = recv_json["task_count"]
            response = {"task": INTERACTIONS[user_id][task_count]["task"]}
            restart = True
            print("Response:", response)
            return response

        elif recv_json["signal"] == "continue":
            
            turn_data = dict()

            utterance = recv_json["message"]
            user_id = recv_json["user_id"]
            task_count = recv_json["task_count"]
        
            #prediction = classifier(utterance)[0]
            #print("Prediction:", prediction)

            utt_idx = INTERACTIONS[user_id][task_count]["utt_idx"]

            turn_data["timestamp"] = time.time()
            turn_data["user_utterance"] = utterance

            utt_trans = tuple(("utt" + str(utt_idx), "transcription", utterance))
            utt_user = tuple(("utt" + str(utt_idx), "speaker", user_id))
            #dialogue_graph.add_triples([utt_trans,utt_user])
            INTERACTIONS[user_id][task_count]["graph"] = INTERACTIONS[user_id][task_count]["graph"].add_triples([utt_trans,utt_user])
            if utt_idx > 0:
                response_link = tuple(("utt" + str(utt_idx), "response_to", "utt" + str(utt_idx-1)))
                INTERACTIONS[user_id][task_count]["graph"] = INTERACTIONS[user_id][task_count]["graph"].add_triples([response_link])
            INTERACTIONS[user_id][task_count]["utt_idx"] += 1
            utt_idx = INTERACTIONS[user_id][task_count]["utt_idx"]

            if INTERACTIONS[user_id]["order"][task_count] == "baseline_raw":
                prompt = chatbot.create_prompt(INTERACTIONS[user_id][task_count]["graph"], no_subgraph=True, enrich=True, verbalise=False)
            elif INTERACTIONS[user_id]["order"][task_count] == "baseline_verbal":
                prompt = chatbot.create_prompt(INTERACTIONS[user_id][task_count]["graph"], no_subgraph=True, enrich=True, verbalise=True)
            elif INTERACTIONS[user_id]["order"][task_count] == "bm25":
                chatbot.retriever = sparse_retriever
                prompt = chatbot.create_prompt(INTERACTIONS[user_id][task_count]["graph"], enrich=True, verbalise=True)
            elif INTERACTIONS[user_id]["order"][task_count] == "dense":
                chatbot.retriever = dense_retriever
                prompt = chatbot.create_prompt(INTERACTIONS[user_id][task_count]["graph"], enrich=True, verbalise=True)

            #print(INTERACTIONS[user_id][task_count]["graph"].get_dialogue_history(utt_idx-1))
            # Create LLM Prompt with updated graph including new response

            prompt = chatbot.create_prompt(INTERACTIONS[user_id][task_count]["graph"])
            
            try:
                output = chatbot.decode_response(prompt)
            except:
                print("CUDA Error thrown, restarting model...")
                time.sleep(1)
                chatbot = ResponseGenerator()
                output = chatbot.decode_response(prompt)

            print("Prompt:", prompt)
            response["response"] = output

            turn_data["agent_response"] = output
            turn_data["prompt"] = prompt

            resp_robot = tuple(("utt" + str(utt_idx), "transcription", output))
            resp_user = tuple(("utt" + str(utt_idx), "speaker", "robot"))
            response_link = tuple(("utt" + str(utt_idx), "response_to", "utt" + str(utt_idx-1)))
            INTERACTIONS[user_id][task_count]["utt_idx"] += 1
            #dialogue_graph.add_triples([resp_robot,resp_user,response_link])
            
            INTERACTIONS[user_id][task_count]["graph"] = INTERACTIONS[user_id][task_count]["graph"].add_triples([resp_robot,resp_user,response_link])
            INTERACTIONS[user_id][task_count]["turns"].append(turn_data)
            
            #print(INTERACTIONS[user_id][task_count]["graph"])
            
            print("Response:", response)
            return response

        elif recv_json["signal"] == "stop" and recv_json["was_completed"] == True:
            
            user_id = recv_json["user_id"]
            task_count = recv_json["task_count"]

            questionnaire = recv_json["questionnaire_responses"]

            savedata = {"user_id" : user_id, "task_count" : task_count, "model": INTERACTIONS[user_id][task_count]["model"], "task": INTERACTIONS[user_id][task_count]["task"],
                    "turns" : INTERACTIONS[user_id][task_count]["turns"], "scores": questionnaire, "data": INTERACTIONS[user_id][task_count]["data"]}

            #INTERACTION["scores"] = questionnaire
            #INTERACTION["data"] = TASK_DATA["data"]
            database.add(savedata) # Add interaction data to DB
            #Reset variables
            #UTT_IDX = 0
            #dialogue_graph = DialogueGraph()
            response["message"] = "Dialogue finished"
            print("Response:", response)
            return response


if __name__ == '__main__':    
    app.run(host='0.0.0.0', port=443)
