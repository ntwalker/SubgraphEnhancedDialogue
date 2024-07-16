from __future__ import annotations

from abc import abstractmethod
import torch
import transformers
import sentence_transformers
from typing import Dict, Set, Tuple, Collection, List
import os, time, re, json
import numpy as np
import rank_bm25
from pcst_fast import pcst_fast
from datetime import datetime
import peft
import tqdm

#############################################
# RESPONSE GENERATION
#############################################

#SYSTEM_PROMPT = ("You are a helpful, friendly and reliable robot receptionist, and have access to the " + 
#                 "agenda and contact details of all employees of your organisation. Answer the user questions " + 
#                 "in 1-3 sentences, making sure your responses are ALWAYS grounded in the provided facts.")

SYSTEM_PROMPT = ("You are a robot receptionist and have access to a set of calendar data to provide to the user. Respond directly to the user " +
                 "in 1-3 sentences only, making sure your responses are ALWAYS grounded in the provided facts.")

class ResponseGenerator:

    def __init__(self, decoder_model="plison/graph-retrieval", encoder_model=None):

        # Initialise the retriever (using BM25 or dense retrieval) 
        if encoder_model is None:
            self.retriever = BM25Retriever()
        else:
            self.retriever = DenseRetriever(encoder_model)

        self.decoder_tokeniser = transformers.AutoTokenizer.from_pretrained(decoder_model)
        self.decoder_tokeniser.pad_token = self.decoder_tokeniser.eos_token

        try:
            print("Loading fine-tuned model...")
            self.decoder = peft.AutoPeftModelForCausalLM.from_pretrained(decoder_model, torch_dtype=torch.bfloat16, 
                                                                         device_map="auto", trust_remote_code=True)
        except:
            self.decoder = transformers.AutoModelForCausalLM.from_pretrained(decoder_model, device_map="auto",
                                                                          torch_dtype=torch.bfloat16)
        

    def create_prompt(self, graph: DialogueGraph, max_history=5, return_messages=False, enrich=True, verbalise=True, no_subgraph=False):
        """Returns the prompt containing the instructions, verbalised subgraph of background 
        knowledge, dialogue history and user query."""
        # STEP 1: graph enrichment
        if enrich:
            graph = self.enrich(graph)
        
        if no_subgraph:
            history = graph.get_dialogue_history(5)
            graph2 = graph.get_subset([(s,l,e) for (s,l,e) in graph.edges if not s.startswith("utt")])
            top_facts = self.retriever.get_most_relevant_edges(graph2, history)
            verbalised_subgraph = "\n".join([graph.verbalise([f],compose=verbalise) for f in top_facts]) # NB: Not a subgraph per se, this is a concatenation of the verbalizations of the top k facts (edges)
        else:
            # STEPS 2 and 3: find relevant subgraph
            #subgraph = self.get_relevant_subgraph(graph)
            subgraph = self.get_2hop_subgraph(graph)
    
            # STEP 4: verbalisation
            verbalised_subgraph = subgraph.verbalise(compose=verbalise)

        # Generate the main prompt
        user_prompt = "You have access to the following information:\n%s\n"%(verbalised_subgraph)

        if len(graph.get_dialogue_history()) > 1:
            history = "\n".join(graph.get_dialogue_history(include_speaker_names=True)[-max_history:-1])
            user_prompt += "The following dialogue has already taken place:\n%s\n"%history

        user_question = graph.get_dialogue_history()[-1]
        user_prompt += "The human user is now saying this:\n%s\n"%user_question
        #user_prompt += "As the robot, answer the user directly using the above facts:"
        user_prompt += "Robot: "

        # Some LLMS (like Llama) distinguish between the system instruction and the user prompt
    #    # while others (like Gemma) put everything in a single prompt
    #    if "gemma" in self.decoder.name_or_path:
        messages = [{"role":"user", "content": SYSTEM_PROMPT + "\n" + user_prompt}]
    #     messages = [{"role":"system", "content":SYSTEM_PROMPT},
    #                    {"role":"user", "content":user_prompt}]


        # Finally, apply the chat template for the model
        return self.decoder_tokeniser.apply_chat_template(messages, tokenize=False, 
                add_generation_prompt=True)
    

    def decode_response(self, prompt:str, max_tokens=200):
        """Generates the response given a prompt"""

        # STEP 5: generation
        prompt_tokens = self.decoder_tokeniser(prompt, return_tensors="pt").to("cuda")
        generated_tokens = self.decoder.generate(**prompt_tokens, max_new_tokens=max_tokens)
        response_tokens = generated_tokens[0, len(prompt_tokens["input_ids"][0]):]
        response = self.decoder_tokeniser.decode(response_tokens, skip_special_tokens=True).strip()

        return response
        
    def get_response(self, graph: DialogueGraph, debug=False):
        """Given a graph representing the current dialogue state (including the dialogue history),
        generate a response in five steps:
        1) Enrich the graph with the help of heuristic rules
        2) Determine relevant nodes and edges for the current dialogue based on the retriever
        3) Use the PCST algorithm to find a subgraph from those retrieved nodes and edges
        4) Verbalise the content of this subgraph
        5) Generate the response based on this verbalised content + the dialogue history
        """

        # STEPS 1-4: graph enrichment, subgraph extraction and verbalisation
        prompt = self.create_prompt(graph)
        if debug:
            print("=======\nFull prompt:", prompt, "\n=======")

        # STEP 5: decoding of the response
        response = self.decode_response(prompt)

        if debug:
            print("RESPONSE:", response)
        return response
        
    def get_response(self, graph: DialogueGraph, debug=False):
        """Given a graph representing the current dialogue state (including the dialogue history),
        generate a response in five steps:
        1) Enrich the graph with the help of heuristic rules
        2) Determine relevant nodes and edges for the current dialogue based on the retriever
        3) Use the PCST algorithm to find a subgraph from those retrieved nodes and edges
        4) Verbalise the content of this subgraph
        5) Generate the response based on this verbalised content + the dialogue history
        """

        # STEPS 1-4: graph enrichment, subgraph extraction and verbalisation
        prompt = self.create_prompt(graph)
        if debug:
            print("=======\nFull prompt:", prompt, "\n=======")

        # STEP 5: decoding of the response
        response = self.decode_response(prompt)

        if debug:
            print("RESPONSE:", response)
        return response


    def convert_graph_pcst(self, graph, relevant_nodes):
        """Converts a Dialogue Graph to a format suitable for the PCST fast algorithm 
        and returns adjacency matrix A, costs C, prizes, and a dictionary mapping edge indexes to original tuples.
        Edge cost is assumed to be uniformly 1, reimplement without np.ones for other values"""
        #Index nodes
        indices = dict()
        for i, n in enumerate(graph.nodes):
            indices[n] = i

        #Create adjacency matrix A
        A = []
        indexed_edges = dict() #Retain indices for PCST results
        prizes = []
        for i, e in enumerate(graph.edges):
            A.append(np.array([indices[e[0]], indices[e[2]]]))
            indexed_edges[i] = e
        
        #A = np.array(indexed_edges, dtype=np.float64)
        #Create edge cost matrix
        C = np.ones(shape=(len(A)), dtype=np.float64)

        #Assign prize to nodes
        for n in graph.nodes:
            if n in relevant_nodes:
                #Node prize is 10 - k (retrieval rank) as in G-Retriever implementation
                node_prize = 10 - relevant_nodes.index(n)
                prizes.append(node_prize)
            else:
                #Node prize is 0 if not in top k
                prizes.append(0)
        prizes = np.array(prizes, dtype=np.float64)
        return A, C, prizes, indexed_edges

    def convert_graph_pcst_with_virtual_nodes(self, graph, relevant_nodes, relevant_edges, edge_cost=1):
        """Converts a Dialogue Graph to a format suitable for the PCST fast algorithm 
        and returns adjacency matrix A, costs C, prizes, and a dictionary mapping edge indexes to original tuples.
        Edge cost is assumed to be uniformly 1, reimplement without np.ones for other values"""
        #Index nodes
        indices = dict()
        idx_to_label = dict()
        running_idx = 0
        
        for i, n in enumerate(graph.nodes):
            indices[n] = i
            idx_to_label[i] = n
            running_idx += 1

        #Create adjacency matrix A
        A = []
        C = []
        indexed_edges = dict() # Retain indices for PCST results
        prizes = []
        for i, e in enumerate(graph.edges):
            indexed_edges[e] = running_idx # Whole edge for relevance
            idx_to_label[running_idx] = e #[1] # Edge label only for the virtual "node"
            indices[e] = running_idx
        
            A.append(np.array([indices[e[0]], running_idx]))
            C.append(edge_cost/2)
            A.append(np.array([running_idx, indices[e[2]]]))
            C.append(edge_cost/2)
            running_idx += 1 # Indices created for edge virtual nodes
    
        #Create edge cost matrix
        C = np.array(C)

        #Assign prize to nodes
        for index, label in idx_to_label.items():
            if label in relevant_nodes:
                #Node prize is 10 - k (retrieval rank) as in G-Retriever implementation
                node_prize = len(relevant_nodes) - relevant_nodes.index(label)
                prizes.append(node_prize)
            elif label in relevant_edges:
                #Node prize is 10 - k (retrieval rank) as in G-Retriever implementation
                node_prize = len(relevant_edges) - relevant_edges.index(label) - (0.5)
                prizes.append(node_prize)
            else:
                #Node prize is 0 if not in top k
                prizes.append(0)
        prizes = np.array(prizes, dtype=np.float64)

        return A, C, prizes, indexed_edges

    def get_relevant_subgraph(self, graph, edge_cost=1):
        """Finds the most relevant nodes and edges in the graph based on the dialogue history,
        and determines a subgraph from those based on the PCST algorithm"""

        # Retrieves the dialogue history
        history = graph.get_dialogue_history(5)

        # For retrieval, we only consider content that is not related to the utterances themselves
        graph2 = graph.get_subset([(s,l,e) for (s,l,e) in graph.edges if not s.startswith("utt")])

        # Find relevant nodes and edges
        relevant_nodes = self.retriever.get_most_relevant_nodes(graph2, history,n=3)
        relevant_edges = self.retriever.get_most_relevant_edges(graph2, history,n=5)
        
        # Implementation of PCST
        # Convert graph to format for PCST
        #A, C, prizes, indexed_edges = self.convert_graph_pcst(graph2,relevant_nodes)
        A, C, prizes, indexed_edges = self.convert_graph_pcst_with_virtual_nodes(graph2,relevant_nodes,relevant_edges)

        # PCST
        vertices, edges = pcst_fast(A, prizes, C, -1, 1, 'gw', 0)
        selected_edges = [e for i, e in enumerate(graph2.edges) if indexed_edges[e] in vertices] #Clumsy

        return graph2.get_subset(selected_edges)

    def get_2hop_subgraph(self, graph, edge_cost=1):
        """Finds the most relevant nodes and edges in the graph based on the dialogue history,
        and determines a subgraph from those based on the PCST algorithm"""

        # Retrieves the dialogue history
        history = graph.get_dialogue_history(5)

        # For retrieval, we only consider content that is not related to the utterances themselves
        graph2 = graph.get_subset([(s,l,e) for (s,l,e) in graph.edges if not s.startswith("utt")])

        # Find relevant nodes and edges
        relevant_nodes = self.retriever.get_most_relevant_nodes(graph2, history)
        relevant_edges = self.retriever.get_most_relevant_edges(graph2, history)

        # Implementation of PCST
        # Convert graph to format for PCST
        selected_edges = []
        one_hop = graph2.get_undirected_edges(relevant_nodes[0])
        selected_edges.extend(one_hop)
        for start, label, end in one_hop:
            if start != relevant_nodes[0]:
                second_hop = graph2.get_undirected_edges(start)
                selected_edges.extend(second_hop)
            elif end != relevant_nodes[0]:
                second_hop = graph2.get_undirected_edges(end)
                selected_edges.extend(second_hop)

        return graph2.get_subset(selected_edges)
    
    def enrich(self, graph):
        """Enrich the graph with the help of heuristic rules."""
        new_edges = []
        distances = dict()
        event_sequence = dict()
        highs = dict()
        for (start, label, end) in graph.edges:
            if label == "attending":
                graph.get_outgoing_edges(end)
                atts = graph.get_outgoing_edges(end)
                new_edges.append(tuple((start, "busy", atts["start_time"].replace("30:00", "30").replace("00:00", "00") + " to " + atts["end_time"].replace("30:00", "30").replace("00:00", "00") +  ", " + atts["date"])))

            if label == "organizer":
                graph.get_outgoing_edges(start)
                atts = graph.get_outgoing_edges(start)
                new_edges.append(tuple((end, "busy", atts["start_time"].replace("30:00", "30").replace("00:00", "00") + " to " + atts["end_time"].replace("30:00", "30").replace("00:00", "00") +  ", " + atts["date"])))

            if label == "office":
                new_edges.append(tuple((end, "floor", end[0])))

            if label == "distance":
                distances[start] = int(end.strip(" miles"))

            if label == "date":
                if end.endswith("th"):
                    if start not in event_sequence:
                        event_sequence[start] = dict()
                    event_sequence[start]["date"] = int(end.split(" ")[-1].strip("th"))
                    time = graph.get_outgoing_edges(start)['time']
                    if "pm" in time:
                        time = int(time.strip("pm")) + 12
                    else:
                        time = int(time.strip("am"))
                    event_sequence[start]["time"] = time

            if "_weather" in label:
                #print(start, label, end)
                if start not in highs:
                    highs[start] = [int(end.split(" ")[-1].strip("F"))]
                else:
                    highs[start].append(int(end.split(" ")[-1].strip("F")))
                #time.sleep(1)

                
        for poi1, distance1 in distances.items():
            for poi2, distance2 in distances.items():
                if poi1 == poi2:
                    continue
                elif distance1 < distance2:
                    new_edges.append(tuple((poi1, "closer", poi2)))
                elif distance2 < distance1:
                    new_edges.append(tuple((poi2, "closer", poi1)))

        for poi1 in event_sequence:
            for poi2 in event_sequence:
                if poi1 == poi2:
                    continue
                elif event_sequence[poi1]["date"] > event_sequence[poi2]["date"] and event_sequence[poi1]["time"] > event_sequence[poi2]["time"]:
                    new_edges.append(tuple((poi1, "after", poi2)))
                elif event_sequence[poi2]["date"] > event_sequence[poi1]["date"] and event_sequence[poi2]["time"] > event_sequence[poi1]["time"]:
                    new_edges.append(tuple((poi2, "after", poi1)))

        for poi, temps in highs.items():
            highs[poi] = np.average(temps)
        
        for poi1, temp1 in highs.items():
            for poi2, temp2 in highs.items():
                if poi1 == poi2:
                    continue
                elif temp2 > temp2:
                    new_edges.append(tuple((poi1, "hotter", poi2)))
                elif temp2 > temp1:
                    new_edges.append(tuple((poi2, "hotter", poi1)))
            
        graph.add_triples(new_edges)

        return graph


#############################################
# DIALOGUE GRAPH
#############################################

    
class DialogueGraph:
    """Representation of a dialogue graph, with nodes and edges. The graph is
    here encoded as a simple set of triples. 
    
    The graph is assumed to also contain the utterances. The convention is that
    the the utterances should be labelled as 'uttX', where X is a number, and
    should be associated with at least two triples: 
    - one (uttX, transcription, utterance string),
    - one (uttX, speaker, name of speaker)"""

    def __init__(self, triples: Collection[Tuple[str,str,str]]=None, event_names=None):
        """Creates a new graph"""

        self.nodes:Set[str] = set()
        self.edges:Set[Tuple[str,str,str]] = set()
        
        self.lookup = event_names

        if triples is not None:
            self.add_triples(triples)
    
    def add_triples(self, triples: Collection[Tuple[str,str,str]]):
        """Adds triples to the graph"""

        for start_node, edge_label, end_node in triples:
            self.edges.add((start_node, edge_label, end_node))
            if start_node not in self.nodes:
                self.nodes.add(start_node)
            if end_node not in self.nodes:
                self.nodes.add(end_node)
        return self

    def get_dialogue_history(self, max_length:int=10, include_speaker_names:bool=False):
        """Returns the dialogue history, assuming the convention mentioned above.
        Only the max_length utterances are returned. If include_speaker_names is 
        set to True, the utterance are prefixed by 'speaker_name: '"""

        utt_nodes = []
        for i in range(100):
            if "utt%i"%i in self.nodes:
                utt_nodes.append("utt%i"%i)

        lines = []
        for utt_node in utt_nodes[-max_length:]:
            outgoing_edges = self.get_outgoing_edges(utt_node)
            if "transcription" not in outgoing_edges:
                raise RuntimeError("Strange: utterance %s has not transcription"%utt_node)
            elif "speaker" not in outgoing_edges:
                raise RuntimeError("Strange: utterance %s has no speaker label"%utt_node)
            line = str(outgoing_edges["transcription"])
            if include_speaker_names:
                line = outgoing_edges["speaker"] + ": " + line
            lines.append(line)
        return lines
    
    def get_outgoing_edges(self, node:str):
        """Returns a dictionary mapping edge labels to outgoing node for the provided node"""
        by_edge_label = {}
        for start_node, label, end_node in self.edges:
            if node==start_node:
                by_edge_label[label] = end_node
        return by_edge_label

    def get_undirected_edges(self, node:str):
        """Returns a list mapping edge labels to outgoing node for the provided node"""
        edges = []
        for start_node, label, end_node in self.edges:
            if node==start_node or node==end_node:
                edges.append((start_node, label, end_node))
        return edges

    def get_subset(self, triples: Set[str,str,str]):
        """Returns a new graph that only includes the provided triples"""
        return DialogueGraph(triples=triples,event_names=self.lookup)
    
    def verbalise(self, entities=None, compose=False):
        """Returns a verbalisation of the graph content (or part of it, if the entities argument
        is specified)"""

        # Implementation
        event_names = dict()
        seen_events = {}
        seen_people = {}
        lines = []

        if compose == True:
            for (start, label, end) in self.edges:

                if entities is not None and (start, label, end) not in entities and (start, label, end) != entities:
                    continue

                #if label in ["transcription"]:
                #    lines.append("The content of " + start + " was: " + end)
                #if label in ["speaker"]:
                #    lines.append(end + " was the speaker of " + start)
                #if label in ["response_to"]:
                #    lines.append(end + " responds to " + start)

                if label in ["name"]:
                    event_names[start] = end
                
                # KVRET navigation
                if label in ["distance"]:
                    lines.append(start + " is " + end + " away. ")
                if label in ["type"]:
                    lines.append(start + " is a " + end + ". ")
                if label in ["address"]:
                    lines.append("The address of " + start + " is " + end + ". ")
                if label in ["traffic"]:
                    lines.append("On the route to " + start + ", there is " + end + ". ")

                # KVRET weather
                if "_weather" in label:
                    lines.append("The weather in " + start + " on " + label.split("_")[0] + " is " + end + ". ")

                # KVRET scheduling
                if "attendee" in label:
                    lines.append(end + " is attending " + start + ". ")

                if label == "after":
                    lines.append(start + " will take place after " + end + ". ")

                if label == "hotter":
                    lines.append(start + " will be hotter on average than " + end + ". ")

                if label == "closer":
                    lines.append(start + " is closer to the user than " + end + ". ")

                if label in ["organizer", "start_time", "end_time", "date", "location"]:
                    if start not in seen_events:
                        seen_events[start] = {label : end.replace("30:00", "30").replace("00:00", "00")}
                    else:
                        seen_events[start][label] = end.replace("30:00", "30").replace("00:00", "00")

                if label in ["office", "member", "phone", "email", "group"]:
                    if label == "group":
                        lines.append(start + ' is associated with the ' + end + ' group. ')
                    if start not in seen_people:
                        seen_people[start] = {label : end}
                    else:
                        seen_people[start][label] = end

                if label == "attending":
                    if end not in seen_events:
                        seen_events[end] = {"attendees" : [start]}
                    elif "attendees" not in seen_events[end]:
                        seen_events[end]["attendees"] = [start]
                    else:
                        seen_events[end]["attendees"].append(start)

                if start == "today":
                    lines.append("Today is " + end + ". ")

                if label == "busy":
                    times, day = end.split(", ")
                    lines.append(start + " is busy from " + times + " on " + day + ". ")

                if label == "floor":
                    lines.append(start + " is on floor " + end + " of the building. ")

            # If no name label retrieved, assign the ID as the label
            for event in seen_events:
                if event not in event_names:
                    try:
                        name = self.lookup[event]
                    except KeyError:
                        name = event
                else:
                    name = event_names[event]
                #if "organizing" in seen_events[event]:
                if "attendees" in seen_events[event]:
                    if len(seen_events[event]["attendees"]) == 1:
                        desc1 = seen_events[event]["attendees"][0] + " is attending " + name + ". "
                    else:
                        desc1 = ", ".join(seen_events[event]["attendees"][0:-1]) +  " and " + seen_events[event]["attendees"][-1] + " are attending " + name + ". "
                    lines.append(desc1.strip())
                if "location" in seen_events[event]:
                    desc1 = name +  " will be held in " + seen_events[event]["location"] + ". "
                    lines.append(desc1.strip())

                desc2 = name.capitalize()
                if "start_time" in seen_events[event] and "end_time" in seen_events[event] and "date" in seen_events[event]: 
                    desc2 += " starts at " + seen_events[event]["start_time"] + " and ends at " + seen_events[event]["end_time"] + " on " + seen_events[event]["date"] + ". "
                elif "start_time" in seen_events[event] and "end_time" in seen_events[event]: 
                    desc2 += " starts at " + seen_events[event]["start_time"] + " and ends at " + seen_events[event]["end_time"] + ". "
                elif "start_time" in seen_events[event] and "date" in seen_events[event]: 
                    desc2 += " starts at " + seen_events[event]["start_time"] + " on " + seen_events[event]["date"] + ". "
                elif "end_time" in seen_events[event] and "date" in seen_events[event]: 
                    desc2 += " ends at " + seen_events[event]["end_time"] + " on " + seen_events[event]["date"] + ". "
                elif "start_time" in seen_events[event]: 
                    desc2 += " starts at " + seen_events[event]["start_time"] + ". "
                elif "end_time" in seen_events[event]: 
                    desc2 += " ends at " + seen_events[event]["end_time"] + ". "
                elif "date" in seen_events[event]: 
                    desc2 += " is on " + seen_events[event]["date"] + ". "
                else:
                    desc2 = "" # Add nothing
                
                # KVRET event times
                if "time" in seen_events[event]:
                    desc2 += event + " is at " + seen_events[event]["time"] + ". "

                if "organizer" in seen_events[event]:
                    desc2 += seen_events[event]["organizer"] + " is the organizer of " + event + ". "

                lines.append(desc2.strip())

            for person in seen_people:
                desc = ""
                if "member" in seen_people[person]:
                    if "office" in seen_people[person]:
                        desc = person + " is a member of the " + seen_people[person]["member"] + " group and their office is room " + seen_people[person]["office"] + ". "
                    else:
                        desc = person + " is a member of the " + seen_people[person]["member"] + " group. "
                elif "office" in seen_people[person]:
                    desc = person + "'s office is room " + seen_people[person]["office"] + ". "

                if "email" in seen_people[person] and "phone" in seen_people[person]:
                    desc += person + "\'s phone number is " + seen_people[person]["phone"] + " and their email address is " + seen_people[person]["email"] + ". "
                elif "email" in seen_people[person]:
                    desc += person + "\'s email address is " + seen_people[person]["email"] + ". "
                elif "phone" in seen_people[person]:
                    desc += person + "\'s phone number is " + seen_people[person]["phone"] + ". "
                lines.append(desc.strip())  

            if len(lines) == 0:
                for event in event_names:
                    lines.append(event + " is an event named " + event_names[event])

        else:
            for (start, label, end) in self.edges:
                if entities is not None and (start, label, end) not in entities and (start, label, end) != entities:
                    continue
                lines.append("%s --%s--> %s"%(start, label, end))
        
        return " ".join(lines).replace("  ", " ")
    
    def __repr__(self) -> str:
        return self.verbalise()


#############################################
# RETRIEVAL
#############################################


class BaseRetriever:
    """Basic retrieval to find nodes or edges in the graph that are relevant to the
    provided dialogue history. This is an abstract class"""


    def get_most_relevant_nodes(self, graph:DialogueGraph, dialogue_history:List[str], n:int=10):
        """Retrieves the n most relevant nodes in the graph, given the dialogue history"""

        node_verbalisations = [node for node in graph.nodes]

        best_nodes_idx = self._get_most_relevant(node_verbalisations, dialogue_history, n)
        best_nodes = [list(graph.nodes)[i] for i in best_nodes_idx]
  #      print("best nodes:", best_nodes)
        return best_nodes

    def get_most_relevant_edges(self, graph:DialogueGraph, dialogue_history:List[str], n:int=5):
        """Retrieves the n most relevant edges in the graph, given the dialogue history"""

        edge_verbalisations = [graph.verbalise([edge]) for edge in graph.edges]
        best_edges_idx = self._get_most_relevant(edge_verbalisations, dialogue_history, n)
        best_edges = [list(graph.edges)[i] for i in best_edges_idx]
  #      print("best edges:", best_edges)
        return best_edges

    def _get_most_relevant(self, verbalisations:List[str], dialogue_history:List[str], n:int=5, 
                           min_threshold:float=-10, history_lengths:List[int]=[1,2,3]):
        """Retrieves the most relevant verbalisation given the dialogue history. To boost the
        importance of most recent utterances, we compute the scores for several lengths of the
        dialogue history (specified by history_lengths), and average their results"""

        # We create the various dialogue histories
        dialogue_histories = ["\n".join(dialogue_history[-k:]) for k in history_lengths]
        scores_to_average = []
        for dialogue_history in dialogue_histories:
            scores_to_average.append(self._get_scores(verbalisations, dialogue_history))
        
        # We take the mean of the scores for the various history lengths
        avg_scores = np.mean(scores_to_average, axis=0)

        # And select the indices associated with the highest scores (and >= minimal threshold)
        best_idx = np.argsort(avg_scores)[::-1][:n]
        best_idx = [idx for idx in best_idx if avg_scores[idx]>=min_threshold]    
        return best_idx       

    @abstractmethod
    def _get_scores(self, verbalisations: List[str], query:str):
        """Computes the relevance scores of the verbalisations given the query (dialogue history). 
        Must be implemented in sub-classes."""
        raise NotImplementedError()



class BM25Retriever(BaseRetriever):
    """Retriever that relies on BM25."""

    def __init__(self, stop_words_file="data/stopwords.txt"):
        """Simple BM25 retriever with a stop words list"""

        with open(stop_words_file) as fd:
            self.stopwords = set(line.rstrip("\n") for line in fd)

    def _get_scores(self, verbalisations:List[str], query:str):
        """Runs the BM25 Okapi algorithm to get relevance scores for the verbalisation given the query"""
        
        # We tokenise the verbalisations and the dialogue history
        tokenised_verbalisations = [self.tokenise(verbalisation) for verbalisation in verbalisations]
        tokenised_query = self.tokenise(query)
        corpus = rank_bm25.BM25Okapi(tokenised_verbalisations)
        scores = corpus.get_scores(tokenised_query)
        
  #      print("Scores:", {verbalisations[i]:round(scores[i],4) for i in range(len(verbalisations))})
        return scores
    
    def tokenise(self, sentence):
        """Tokenise the text into words."""

        tokens = []
        for token in sentence.split(" "):
            token = token.lower().strip(".,?!><-")
            if token.endswith("'s"):
                token = token[:-2]
            if token not in self.stopwords:
                tokens.append(token)
        return tokens
    
    
class DenseRetriever(BaseRetriever):
    """Dense retriever based on a sentence-transformer model"""

    def __init__(self, encoder_model="multi-qa-MiniLM-L6-cos-v1", max_cache_size=1000):
        """Initialises a dense retriever based on a sentence transformer model."""

        self.encoder = sentence_transformers.SentenceTransformer(encoder_model, device="cuda")
        
        # We create a cache of mappings between sentences and embeddings
        self.cache = {}
        self.max_cache_size=max_cache_size

    def embed(self, docs: List[str]):
        """Returning the document embeddings for the provided strings, using the cache to speed
        up the embedding operation"""

        docs_with_unknown_embeddings = list(set(v for v in docs if v not in self.cache))
        if len(docs_with_unknown_embeddings) > 0:
   #         print("Computing %i new embeddings"%len(docs_with_unknown_embeddings), end="...", flush=True)
            encoded_vectors = self.encoder.encode(docs_with_unknown_embeddings, device="cuda")
            embeddings_to_add = {verb:embedding for verb, embedding in zip(docs_with_unknown_embeddings, encoded_vectors)}
            self.cache.update(embeddings_to_add)
   #         print("Done")
        
        embeddings = np.array([self.cache[v] for v in docs])
        return embeddings

    def _get_scores(self, verbalisations:List[str], query:str):
        """Retrieves the n most relevant verbalisations given the query (dialogue history), based on 
        cosine similarity scores between the sentence embeddings for the verbalisations and
        the dialogue history.""" 

        # We embed both the verbalisations and the query
        verbalisation_vectors = self.embed(verbalisations)
        query_vector = self.embed([query])[0]

        # We compute the cosine similarity as scores
        similarity_scores = sentence_transformers.util.cos_sim(verbalisation_vectors, query_vector)
        similarity_scores = similarity_scores.cpu().flatten().numpy()
     
        # If the cache size grows too big, we truncate it
        if len(self.cache) > self.max_cache_size:
            self.cache = {k:v for i, (k,v) in enumerate(self.cache.items()) if i <= self.max_cache_size}

        return similarity_scores

def loadGraphWOZ(path, return_responses=False):

    with open(path, "r") as file:
        graphwoz = json.load(file)

    dialogue_graphs = []
    wizard_responses = []
    for dialogue in graphwoz:
        dialogue_data = graphwoz[dialogue]["data"]
        
        event_names = dict()
        edges = list()
        for e in dialogue_data["events"]:
            event_names[e["id"]] = e["name"]
            edges.append(tuple((e["id"],"name",e["name"])))
            edges.append(tuple((e["id"],"start_time",e["start_time"])))
            edges.append(tuple((e["id"],"end_time",e["end_time"])))
            edges.append(tuple((e["id"],"date",e["date"])))
            edges.append(tuple((e["id"],"location",e["location"])))
            edges.append(tuple((e["id"],"organizer",e["organizer"])))
            for a in e["attendees"]:
                edges.append(tuple((a,"attending",e["id"])))
        for p in dialogue_data["people"]:
            edges.append(tuple((p["name"],"office",p["office"])))
            edges.append(tuple((p["name"],"phone",p["phone"])))
            edges.append(tuple((p["name"],"email",p["email"])))
            edges.append(tuple((p["name"],"member",p["group"])))

        utt_idx = 1 #1 indexed because reasons
        resp_idx = 1
        for turn in graphwoz[dialogue]["log"]:
            utterance = turn["alternative"][0]["transcript"]
            utt_trans = tuple(("utt" + str(utt_idx), "transcription", utterance))
            utt_user = tuple(("utt" + str(utt_idx), "speaker", "speaker" + str(graphwoz[dialogue]["speaker"])))

            edges.append(utt_trans)
            edges.append(utt_user)
            utt_idx += 1
            
            #Add to graphs BEFORE agent response
            graph = DialogueGraph(triples=edges, event_names=event_names)
            graph.lookup = event_names
            dialogue_graphs.append(graph) # Add each TURN as a graph (separate graph at each turn)
            
            resp_robot = tuple(("utt" + str(utt_idx), "transcription", turn["agent_response"]))
            resp_user = tuple(("utt" + str(utt_idx), "speaker", "robot"))
            response_link = tuple(("utt" + str(utt_idx), "response_to", "utt" + str(utt_idx)))

            edges.append(resp_robot)
            edges.append(resp_user)
            edges.append(response_link)

            #Increment indices
            utt_idx += 1

            wizard_responses.append(turn["agent_response"])

        if len(graphwoz[dialogue]["log"]) > 0:
            today = str(datetime.fromtimestamp(graphwoz[dialogue]["log"][0]["start_timestamp"])).split(" ")[0]
            edges.append(tuple(("today","is",today)))

    if return_responses:
        return dialogue_graphs, wizard_responses
    else:
        return dialogue_graphs

def loadKVRET(path, return_responses=False):
    with open(path, "r") as file:
        kvret = json.load(file)

    dialogue_graphs = []
    wizard_responses = []
    for index, dialogue in enumerate(kvret):
        dialogue_data = dialogue['scenario']['kb']['items']

        if not dialogue_data or dialogue_data == {} or len(dialogue['scenario']['kb']['items']) == 0:
            continue

        event_names = dict()
        edges = list()
        if dialogue_data:
            for e in dialogue_data:
                if dialogue['scenario']['task']['intent'] == "navigate":
                    # event_names[e["id"]] = e["name"]
                    edges.append(tuple((e["poi"], "name", e["poi"])))
                    edges.append(tuple((e["poi"], "distance", e["distance"])))
                    edges.append(tuple((e["poi"], "type", e["poi_type"])))
                    edges.append(tuple((e["poi"], "traffic", e["traffic_info"])))
                    edges.append(tuple((e["poi"], "address", e["address"])))

                elif dialogue['scenario']['task']['intent'] == "schedule":
                    edges.append(tuple((e["event"], "name", e["event"])))
                    edges.append(tuple((e["event"], "time", e["time"])))
                    edges.append(tuple((e["event"], "date", e["date"])))
                    edges.append(tuple((e["event"], "location", e["room"])))
                    edges.append(tuple((e["event"], "attendee", e["party"])))

                else:
                    edges.append(tuple((e["location"], "name", e["location"])))
                    edges.append(tuple(("today", "is", e["today"])))
                    edges.append(tuple((e["location"], "monday_weather", e["monday"])))
                    edges.append(tuple((e["location"], "tuesday_weather", e["tuesday"])))
                    edges.append(tuple((e["location"], "wednesday_weather", e["wednesday"])))
                    edges.append(tuple((e["location"], "thursday_weather", e["thursday"])))
                    edges.append(tuple((e["location"], "friday_weather", e["friday"])))
                    edges.append(tuple((e["location"], "saturday_weather", e["saturday"])))
                    edges.append(tuple((e["location"], "sunday_weather", e["sunday"])))

        utt_idx = 1  # 1 indexed because reasons
        resp_idx = 1
        for turn in dialogue["dialogue"]:
            if turn['turn'] == 'driver':
                #print(turn)
                utterance = turn["data"]["utterance"]
                utt_trans = tuple(("utt" + str(utt_idx), "transcription", utterance))
                utt_user = tuple(("utt" + str(utt_idx), "speaker", "speaker" + str(index)))

                edges.append(utt_trans)
                edges.append(utt_user)
                utt_idx += 1

                # Add to graphs BEFORE agent response
                graph = DialogueGraph(triples=edges)
                graph.lookup = event_names
                dialogue_graphs.append(graph)  # Add each TURN as a graph (separate graph at each turn)

            last_wizard_response = ''
            if turn['turn'] == 'assistant' and return_responses == True:
                resp_robot = tuple(("utt" + str(utt_idx), "transcription", turn["data"]["utterance"]))
                resp_user = tuple(("utt" + str(utt_idx), "speaker", "robot"))
                response_link = tuple(("utt" + str(utt_idx), "response_to", "utt" + str(utt_idx)))

                last_wizard_response = turn["data"]["utterance"]

                edges.append(resp_robot)
                edges.append(resp_user)
                edges.append(response_link)

                # Increment indices
                utt_idx += 1

            if last_wizard_response != '':
                wizard_responses.append(turn["data"]["utterance"])
            

    if return_responses:
        return dialogue_graphs, wizard_responses
    else:
        return dialogue_graphs

def basic_test():

    chatbot = ResponseGenerator()
    graph = DialogueGraph([("utt1", "transcription", "Hi there"), 
                           ("utt1", "speaker", "user"), 
                           ("utt1", "interlocutor", "robot"),
                           ("utt2", "transcription", "Hello, how can I help you?"), 
                           ("utt2", "speaker", "robot"), 
                           ("utt2", "interlocutor", "user"),
                           ("utt3", "transcription", "When is the status meeting scheduled?"), 
                           ("utt3", "speaker", "user"), 
                           ("utt3", "interlocutor", "robot"), 
                           ("robot", "name", "Pepper"), 
                           ("user", "name", "Pierre"),
                           ("e1", "date", "April 4"),
                           ("e1", "name", "status meeting"),
                           ("John", "participant", "e1"),
                           ("Albert", "participant", "e1"),
                           ("e2", "date", "April 6"),
                           ("e2", "name", "budget meeting"),
                           ("George", "participant", "e2")])
    response = chatbot.get_response(graph)
    print("Response:", response)
    


def run_graphwoz(input_file="data/graphwoz_dev.json",
                 output_file="output/graphwoz_dev_2hop_results.json", generate_responses=False,
                 do_bm25=True, do_dense=True, do_baseline=True, do_logic=True):

    chatbot = ResponseGenerator()
    sparse_retriever = BM25Retriever()
    dense_retriever = DenseRetriever()

    graphs, responses = loadGraphWOZ(input_file, return_responses=True)

    #graphs, responses = loadKVRET("data/kvret_dataset_public/kvret_dev_public.json", return_responses=True)
  
    #graphs.extend(k_graphs)
    #responses.extend(k_responses)

  #  for i, (graph, response) in enumerate(zip(graphs, responses)):
  #      for edge in graph.edges:
  #          print(graph.verbalise(edge))

    outputs = []
    for i, (graph, response) in tqdm.tqdm(enumerate(zip(graphs, responses))):

        print("Turn %i:"%(i+1))

        # BM25 retriever variants
        if do_bm25:
            chatbot.retriever = sparse_retriever
            output = {"prompt_bm25_non_enriched":chatbot.create_prompt(graph,enrich=False)}
            print("Prompt (using BM25):", output["prompt_bm25_non_enriched"])

            if generate_responses:
                output["response_bm25_non_enriched"] = chatbot.decode_response(output["prompt_bm25_non_enriched"])
                print("Response (from BM25 prompt):", output["response_bm25_non_enriched"])

            if do_logic:
                output["prompt_bm25_enriched"] = chatbot.create_prompt(graph,enrich=True)
                print("Prompt (using BM25):", output["prompt_bm25_enriched"])

                if generate_responses:
                    output["response_bm25_enriched"] = chatbot.decode_response(output["prompt_bm25_enriched"])
                    print("Response (from BM25 prompt):", output["response_bm25_enriched"])

        # Dense retriever variants
        if do_dense:
            chatbot.retriever = dense_retriever
            output["prompt_dense_non_enriched"] = chatbot.create_prompt(graph,enrich=False)
            print("Prompt (using dense retriever):", output["prompt_dense_non_enriched"])

            if generate_responses:
                output["response_dense_non_enriched"] = chatbot.decode_response(output["prompt_dense_non_enriched"])
                print("Response (from dense retriever prompt):", output["response_dense_non_enriched"])


            if do_logic:
                output["prompt_dense_enriched"] = chatbot.create_prompt(graph,enrich=True)
                print("Prompt (using dense retriever):", output["prompt_dense_enriched"])

                if generate_responses:
                    output["response_dense_enriched"] = chatbot.decode_response(output["prompt_dense_enriched"])
                    print("Response (from dense retriever prompt):", output["response_dense_enriched"])

        # Baseline (non-subgraph) variants
        if do_baseline:
            output["prompt_baseline_non_enriched"] = chatbot.create_prompt(graph, no_subgraph=True, enrich=False)
            print("Prompt (using direct triple baseline retriever):", output["prompt_baseline_non_enriched"])

            if generate_responses:
                output["response_baseline_non_enriched"] = chatbot.decode_response(output["prompt_baseline_non_enriched"])
                print("Response (from direct triple baseline retriever prompt):", output["response_baseline_non_enriched"])

            if do_logic:
                output["prompt_baseline_enriched"] = chatbot.create_prompt(graph, no_subgraph=True, enrich=True)
                #print("Prompt (using direct triple baseline retriever):", output["prompt_baseline_enriched"])
                if generate_responses:
                    output["response_baseline_enriched"] = chatbot.decode_response(output["prompt_baseline_enriched"])
                    print("Response (from direct triple baseline retriever prompt):", output["response_baseline_enriched"])

            # Raw baseline
            output["prompt_baseline_non_verbalize_non_enrich"] = chatbot.create_prompt(graph, no_subgraph=True, enrich=False, verbalise=False)
            print("Prompt (using direct triple baseline retriever):", output["prompt_baseline_non_verbalize_non_enrich"])
            if generate_responses:
                output["response_baseline_non_verbalize_non_enrich"] = chatbot.decode_response(output["prompt_baseline_non_verbalize_non_enrich"])
                print("Response (from direct triple baseline retriever prompt):", output["response_baseline_non_verbalize_non_enrich"])


        output["wizard_response"] = response
        print("Wizard response:", output["wizard_response"])
        print("-----")

        outputs.append(output)

    with open(output_file, "w") as fd:
        json.dump(outputs, fd, indent=4)

# Super-basic test
if __name__ == "__main__":
   run_graphwoz(do_baseline=True, generate_responses=False)
