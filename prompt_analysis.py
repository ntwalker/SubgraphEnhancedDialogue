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
import networkx as nx
import rank_bm25
import evaluate
import ast
import pandas as pd

evaluate.load("meteor")
metrics = evaluate.combine(["bleu", "meteor"])

ref_labels = ['triples_bm25_non_enriched_subgraph', 'triples_bm25_enriched_subgraph', 'triples_dense_non_enriched_subgraph', 'triples_dense_enriched_subgraph', 'triples_bm25_non_enriched_2hop', 
        'triples_bm25_enriched_2hop', 'triples_dense_non_enriched_2hop', 'triples_dense_enriched_2hop', 'triples_baseline_non_enriched', 'triples_baseline_enriched', 'triples_baseline_non_verbalize_non_enrich']

ref_prize_labels = [lab.replace("triples", "prizes_dense_full") for lab in ref_labels]


df_node_diffs_avg = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_edge_diffs_avg = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_prize_diffs_avg = pd.DataFrame(index=ref_prize_labels, columns=ref_prize_labels)

df_node_diffs_med = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_edge_diffs_med = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_prize_diffs_med = pd.DataFrame(index=ref_prize_labels, columns=ref_prize_labels)

df_node_diffs_var = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_edge_diffs_var = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_prize_diffs_var = pd.DataFrame(index=ref_prize_labels, columns=ref_prize_labels)

df_node_diffs_std = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_edge_diffs_std = pd.DataFrame(index=ref_labels, columns=ref_labels)
df_prize_diffs_std = pd.DataFrame(index=ref_prize_labels, columns=ref_prize_labels)

def run_graphwoz(input_file="data/graphwoz_dev.json",
                 output_file="output/graphwoz_kvret_train_prompts.json", generate_responses=False,
                 do_bm25=True, do_dense=True, do_baseline=True):

    
    with open("output/graphwoz_dev_prompts.json", "r") as file:
        turns = json.load(file)

    #print(turns[0].keys())

    prompt_texts = dict()
    prompt_lengths = dict()
    prize_lists = dict()
    prize_per_node = dict()
    prize_sums = dict()
    graphs = dict()
    for element in turns:

        triple_labels = [key for key in element.keys() if "triple" in key]
        prompt_labels = [key for key in element.keys() if "prompt" in key]
        prize_labels = [key for key in element.keys() if "prizes_bm25_resp" in key]

        for lab in prompt_labels:
            if lab not in prompt_lengths:
                prompt_lengths[lab] = [len(element[lab].split(" "))]
                prompt_texts[lab] = [element[lab]]
            else:
                prompt_lengths[lab].append(len(element[lab].split(" ")))
                prompt_texts[lab].append(element[lab])

        for label in triple_labels:
            triples = [tuple(e) for e in element[label]]
            if not label in graphs:
                graphs[label] = [triples]
            else:
                graphs[label].append(triples)
                
        for label in prize_labels:
            p = element[label]
            if not label in prize_lists:
                prize_lists[label] = [p]
                prize_per_node[label] = [sum(p) / len(p)]
                prize_sums[label] = [sum(p)]
            else:
                prize_lists[label].append(p)
                prize_per_node[label].append(sum(p) / len(p))
                prize_sums[label].append(sum(p))

    #for key, value in prompt_lengths.items():
    #    print(key, " average length: ", np.average(value))
    #    print(key, " length variance: ", np.var(value))
    #    print(key, " length standard deviation: ", np.std(value))
    #print("\n")


    node_counts = dict()
    edge_counts = dict()

    node_diffs = dict()
    edge_diffs = dict()

    prize_diffs = dict()

    for key, value in graphs.items():
        for comparekey in graphs.keys():
            for i, subgraph_edges in enumerate(value):
                subgraph_nodes = set([start for (start, label, end) in subgraph_edges])
                subgraph_nodes.update([end for (start, label, end) in subgraph_edges])

                nonverbal_edges = graphs[comparekey][i] # Get the baseline triples of the same turn
                nonverbal_nodes = set([start for (start, label, end) in nonverbal_edges]) 
                nonverbal_nodes.update([end for (start, label, end) in nonverbal_edges])

                if key not in node_diffs:
                    node_diffs[key] = dict()
                    node_diffs[key][comparekey] = [len(subgraph_nodes - nonverbal_nodes)]
                elif comparekey not in node_diffs[key]:
                    node_diffs[key][comparekey] = [len(subgraph_nodes - nonverbal_nodes)]
                else:
                    node_diffs[key][comparekey].append(len(subgraph_nodes - nonverbal_nodes))

                if key not in edge_diffs:
                    edge_diffs[key] = dict()
                    edge_diffs[key][comparekey] = [len(set(subgraph_edges) - set(nonverbal_edges))]
                elif comparekey not in edge_diffs[key]:
                    edge_diffs[key][comparekey] = [len(set(subgraph_edges) - set(nonverbal_edges))]
                else:
                    edge_diffs[key][comparekey].append(len(set(subgraph_edges) - set(nonverbal_edges)))

                #if "subgraph" in key and "2hop" in comparekey:
                #    print(key, comparekey)
                #    print(len(subgraph_edges), len(nonverbal_edges))
                #    time.sleep(1)

        for edge_set in value:
            if key not in node_counts:
                subgraph_nodes = set([start for (start, label, end) in subgraph_edges])
                subgraph_nodes.update([end for (start, label, end) in subgraph_edges])
                node_counts[key] = [len(subgraph_nodes)]
                edge_counts[key] = [len(edge_set)]

    for key, value in prize_lists.items():
        for comparekey in prize_lists.keys():
            for i, subgraph_prizes in enumerate(value):
                subgraph_prize_set = set([p for p in subgraph_prizes])

                compare_prizes = prize_lists[comparekey][i] # Get the baseline triples of the same turn
                compare_prize_set = set([p for p in compare_prizes])

                if key not in prize_diffs:
                    prize_diffs[key] = dict()
                    prize_diffs[key][comparekey] = [len(set(subgraph_prize_set) - set(compare_prize_set))]
                elif comparekey not in prize_diffs[key]:
                    prize_diffs[key][comparekey] = [len(set(subgraph_prize_set) - set(compare_prize_set))]
                else:
                    prize_diffs[key][comparekey].append(len(set(subgraph_prize_set) - set(compare_prize_set)))


    for key, value in node_counts.items():
        print(key + " avg # nodes: ", np.average(value))
        #print(key + " # nodes variance: ", np.var(value))
        #print(key + " # nodes STD: ", np.std(value))
    for key, value in edge_counts.items():
        print(key + " avg # edges: ", np.average(value))    
        #print(key + " # edges variance: ", np.var(value))
        #print(key + " # edges STD: ", np.std(value))
    
    print("\n")

    #print([thing.replace("prizes_", "") for thing in df_prize_diffs_avg.columns])
    #print(df_prize_diffs_avg.index)
    
    #df_prize_diffs_avg.rename(columns={label:label.replace("prizes_", "") for label in df_prize_diffs_avg.columns}, index={label:label.replace("prizes_", "") for label in df_prize_diffs_avg.index})
    #df_prize_diffs_var.rename(columns={label:label.replace("prizes_", "") for label in df_prize_diffs_var.columns}, index={label:label.replace("prizes_", "") for label in df_prize_diffs_var.index})

    #df_prize_diffs_var.columns = df_prize_diffs_var.columns.str.replace("prizes_", "", regex=True)
    #df_prize_diffs_std.columns = df_prize_diffs_std.columns.str.replace("prizes_", "", regex=True)
    
    #df_node_diffs_avg.columns = df_node_diffs_avg.columns.str.replace("triples_", "", regex=True)
    #df_node_diffs_var.columns = df_node_diffs_var.columns.str.replace("triples_", "", regex=True)
    #df_node_diffs_std.columns = df_node_diffs_std.columns.str.replace("triples_", "", regex=True)

    #df_edge_diffs_avg.columns = df_edge_diffs_avg.columns.str.replace("triples_", "", regex=True)
    #df_edge_diffs_var.columns = df_edge_diffs_var.columns.str.replace("triples_", "", regex=True)
    #df_edge_diffs_std.columns = df_edge_diffs_std.columns.str.replace("triples_", "", regex=True)

    for key, comparison in prize_diffs.items():
        for comparekey, value in comparison.items():
            df_prize_diffs_avg.at[key, comparekey] = np.average(value)
            df_prize_diffs_med.at[key, comparekey] = np.median(value)
            df_prize_diffs_var.at[key, comparekey] = np.var(value)
            df_prize_diffs_std.at[key, comparekey] = np.std(value)
    #print("Average Prize Diff ",df_prize_diffs_avg)
    #print("Median Prize Diff ",df_prize_diffs_med)
    #print("Var Prize Diff ",df_prize_diffs_var)
    #print("STD Prize Diff ",df_prize_diffs_std)
    #print("\n")

    print("\n")
    for key, value in prize_per_node.items():
        print(key + " avg reward PER NODE: ", np.average(value))
        #print(key + " median reward PER NODE: ", np.median(value))
        #print(key + " reward variance PER NODE: ", np.var(value))
        #print(key + " reward PER NODE standard deviation: ", np.std(value))
    print("\n")
    for key, value in prize_sums.items():
        print(key + " avg reward SUM: ", np.average(value))
        #print(key + " median reward SUM: ", np.median(value))
        #print(key + " reward SUM variance: ", np.var(value))
        #print(key + " reward SUM standard deviation: ", np.std(value))
    print("\n")

    for key, comparison in node_diffs.items():
        for comparekey, value in comparison.items():
            df_node_diffs_avg.at[key, comparekey] = np.average(value)
            df_node_diffs_med.at[key, comparekey] = np.median(value)
            df_node_diffs_var.at[key, comparekey] = np.var(value)
            df_node_diffs_std.at[key, comparekey] = np.std(value)
    #print("Average Node Diff ",df_node_diffs_avg)
    #print("Median Node Diff ",df_node_diffs_med)
    #print("Var Node Diff ",df_node_diffs_var)
    #print("STD Node Diff ",df_node_diffs_std)
    #print("\n")
    
    #time.sleep(1000)

    print("=======\nEdge differences\n======")
    for key, comparison in edge_diffs.items():
        for comparekey, value in comparison.items():
            df_edge_diffs_avg.at[key, comparekey] = np.average(value)
            df_edge_diffs_med.at[key, comparekey] = np.median(value)
            df_edge_diffs_var.at[key, comparekey] = np.var(value)
            df_edge_diffs_std.at[key, comparekey] = np.std(value)
    print("Average Edge Diff ",df_edge_diffs_avg)
    print("Median Edge Diff ",df_edge_diffs_med)
    print("Var Edge Diff ",df_edge_diffs_var)
    print("STD Edge Diff ",df_edge_diffs_std)
    print("\n")

    print("\nComputing metrics...")

    for key, value in prompt_texts.items():
        if not "baseline" in key:
            results = metrics.compute(references=prompt_texts["prompt_baseline_non_enriched"], predictions=value)
            print(key + " vs Non-Enriched Baseline: ", results)

            results = metrics.compute(references=prompt_texts["prompt_baseline_non_verbalize_non_enrich"], predictions=value)
            print(key + " vs Non-Enriched Non-Verbalized Baseline:", results)
            print("\n")
    
    #output = {"node_differences": node_diffs}



# Super-basic test
if __name__ == "__main__":
   run_graphwoz(do_baseline=True, generate_responses=False)
