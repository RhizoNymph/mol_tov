import os
import math

from dotenv import load_dotenv
from functools import lru_cache
from flask import Flask, request, jsonify
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

from pretrained_model import RAGPretrainedModel

load_dotenv()

INDEX_ROOT = os.getenv("INDEX_ROOT")

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", index_root=INDEX_ROOT)

app = Flask(__name__)

counter = {"api" : 0}

indices = {}

@app.route("/api/delete_index/<index_name>", methods=["GET"])
def delete_index(index_name):
    if index_name in indices:
        del indices[index_name]
        return {"message": "Index deleted successfully"}, 200
    else:
        return {"error": "Index not found"}, 404

@app.route("/api/list_indices", methods=["GET"])
def list_indices():
    return {"indices": list(indices.keys())}

@app.route("/api/add_document/<index_name>/<collections>", methods=["POST"])
def add_document(index_name, collections):
    data = request.json
    document = data.get("document")
    document_id = data.get("document_id")
    if not document_id:
        return jsonify({"error": "Missing document_id"}), 400

    document_metadata = data.get("document_metadata", {})
    
    if not index_name or not document or not collections:
        return jsonify({"error": "Missing index_name, collections, or document"}), 400
    
    # Create index if it does not exist
    if index_name not in indices:
        RAG.index(
            [document],
            index_name=index_name,
            document_metadatas=[document_metadata]
        )
        indices[index_name] = Searcher(index=index_name, index_root=index_root)
    else:
        RAG.add_to_index(
            [document],
            index_name=index_name,
            document_metadatas=[document_metadata]
        )
        # TODO: Figure out if I need to reinstantiate this
        indices[index_name] = Searcher(index=index_name, index_root=index_root)
    
    return jsonify({"message": "Document added successfully"}), 200

@lru_cache(maxsize=1000000)
def api_search_query(query, k, searcher):
    print(f"Query={query}")
    if k == None: k = 10
    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]            
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}

@app.route("/api/search/<index_name>", methods=["GET"])
def api_search():
    index_name = request.args.get("index_name")
    if index_name not in indices:
        return {"error": "Index not found"}, 404
    searcher = indices[index_name]
    return api_search_query(request.args.get("query"), request.args.get("k"), searcher)

if __name__ == "__main__":
    app.run("0.0.0.0", 5000, debug=True)
