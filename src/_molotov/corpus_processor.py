from typing import Callable, List, Optional, Union
from uuid import uuid4

try:
    from llama_index import Document
    from llama_index.text_splitter import SentenceSplitter
except ImportError:
    from llama_index.core import Document
    from llama_index.core.text_splitter import SentenceSplitter


def llama_index_sentence_splitter(
    documents: list[str], document_ids: list[str], chunk_size=256
):
    chunk_overlap = min(chunk_size / 4, min(chunk_size / 2, 64))
    chunks = []
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = [[Document(text=doc)] for doc in documents]
    for doc_id, doc in zip(document_ids, docs):
        chunks += [
            {"document_id": doc_id, "content": node.text} for node in node_parser(doc)
        ]
    return chunks

class CorpusProcessor:
    def __init__(
        self,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        self.document_splitter_fn = document_splitter_fn
        self.preprocessing_fn = preprocessing_fn

    def process_corpus(
        self,
        documents: list[str],
        document_ids: Optional[list[str]] = None,
        **splitter_kwargs,
    ) -> List[dict]:
        # TODO CHECK KWARGS
        document_ids = (
            [str(uuid4()) for _ in range(len(documents))]
            if document_ids is None
            else document_ids
        )
        if self.document_splitter_fn is not None:
            documents = self.document_splitter_fn(
                documents, document_ids, **splitter_kwargs
            )
        if self.preprocessing_fn is not None:
            if isinstance(self.preprocessing_fn, list):
                for fn in self.preprocessing_fn:
                    documents = fn(documents, document_ids)
                return documents
            return self.preprocessing_fn(documents, document_ids)
        return documents
