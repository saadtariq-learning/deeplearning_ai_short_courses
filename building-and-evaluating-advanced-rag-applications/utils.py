
import os
from dotenv import load_dotenv, find_dotenv

import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI
)

from trulens_eval.feedback import GroundTruthAgreement
import nest_asyncio

nest_asyncio.apply()
load_dotenv(find_dotenv())

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")

openai = OpenAI()

qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

golden_set = [
    {"query": "who invented the lightbulb?", "expected_response": "Thomas Edison"},
    {"query": "¿quien invento la bombilla?", "expected_response": "Thomas Edison"}
]

f_groundtruth = Feedback(GroundTruthAgreement(golden_set).agreement_measure, name = "Ground Truth").on_input_output()

feedbacks = [qa_relevance, qs_relevance, f_groundtruth]

def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder

def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
        )
    return tru_recorder

from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os


def build_sentence_window_index(
    document,
    llm,
    embed_model_name="BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
):
    # Create the sentence window node parser
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # Instantiate the embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    # Load or create the index
    if not os.path.exists(save_dir):
        index = VectorStoreIndex.from_documents([document])
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from llama_index.core.node_parser import HierarchicalNodeParser

from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model_name="BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]

    #Create node parser
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

    #Create embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    #Set the global LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    #Convert documents into nodes
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    #Prepare storage
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    #Load or build index
    if not os.path.exists(save_dir):
        index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        index.storage_context.persist(persist_dir=save_dir)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir)
        )

    return index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine
