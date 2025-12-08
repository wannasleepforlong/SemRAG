import os
import json
import numpy as np
import networkx as nx
import spacy
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from community import community_louvain
from langchain_community.chat_models import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI

import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

class SemanticChunker:
    def __init__(self, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("sentencizer", before="parser")

    def _split_into_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents if sent.text.strip()]

    def _buffer_merge(self, sentences: List[str], buffer_size: int) -> List[str]:
        if buffer_size <= 0:
            return sentences

        merged_sentences = []
        for i, sentence in enumerate(sentences):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + buffer_size + 1)
            merged_content = " ".join(sentences[start:end])
            merged_sentences.append(merged_content)
        return merged_sentences

    def create_semantic_chunks(self, text: str, threshold: float = 0.8, buffer_size: int = 3) -> List[str]:
        sentences = self._split_into_sentences(text)
        merged_sents = self._buffer_merge(sentences, buffer_size)

        embeddings = self.model.encode(merged_sents)

        distances = []
        for i in range(len(embeddings) - 1):
            dist = 1 - cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distances.append(dist)

        chunks = []
        current_chunk_sentences = []

        for i, dist in enumerate(distances):
            current_chunk_sentences.append(sentences[i])
            if dist > threshold:
                chunks.append(" ".join(current_chunk_sentences).strip())
                current_chunk_sentences = []

        if sentences:
            current_chunk_sentences.append(sentences[-1])
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences).strip())

        logging.info(f"Generated {len(chunks)} semantic chunks.")
        return [c for c in chunks if c]


class SemRAG:
    def __init__(self, llm_model="tinyllama", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    #     self.llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",   # or gemini-2.0-pro
    #     temperature=0.2,
    #     google_api_key=""
    # )

        self.llm = ChatOllama(model=llm_model)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.chunker = SemanticChunker(embedding_model_name=embedding_model)
        self.nlp = spacy.load("en_core_web_sm")
        self.kg = nx.MultiDiGraph()

        self.vectorstore: Chroma = None
        self.bm25_index: BM25Okapi = None
        self.chunks: List[str] = []

    # ---- Triplet Extraction ----
    def _extract_triplets_local(self, chunk: str) -> List[Dict]:
        doc = self.nlp(chunk)
        triplets = []
        for sent in doc.sents:
            subject = ""
            verb = ""
            obj = ""

            for token in sent:
                if "nsubj" in token.dep_:
                    subject = token.text.strip()
                elif token.pos_ == "VERB":
                    verb = token.lemma_.strip()
                    for child in token.children:
                        if "obj" in child.dep_ or "attr" in child.dep_:
                            obj = child.text.strip()
                            break
                    if subject and verb and obj and subject != obj:
                        triplets.append({"head": subject, "tail": obj, "type": verb, "source": chunk})
                        subject, verb, obj = "", "", ""
        return triplets

    # ---- Algorithm 2 ----
    def algorithm_2_extract_entities_relations(self, chunks: List[str]) -> Dict[str, Any]:
        logging.info("Starting Algorithm 2: Entity/Relation Extraction")
        entities = Counter()
        relations = []

        for chunk in chunks:
            doc = self.nlp(chunk)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    entities[ent.text.strip()] += 1

            relations.extend(self._extract_triplets_local(chunk))

        return {"entities": entities, "relations": relations}

    # ---- Algorithm 3 ----
    def algorithm_3_build_knowledge_graph(self, entity_rel_data: Dict) -> nx.MultiDiGraph:
        logging.info("Building KG...")
        self.kg.clear()

        for entity, count in entity_rel_data["entities"].items():
            if count >= 2:
                self.kg.add_node(entity, type="entity", freq=count)

        for rel in entity_rel_data["relations"]:
            head, tail = rel.get("head"), rel.get("tail")
            rel_type = rel.get("type", "RELATED")
            source_chunk = rel.get("source")
            if head in self.kg and tail in self.kg:
                self.kg.add_edge(head, tail, relation=rel_type, source_chunk=source_chunk)

        if self.kg.number_of_nodes() > 0:
            partition = community_louvain.best_partition(self.kg.to_undirected())
            nx.set_node_attributes(self.kg, partition, 'community_id')

        logging.info(f"KG: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges.")
        return self.kg

    # -------- SAVE / LOAD KG --------
    def save_kg(self, path="kg_graph.json"):
        data = nx.node_link_data(self.kg)
        with open(path, "w") as f:
            json.dump(data, f)

    def load_kg(self, path="kg_graph.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self.kg = nx.node_link_graph(data)

    # ------------------ CHECK BUILD STATUS ------------------
    def _is_pipeline_built(self, vectordb_dir: str) -> bool:
        chunks_ok = os.path.exists("chunks.json")
        vectordb_ok = os.path.exists(vectordb_dir) and len(os.listdir(vectordb_dir)) > 0
        kg_ok = os.path.exists("kg_graph.json")
        return chunks_ok and vectordb_ok and kg_ok

    # ------------------ BUILD FULL SEMRAG ------------------
    def build_complete_semrag(self, pdf_path: str, vectordb_dir: str):

        if self._is_pipeline_built(vectordb_dir):
            logging.info("Pipeline already exists → Loading components...")
            self.chunks = json.load(open("chunks.json"))
            self.vectorstore = Chroma(persist_directory=vectordb_dir, embedding_function=self.embeddings)
            self.bm25_index = BM25Okapi([c.lower().split() for c in self.chunks])
            self.load_kg("kg_graph.json")
            return

        # ---- Build from scratch ----
        logging.info("Building SemRAG pipeline for the first time...")

        text = load_pdf(pdf_path)
        self.chunks = self.chunker.create_semantic_chunks(text)
        json.dump(self.chunks, open("chunks.json", "w"))

        documents = [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(self.chunks)]
        self.vectorstore = Chroma.from_documents(documents, self.embeddings, persist_directory=vectordb_dir)

        self.bm25_index = BM25Okapi([c.lower().split() for c in self.chunks])

        entity_rel_data = self.algorithm_2_extract_entities_relations(self.chunks)
        self.algorithm_3_build_knowledge_graph(entity_rel_data)
        self.save_kg("kg_graph.json")

        logging.info("SemRAG fully built and cached.")

    # ------------------ RETRIEVAL + LLM ------------------
    def _hybrid_search(self, query: str, k: int = 2):
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
        semantic_docs = [doc for doc, _ in vector_results]

        tokenized_query = query.lower().split()
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:k]
        keyword_docs = [Document(page_content=self.chunks[i]) for i in top_indices]

        combined_docs = {d.page_content: d for d in (semantic_docs + keyword_docs)}.values()
        return list(combined_docs)

    def _graph_search(self, query: str):
        query_entities = [ent.text for ent in self.nlp(query).ents]
        graph_docs = []

        for entity in query_entities:
            if entity in self.kg:
                for h, t, d in self.kg.out_edges(entity, data=True):
                    graph_docs.append(
                        Document(page_content=f"{h} {d.get('relation')} {t}. Context: {d.get('source_chunk')}")
                    )

                community_id = self.kg.nodes[entity].get('community_id')
                if community_id:
                    nodes = [n for n, data in self.kg.nodes(data=True) if data.get('community_id') == community_id]
                    graph_docs.append(Document(page_content=f"Community {community_id} nodes: {nodes[:10]}"))

        return graph_docs

    def _custom_reranking(self, results: List[Document], query: str):
        query_emb = self.embeddings.embed_query(query)
        scored = []

        for doc in results:
            doc_emb = self.embeddings.embed_query(doc.page_content)
            semantic_score = cosine_similarity([query_emb], [doc_emb])[0][0]
            keyword_score = sum(1 for w in query.lower().split() if w in doc.page_content.lower())
            final = semantic_score * 0.7 + keyword_score * 0.3
            scored.append((doc, final))

        return [d for d, _ in sorted(scored, key=lambda x: x[1], reverse=True)]

    # ------------------ QUERY ------------------
    def query(self, question: str) -> str:
        hybrid = self._hybrid_search(question)
        graph = self._graph_search(question)

        docs = self._custom_reranking(hybrid + graph, question)[:5]

        #   Comment ou t the following lines if you don't want to see the selected chunks
        # print("\n================ SELECTED CHUNKS FOR ANSWER ================\n")
        # for i, d in enumerate(docs):
        #     print(f"\n--- Chunk {i} ---\n{d.page_content}\n")
        #     print("============================================================\n")


        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = PromptTemplate.from_template("""
You are a retrieval-augmented answering system. 
Your job is to answer ONLY using the provided context.  
Do NOT use prior knowledge, assumptions, or external facts.


INSTRUCTIONS:
• If the answer is found in the context, give a clear and direct response.
• If the context partially answers the question, answer only the supported part and say what is missing.
• If the answer is NOT in the context, reply:  
  "The context does not provide enough information to answer this question."

• Do NOT add new facts.
• Do NOT guess.

===================== CONTEXT START =====================
{context}
===================== CONTEXT END =======================

QUESTION:
{question}

FINAL ANSWER:
""")
        chain = prompt | self.llm
        try:
            response = chain.invoke({"context": context, "question": question})
            return response.content
        except Exception as e:
            return f"LLM Error: {e}"


if __name__ == "__main__":
    PDF_FILE_PATH = "Ambedkar_book.pdf"
    VECTORDB_DIRECTORY = "./semrag_db"
    semrag = SemRAG()
    semrag.build_complete_semrag(PDF_FILE_PATH, VECTORDB_DIRECTORY)
    # q = "What were Dr. Ambedkar's ideal society based upon?"
    # print(semrag.query(q))
    while True:
        q = input("User: ")
        print("SemRAG: ", semrag.query(q))