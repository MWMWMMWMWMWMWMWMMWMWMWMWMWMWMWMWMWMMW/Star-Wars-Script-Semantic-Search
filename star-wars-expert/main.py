import sys
print(sys.executable)

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
import random  # for shuffling results

PERSIST_PATH = "./qdrant_db"
COLLECTION_NAME = "star-wars-scripts"

# local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_star_wars_script(url, movie_title):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    script_raw = soup.find("pre").get_text()
    return Document(page_content=script_raw, metadata={"title": movie_title})


def main():
    client = QdrantClient(path=PERSIST_PATH)

    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        vectorstore = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            client=client,
        )
    except Exception:
        client.close()
        star_wars_scripts = [
            {"title": "Star Wars: A New Hope", "url": "https://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html"},
            {"title": "Star Wars: The Empire Strikes Back", "url": "https://www.imsdb.com/scripts/Star-Wars-The-Empire-Strikes-Back.html"},
            {"title": "Star Wars: Return of the Jedi", "url": "https://www.imsdb.com/scripts/Star-Wars-Return-of-the-Jedi.html"},
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,   # smaller chunks for better retrieval
            chunk_overlap=200,
        )

        all_chunks = []
        for script in star_wars_scripts:
            doc = load_star_wars_script(script["url"], script["title"])
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
            print(f"Loaded and split script for {script['title']} ({len(chunks)} chunks)")

        vectorstore = QdrantVectorStore.from_documents(
            all_chunks,
            embedding=embeddings,
            path=PERSIST_PATH,
            collection_name=COLLECTION_NAME,
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # fetch top 10 for variety

    print("\n--- Star Wars Expert Ready (retriever only) ---")

    seen_snippets = set()  # track duplicates globally

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        results = retriever.get_relevant_documents(query)
        if not results:
            print("\nAssistant: There is no information about this in the original Star Wars scripts.")
            continue

        # shuffle results to mix scripts
        random.shuffle(results)

        # print top 5 unique chunks
        printed = 0
        for doc in results:
            snippet = doc.page_content[:500]  # first 500 chars to deduplicate
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)

            # print a readable excerpt (first 1000 chars)
            print(f"\n--- Excerpt from {doc.metadata.get('title', 'Unknown')} ---\n{doc.page_content[:1000].strip()}...\n")
            printed += 1
            if printed >= 5:
                break

        if printed == 0:
            print("\nAssistant: There is no new information in the scripts for this query.")


if __name__ == "__main__":
    main()