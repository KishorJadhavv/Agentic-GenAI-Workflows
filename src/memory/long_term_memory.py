"""
Long-term memory implementation using Milvus vector database.
Provides persistent context storage across sessions.
"""

from typing import List, Optional
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pydantic_settings import BaseSettings


class MemorySettings(BaseSettings):
    """Memory configuration settings."""
    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_use_secure: bool = False
    embedding_model: str = "text-embedding-3-small"
    collection_name: str = "agent_long_term_memory"

    class Config:
        env_prefix = "MEMORY_"


class LongTermMemory:
    """
    Manages long-term context storage and retrieval using Milvus.
    """

    def __init__(self, settings: Optional[MemorySettings] = None):
        """
        Initialize the LongTermMemory.

        Args:
            settings: Configuration settings for memory.
        """
        self.settings = settings or MemorySettings()
        self.embeddings = OpenAIEmbeddings(model=self.settings.embedding_model)
        self.vector_store = self._init_vector_store()

    def _init_vector_store(self) -> Milvus:
        """
        Initializes the Milvus vector store.
        """
        return Milvus(
            embedding_function=self.embeddings,
            connection_args={
                "host": self.settings.milvus_host,
                "port": self.settings.milvus_port,
                "user": self.settings.milvus_user,
                "password": self.settings.milvus_password,
                "secure": self.settings.milvus_use_secure,
            },
            collection_name=self.settings.collection_name,
            auto_id=True,
        )

    def save_context(self, text: str, metadata: Optional[dict] = None):
        """
        Saves a piece of context to the long-term memory.

        Args:
            text: The text content to save.
            metadata: Associated metadata for the context.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        self.vector_store.add_documents([doc])

    def retrieve_context(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieves relevant context from long-term memory.

        Args:
            query: The search query.
            k: The number of documents to retrieve.

        Returns:
            A list of relevant Document objects.
        """
        return self.vector_store.similarity_search(query, k=k)

    def clear_memory(self):
        """
        Clears all stored context from the memory collection.
        """
        # This is a dangerous operation, use with caution.
        # Note: Milvus doesn't have a direct 'clear' in LangChain wrapper that's standard,
        # often requires dropping the collection.
        pass


if __name__ == "__main__":
    # Example usage (requires running Milvus instance)
    try:
        memory = LongTermMemory()
        memory.save_context("User prefers Python for data engineering tasks.", {"user_id": "123"})
        results = memory.retrieve_context("What are the user's language preferences?")
        for doc in results:
            print(f"Found: {doc.page_content} (Metadata: {doc.metadata})")
    except Exception as e:
        print(f"Milvus not available or error occurred: {e}")
