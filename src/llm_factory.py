import os
import json
import hashlib
from typing import Optional, Any, List
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class CachedEmbeddingsWrapper(Embeddings):
    """
    Wraps any underlying Embeddings provider (Ollama, OpenAI, AWS Bedrock, Cohere)
    with a persistent disk cache for warm-starting embeddings.
    Avoids re-computing embedding vectors for texts already processed.
    """

    def __init__(self, underlying_embeddings: Embeddings, cache_dir: Optional[str] = None):
        self.underlying = underlying_embeddings
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(base_dir, "data", "embedding_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._memory_cache = {}

    def _get_cache_path(self, text: str) -> str:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.json")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, text in enumerate(texts):
            path = self._get_cache_path(text)
            if text in self._memory_cache:
                results[idx] = self._memory_cache[text]
            elif os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        vec = json.load(f)
                        self._memory_cache[text] = vec
                        results[idx] = vec
                except Exception:
                    missing_indices.append(idx)
                    missing_texts.append(text)
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        if missing_texts:
            new_vectors = self.underlying.embed_documents(missing_texts)
            for idx, text, vec in zip(missing_indices, missing_texts, new_vectors):
                results[idx] = vec
                self._memory_cache[text] = vec
                path = self._get_cache_path(text)
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(vec, f)
                except Exception:
                    pass

        return results

    def embed_query(self, text: str) -> List[float]:
        path = self._get_cache_path(text)
        if text in self._memory_cache:
            return self._memory_cache[text]
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    vec = json.load(f)
                    self._memory_cache[text] = vec
                    return vec
            except Exception:
                pass

        vec = self.underlying.embed_query(text)
        self._memory_cache[text] = vec
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vec, f)
        except Exception:
            pass
        return vec


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """
    Factory function for instantiating LLM models based on provider.
    Supports runtime selection of provider and model.
    Supported providers: 'ollama', 'openai', 'bedrock'.
    """
    default_provider = os.getenv("LLM_PROVIDER", "ollama")
    default_model = os.getenv("LLM_MODEL", "gemma4:12b")
    default_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    selected_provider = (provider or os.getenv("LLM_PROVIDER") or default_provider).lower()
    selected_model = model or os.getenv("LLM_MODEL") or default_model

    if selected_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        url = base_url or os.getenv("OLLAMA_BASE_URL") or default_base_url
        return ChatOllama(model=selected_model, base_url=url, temperature=temperature, **kwargs)

    elif selected_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. Install it using `pip install langchain-openai`."
            ) from e

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            url = base_url or os.getenv("OLLAMA_BASE_URL") or default_base_url
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(model=default_model, base_url=url, temperature=temperature, **kwargs)
            except Exception:
                key = "sk-placeholder-key"

        url = base_url or os.getenv("OPENAI_BASE_URL") or None
        return ChatOpenAI(
            model=selected_model,
            api_key=key,
            base_url=url,
            temperature=temperature,
            **kwargs
        )

    elif selected_provider == "bedrock":
        region = kwargs.get("region") or os.getenv("AWS_REGION", "us-east-1")
        try:
            from langchain_aws import ChatBedrock
            return ChatBedrock(
                model_id=selected_model,
                region_name=region,
                model_kwargs={"temperature": temperature},
                **kwargs
            )
        except ImportError:
            try:
                from langchain_community.chat_models import BedrockChat
                return BedrockChat(
                    model_id=selected_model,
                    region_name=region,
                    model_kwargs={"temperature": temperature},
                    **kwargs
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-aws is required for AWS Bedrock provider. Install it using `pip install langchain-aws`."
                ) from e

    else:
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        url = base_url or os.getenv("OLLAMA_BASE_URL") or default_base_url
        return ChatOllama(model=selected_model, base_url=url, temperature=temperature, **kwargs)


def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    **kwargs: Any
) -> Embeddings:
    """
    Factory function for instantiating embedding models based on provider.
    Supported providers: 'ollama', 'openai', 'bedrock', 'cohere'.
    Supports embedding caching for warm-start performance optimization.
    """
    default_provider = os.getenv("EMBEDDING_PROVIDER", "ollama")
    default_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    default_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    selected_provider = (provider or os.getenv("EMBEDDING_PROVIDER") or default_provider).lower()
    selected_model = model or os.getenv("EMBEDDING_MODEL") or default_model

    if selected_provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings

        url = base_url or os.getenv("OLLAMA_BASE_URL") or default_base_url
        raw_embeddings = OllamaEmbeddings(model=selected_model, base_url=url, **kwargs)

    elif selected_provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI embeddings. Install it using `pip install langchain-openai`."
            ) from e

        key = api_key or os.getenv("OPENAI_API_KEY", "")
        raw_embeddings = OpenAIEmbeddings(model=selected_model, api_key=key, **kwargs)

    elif selected_provider == "bedrock":
        region = kwargs.get("region") or os.getenv("AWS_REGION", "us-east-1")
        try:
            from langchain_aws import BedrockEmbeddings
            raw_embeddings = BedrockEmbeddings(model_id=selected_model, region_name=region, **kwargs)
        except ImportError:
            try:
                from langchain_community.embeddings import BedrockEmbeddings
                raw_embeddings = BedrockEmbeddings(model_id=selected_model, region_name=region, **kwargs)
            except ImportError as e:
                raise ImportError(
                    "langchain-aws is required for AWS Bedrock embeddings. Install it using `pip install langchain-aws`."
                ) from e

    elif selected_provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import CohereEmbeddings
            except ImportError as e:
                raise ImportError(
                    "langchain-cohere is required for Cohere embeddings. Install it using `pip install langchain-cohere`."
                ) from e

        key = api_key or os.getenv("COHERE_API_KEY", "")
        raw_embeddings = CohereEmbeddings(model=selected_model, cohere_api_key=key, **kwargs)

    else:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            from langchain_community.embeddings import OllamaEmbeddings

        url = base_url or os.getenv("OLLAMA_BASE_URL") or default_base_url
        raw_embeddings = OllamaEmbeddings(model=selected_model, base_url=url, **kwargs)

    if use_cache:
        return CachedEmbeddingsWrapper(raw_embeddings)
    return raw_embeddings
