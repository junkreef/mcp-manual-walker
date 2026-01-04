import logging
from typing import Optional

try:
    import chromadb
    from chromadb import Documents, EmbeddingFunction, Embeddings
except ImportError:
    chromadb = None
    EmbeddingFunction = object  # type: ignore

logger = logging.getLogger(__name__)

# Common Model Name to ensure compatibility between builder and search.
# FastEmbed supports "intfloat/multilingual-e5-small".
MODEL_NAME = "intfloat/multilingual-e5-small"


class FastEmbedEmbeddingFunction(EmbeddingFunction):
    """
    Wrapper for FastEmbed to be compatible with ChromaDB EmbeddingFunction.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError("fastembed is not installed. Please install it.")

        # Manually register the model if it's the target one and not supported by default
        if model_name == "intfloat/multilingual-e5-small":
            try:
                # check if already supported (to avoid error on re-registration)
                supported = any(
                    m["model"] == model_name
                    for m in TextEmbedding.list_supported_models()
                )
                if not supported:
                    logger.info(
                        f"Registering custom model {model_name} using Xenova artifacts"
                    )
                    # Need ModelSource object
                    from fastembed.common.model_description import (
                        ModelSource,
                        PoolingType,
                    )

                    TextEmbedding.add_custom_model(
                        model=model_name,
                        pooling=PoolingType.MEAN,
                        normalization=True,
                        sources=ModelSource(hf="Xenova/multilingual-e5-small"),
                        dim=384,
                        model_file="onnx/model_quantized.onnx",
                    )
            except ValueError as e:
                # Handle race condition or if it was registered between checks
                logger.debug(f"Model already registered or error: {e}")
            except Exception as e:
                logger.warning(f"Failed to register custom model: {e}")

        logger.info(f"Initializing FastEmbed with model: {model_name}")
        self.model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return list(self.model.embed(input))


def get_embedding_function() -> Optional[EmbeddingFunction]:
    """
    Returns an appropriate embedding function based on available libraries.
    Prioritizes SentenceTransformers (Builder/Heavy) if available,
    otherwise falls back to FastEmbed (Search/Light).
    """
    if not chromadb:
        logger.error("chromadb is not installed.")
        return None

    # Try SentenceTransformers first (Builder preference)
    try:
        # Start a check to see if sentence_transformers is actually importable
        import sentence_transformers  # noqa: F401
        from chromadb.utils import embedding_functions

        logger.info(f"Using SentenceTransformers with model: {MODEL_NAME}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
    except ImportError:
        pass

    # Fallback to FastEmbed
    try:
        import fastembed  # noqa: F401

        logger.info(f"Using FastEmbed with model: {MODEL_NAME}")
        return FastEmbedEmbeddingFunction(model_name=MODEL_NAME)
    except ImportError:
        logger.error("Neither sentence-transformers nor fastembed found.")
        return None
