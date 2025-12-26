import sys
import shutil
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Mock dependencies if they don't exist, so we can test the logic flow
from unittest.mock import MagicMock
import sys

def mock_module(module_name):
    if module_name not in sys.modules:
        m = MagicMock()
        sys.modules[module_name] = m
        return m
    return sys.modules[module_name]

# Mocking modules before importing builder
# We need to ensure nested modules exist too
docling = mock_module("docling")
docling_converter = mock_module("docling.document_converter")
docling_models = mock_module("docling.datamodel.base_models")
docling_opts = mock_module("docling.datamodel.pipeline_options")

chromadb = mock_module("chromadb")
chromadb_utils = mock_module("chromadb.utils")
chromadb_ef = mock_module("chromadb.utils.embedding_functions")
chromadb_config = mock_module("chromadb.config")

lts = mock_module("langchain_text_splitters")

# Now we can import builder, and it should find "docling" etc.
docling_converter.DocumentConverter = MagicMock()
lts.MarkdownHeaderTextSplitter = MagicMock()
chromadb.PersistentClient = MagicMock()

# Reload builder if it was already imported (unlikely in fresh pytest run but good practice)
if "mcp_manual_walker.builder" in sys.modules:
    del sys.modules["mcp_manual_walker.builder"]
from mcp_manual_walker import builder


@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a simple text file pretending to be PDF for logic test."""
    # We don't need real PDF content since we mock the converter
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("dummy pdf content")
    return pdf_path


def test_builder_smoke(tmp_path, sample_pdf):
    """Smoke test for builder.py using MOCKED dependencies."""
    
    output_dir = tmp_path / "output"
    # Create nested PDF structure to test recursive scan
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    sub_dir = pdf_dir / "subdir"
    sub_dir.mkdir()
    
    shutil.copy(sample_pdf, pdf_dir / "sample.pdf")
    shutil.copy(sample_pdf, sub_dir / "nested.pdf")
    
    # Configure Mocks
    
    # 1. Docling Converter
    mock_inst = MagicMock()
    mock_doc = MagicMock()
    # Export returns markdown
    mock_doc.export_to_markdown.return_value = "# Header 1\n\nContent"
    
    # Mock doc.texts for chunking logic
    # Structure:
    # Header 1 (Level 1)
    #   Text 1
    #   Header 2 (Level 2)
    #     Text 2
    
    item1 = MagicMock()
    item1.text = "Header 1"
    item1.label = "section_header"
    item1.level = 1
    
    item2 = MagicMock()
    item2.text = "Text 1"
    item2.label = "text"
    # No level attr usually on text
    
    item3 = MagicMock()
    item3.text = "Header 2"
    item3.label = "section_header"
    item3.level = 2
    
    item4 = MagicMock()
    item4.text = "Text 2"
    item4.label = "text"
    
    mock_doc.texts = [item1, item2, item3, item4]
    
    mock_res = MagicMock()
    mock_res.document = mock_doc
    mock_inst.convert.return_value = mock_res
    
    builder.DocumentConverter.return_value = mock_inst
    
    # 2. Text Splitter - No longer used, but import mocked anyway
    
    # 3. ChromaDB
    # get_embedding_function returns an object
    # client.get_or_create_collection returns collection
    mock_client = MagicMock()
    mock_coll = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_coll
    builder.chromadb.PersistentClient.return_value = mock_client
    
    # Run the build
    builder.build(pdf_dir, output_dir, reset=True)
    
    # Verify Interactions
    
    # Did we convert both files?
    assert mock_inst.convert.call_count == 2
    
    # Did we add to Chroma?
    # Each file should produce chunks.
    # File 1: Header 1 (+ Text 1), Header 2 (+ Text 2) -> 2 chunks (because flush happens at next header or end)
    # Wait, simple logic:
    # 1. H1 -> Buffer=[# H1]
    # 2. T1 -> Buffer=[# H1, T1]
    # 3. H2 -> Flush (# H1\nT1) to Chunk 1. Buffer=[# H2]
    # 4. T2 -> Buffer=[# H2, T2]
    # End -> Flush (# H2\nT2) to Chunk 2.
    # Total 2 chunks per file.
    # 2 files => 4 chunks. But mock might be reset or reused.
    # Since we mocked builder.DocumentConverter(), it returns same instance mock_inst.
    # So add is called twice (batch likely per file in current impl).
    
    assert mock_coll.add.call_count == 2 
    
    # Check args of last call
    call_args = mock_coll.add.call_args
    kwargs = call_args[1]
    
    assert len(kwargs["ids"]) == 2
    # Check hierarchy
    # Chunk 1 meta: Header 1
    # Chunk 2 meta: Header 1 > Header 2
    metas = kwargs["metadatas"]
    assert metas[0]["section_hierarchy"] == "Header 1"
    assert metas[1]["section_hierarchy"] == "Header 1 > Header 2"
    
    # Check source path relative
    # We don't know which file was processed last (glob order), but it should be relative
    source = metas[0]["source"]
    assert source in ["sample.pdf", "subdir/nested.pdf"]
