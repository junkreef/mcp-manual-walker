from mcp_manual_walker.main import _merge_chunks

def test_merge_no_overlap():
    chunks = ["Hello World.", "This is a test."]
    merged = _merge_chunks(chunks)
    assert merged == "Hello World.\n\nThis is a test."

def test_merge_simple_overlap():
    # overlap = "World."
    chunks = ["Hello World.", "World. This is a test."]
    merged = _merge_chunks(chunks)
    assert merged == "Hello World. This is a test."

def test_merge_partial_overlap():
    # overlap = "lap"
    chunks = ["This is overlap", "lap logic check"]
    merged = _merge_chunks(chunks)
    assert merged == "This is overlap logic check"

def test_merge_empty():
    assert _merge_chunks([]) == ""

def test_merge_single():
    assert _merge_chunks(["Only one"]) == "Only one"

def test_merge_multiple():
    # A->B overlap: "B"
    # B->C overlap: "C"
    chunks = ["Chunk A ends with B", "B starts Chunk B and ends with C", "C starts Chunk C"]
    merged = _merge_chunks(chunks)
    expected = "Chunk A ends with B starts Chunk B and ends with C starts Chunk C"
    assert merged == expected

def test_merge_no_overlap_edge_case():
    # Suffix matches but not fully at start? No, prefix must match suffix.
    # "abc" end with "c"
    # "def" starts with "d". No match.
    chunks = ["abc", "def"]
    assert _merge_chunks(chunks) == "abc\n\ndef"
