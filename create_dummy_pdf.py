from pypdf import PdfWriter

# Create a writer
writer = PdfWriter()

# Add two blank pages
writer.add_blank_page(width=210, height=297)
writer.add_blank_page(width=210, height=297)

# Add metadata
writer.add_metadata({"/Title": "Dummy PDF for Testing"})

# Add a parent bookmark pointing to the first page (index 0)
parent_bookmark = writer.add_outline_item("Chapter 1", 0)

# Add a child bookmark pointing to the second page (index 1)
# The parent parameter should be the OutlineItem object returned before
writer.add_outline_item("Section 1.1", 1, parent=parent_bookmark)

# Write the PDF to a file
with open("tests/assets/dummy.pdf", "wb") as f:
    writer.write(f)