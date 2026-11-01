from langchain_text_splitters import RecursiveCharacterTextSplitter


def splitter(size, overlap):
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)


def recursive(pages, size=800, overlap=150):
    split = splitter(size, overlap)
    chunks = []
    for page in pages:
        meta = {k: v for k, v in page.items() if k != "text"}
        chunks += [{**meta, "text": t} for t in split.split_text(page["text"])]
    return chunks


def parent_child(pages, parent_size=1500, child_size=300):
    split_parent, split_child = splitter(parent_size, 200), splitter(child_size, 50)
    chunks = []
    for page in pages:
        meta = {k: v for k, v in page.items() if k != "text"}
        for parent in split_parent.split_text(page["text"]):
            chunks += [{**meta, "text": c, "parent": parent} for c in split_child.split_text(parent)]
    return chunks
