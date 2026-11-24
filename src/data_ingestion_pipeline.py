from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(paths):
    docs = []
    for path in paths:
        path= Path(path)
        loaded = PyPDFLoader(str(path)).load()

        for d in loaded:
            d.metadata['source'] = path.name
            d.metadata['page'] = (
                int(d.metadata.get('page', 0)) + 1
            )
        docs.extend(loaded)
    return docs    
