import os
import lancedb  

from Pharmagent.app.constants import RSC_PATH

      

###############################################################################
class VectorDatabase:

    def __init__(self):
        self.collection = "embeddings"
        self.db_path = os.path.join(RSC_PATH, 'fragments.lancedb')
        self.db = lancedb.connect(self.db_path)
        try:
            self.col = self.db.open_collection(self.collection)
        except Exception:
            self.col = self.db.create_collection(self.collection, overwrite=True)

    def save_embeddings(self, fragments, embeddings):
        for text, vector in zip(fragments, embeddings):
            self.col.insert({"text": text, "vector": vector})

    def search(self, query_vec, k=5):
        return self.col.search(query_vec, k=k)

 
    
    