from pydantic import BaseModel

class DataModel(BaseModel):
    Review: str

    def columns(self):
        return ["Review"]
