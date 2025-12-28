from pydantic import BaseModel

class SentimentRequest(BaseModel):
    """
    Request model for sentiment prediction.
    """
    text: str


class SentimentResponse(BaseModel):
    """
    Response model returned by the sentiment API.
    """
    sentiment: str
