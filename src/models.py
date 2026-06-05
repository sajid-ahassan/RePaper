import warnings
from typing import Literal

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")
from pydantic import BaseModel, Field






class RetrieveDecision(BaseModel):
    route : Literal['retrieval','direct_generate','verify_claim'] = Field(
        ...,
        description="'retrieval' if the query needs vector search or/and web search result to make context. 'direct_generate' if the parametric knowladge is enough to answer the query without any hellocination.'verify_claim' if the query ask any claim that need to be verified using web search and arxiv research paper search."
    )
    reason : list[str]
    

class DecomposeDecision(BaseModel):
    need_decompose : bool = Field(...,description="True if the query is complex and it need to decompose, else False.")
    
    
class DecomposeQueries(BaseModel):
    queries : list[str] = Field(description='Decompose the query into a list of subquries')
    
    
class relevency_schema(BaseModel):
    is_relevent: bool = Field(description="True if document is relevent to the query else False")
    
class claim_ev(BaseModel):
    title : str
    url : str
    summary : str
    
class claim_vardics(BaseModel):
    claim_verification_result : Literal['superseded','partially superseded','not superseded']
    claim_evidance : list[claim_ev]
    
class btw_retrieval(BaseModel):
    need_web_retrieval : bool = Field(description='True if the query requires web search else False')