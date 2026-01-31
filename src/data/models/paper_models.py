"""
Data models for Research Navigator papers
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Author(BaseModel):
    """Author information"""
    name: str
    affiliations: List[str] = Field(default_factory=list)


class Paper(BaseModel):
    """Academic paper with metadata"""
    
    arxiv_id: str
    title: str
    authors: List[Author]
    institution: str
    abstract: str
    categories: List[str]
    published: datetime
    updated: datetime
    arxiv_url: str
    pdf_url: str
    s2_paper_id: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PaperCollection(BaseModel):
    """Collection of papers with metadata"""
    papers: List[Paper]
    collected_at: datetime = Field(default_factory=datetime.now)
    total_count: int
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
