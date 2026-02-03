import re
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass
import pdfplumber


@dataclass
class PageContent:
    page_number: int
    text_content: str


@dataclass
class Document:
    file_path: str
    file_name: str
    document_id: str
    num_pages: int
    pages: List[PageContent]
    
    def get_all_text(self) -> str:
        return "\n\n".join([p.text_content for p in self.pages])


class DocumentLoader:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def _extract_document_id(self, file_name: str) -> str:
        match = re.search(r'(\d{2}[A-Z]{2}\d{6})', file_name)
        return match.group(1) if match else Path(file_name).stem.split('_')[0]
    
    def load(self, file_path: Union[str, Path]) -> Document:
        file_path = Path(file_path)
        
        if self.verbose:
            print(f"Loading: {file_path.name}")
        
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                pages.append(PageContent(
                    page_number=i + 1,
                    text_content=page.extract_text() or ""
                ))
        
        return Document(
            file_path=str(file_path),
            file_name=file_path.name,
            document_id=self._extract_document_id(file_path.name),
            num_pages=len(pages),
            pages=pages
        )
    
    def load_directory(self, dir_path: Union[str, Path]) -> List[Document]:
        dir_path = Path(dir_path)
        files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))
        
        documents = []
        for file_path in sorted(set(files)):
            try:
                documents.append(self.load(file_path))
            except Exception as e:
                raise Exception(f"Error loading {file_path.name}: {e}")
        
        return documents