import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import hashlib

from document_loader import DocumentLoader


@dataclass
class TrainingExample:
    example_id: str
    document_id: str
    page_texts: List[str]
    program_name: str
    organization_name: str
    grant_number: str
    total_budget: Optional[float]
    total_reach: Optional[int]
    reach_type: str
    program_description: str
    geographic_states: List[str]
    focus_area: str
    start_date: str
    end_date: str
    
    def to_training_format(self):
        text_content = "\n\n".join([f"=== Page {i+1} ===\n{t}" for i, t in enumerate(self.page_texts)])
        
        user_prompt = f"""Extract from this document:
- program_name, organization_name, grant_number
- total_budget, total_reach, reach_type
- program_description, geographic_location, focus_area
- start_date, end_date

DOCUMENT:
{text_content}

Return JSON with value and confidence for each field."""

        response = {
            "program_name": {"value": self.program_name, "confidence": 1.0},
            "organization_name": {"value": self.organization_name, "confidence": 1.0},
            "grant_number": {"value": self.grant_number, "confidence": 1.0},
            "total_budget": {"value": self.total_budget, "confidence": 1.0},
            "total_reach": {"value": self.total_reach, "confidence": 1.0},
            "reach_type": {"value": self.reach_type, "confidence": 1.0},
            "program_description": {"value": self.program_description, "confidence": 1.0},
            "geographic_location": {"value": {"states": self.geographic_states}, "confidence": 1.0},
            "focus_area": {"value": self.focus_area, "confidence": 1.0},
            "start_date": {"value": self.start_date, "confidence": 1.0},
            "end_date": {"value": self.end_date, "confidence": 1.0}
        }
        
        return {
            "id": self.example_id,
            "conversations": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps(response)}
            ]
        }


class TrainingDataPreparer:
    def __init__(self, pdf_directory: str, verbose: bool = False):
        self.pdf_directory = Path(pdf_directory)
        self.verbose = verbose
        self.loader = DocumentLoader(verbose=verbose)
        self.pdf_index = self._build_pdf_index()
    
    def _build_pdf_index(self):
        index = {}
        for pdf_path in list(self.pdf_directory.glob("*.pdf")) + list(self.pdf_directory.glob("*.PDF")):
            match = re.search(r'(\d{2}[A-Z]{2}\d{6})', pdf_path.name)
            if match:
                doc_id = match.group(1)
                if doc_id not in index:
                    index[doc_id] = []
                index[doc_id].append(str(pdf_path))
        
        if self.verbose:
            print(f"Indexed {len(index)} unique document IDs from {sum(len(v) for v in index.values())} PDF files")
        return index
    
    def _parse_location(self, row: pd.Series) -> List[str]:
        states = []
        for col in ['intervention_location', 'organization_region']:
            if col in row and pd.notna(row[col]):
                val = str(row[col])
                if val.startswith('['):
                    try:
                        for item in json.loads(val.replace("'", '"')):
                            if 'United States' in str(item):
                                state = str(item).split(',')[0].strip()
                                if state and state not in states:
                                    states.append(state)
                    except:
                        pass
                elif val:
                    states.append(val)
        return states
    
    def _parse_reach_type(self, val) -> str:
        if not val or pd.isna(val):
            return "unknown"
        val_lower = str(val).lower()
        for key in ['individuals', 'organizations', 'students', 'volunteers', 'children', 'youth', 'families']:
            if key in val_lower:
                return key
        return "individuals"
    
    def create_training_example(self, row: pd.Series, excel_source: str) -> Optional[TrainingExample]:
        program_name = row.get('program_name', '')
        match = re.search(r'(\d{2}[A-Z]{2}\d{6})', str(program_name))
        if not match:
            return None
        
        doc_id = match.group(1)
        if doc_id not in self.pdf_index:
            if self.verbose:
                print(f"  No PDF found for doc ID: {doc_id}")
            return None
        
        pdf_paths = self.pdf_index[doc_id]
        pdf_path = next((p for p in pdf_paths if 'application' in p.lower()), pdf_paths[0])
        
        try:
            doc = self.loader.load(pdf_path)
        except Exception as e:
            if self.verbose:
                print(f"  Error loading {pdf_path}: {e}")
            return None
        
        return TrainingExample(
            example_id=hashlib.md5(f"{doc_id}_{excel_source}".encode()).hexdigest()[:12],
            document_id=doc_id,
            page_texts=[p.text_content for p in doc.pages[:3]],
            program_name=str(program_name) if pd.notna(program_name) else '',
            organization_name=str(row.get('organization_name', '')) if pd.notna(row.get('organization_name')) else '',
            grant_number=doc_id,
            total_budget=float(row['totalbudget_usd']) if pd.notna(row.get('totalbudget_usd')) else None,
            total_reach=int(row['primary_outcome_measured']) if pd.notna(row.get('primary_outcome_measured')) else None,
            reach_type=self._parse_reach_type(row.get('program_beneficiary_type')),
            program_description=str(row.get('program_description', ''))[:500] if pd.notna(row.get('program_description')) else '',
            geographic_states=self._parse_location(row),
            focus_area=str(row.get('focus_area_name', '')) if pd.notna(row.get('focus_area_name')) else '',
            start_date=str(row.get('program_report_start_at', ''))[:10] if pd.notna(row.get('program_report_start_at')) else '',
            end_date=str(row.get('program_report_end_at', ''))[:10] if pd.notna(row.get('program_report_end_at')) else ''
        )
    
    def prepare_training_data(self, excel_paths: List[str], output_dir: str) -> List[TrainingExample]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_examples = []
        for excel_path in excel_paths:
            if self.verbose:
                print(f"\nProcessing: {excel_path}")
            
            df = pd.read_excel(excel_path)
            if self.verbose:
                print(f"Loaded {len(df)} rows from {excel_path}")
            
            for _, row in df.iterrows():
                example = self.create_training_example(row, excel_path)
                if example:
                    all_examples.append(example)
                    if self.verbose:
                        print(f"  Created example for {example.document_id}")
        
        print(f"Created {len(all_examples)} training examples")
        
        with open(output_dir / 'training_data.jsonl', 'w') as f:
            for ex in all_examples:
                f.write(json.dumps(ex.to_training_format()) + '\n')
        
        stats = {
            'total_examples': len(all_examples),
            'with_budget': sum(1 for e in all_examples if e.total_budget),
            'with_reach': sum(1 for e in all_examples if e.total_reach),
            'focus_areas': {}
        }
        for e in all_examples:
            if e.focus_area:
                stats['focus_areas'][e.focus_area] = stats['focus_areas'].get(e.focus_area, 0) + 1
        
        print(f"Training data saved to {output_dir}")
        print(json.dumps(stats, indent=2))
        
        return all_examples