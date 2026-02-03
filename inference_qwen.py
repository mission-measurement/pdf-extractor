import os
import json
import argparse
import re
import csv
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from document_loader import DocumentLoader


EXTRACTION_PROMPT = """Extract from this document:
- program_name, organization_name, grant_number
- total_budget (number), federal_funding (number), budget_unit (yearly/total_grant/unknown)
- total_reach (number), reach_type (individuals/students/organizations/unknown)
- program_description (2-3 sentences)
- geographic_location (states, cities, counties)
- start_date, end_date, focus_area

DOCUMENT:
{text_content}

Return JSON with value and confidence (0-1) for each field. Use null for unfound values."""


class Extractor:
    def __init__(self, model_path, base_model="Qwen/Qwen2-VL-7B-Instruct", max_pages=3):
        self.max_pages = max_pages
        self.loader = DocumentLoader()
        
        lora_path = os.path.join(model_path, "lora_weights")
        
        if os.path.exists(lora_path):
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base, lora_path)
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
            )
        
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    def extract(self, file_path):
        doc = self.loader.load(file_path)
        prompt = EXTRACTION_PROMPT.format(text_content=doc.get_first_n_pages_text(self.max_pages))
        
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=True)
        
        response = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON in response")
        
        data = json.loads(json_match.group(0))
        
        def get_val(field):
            if field in data:
                f = data[field]
                return f.get('value') if isinstance(f, dict) else f
            return None
        
        return {
            'file': Path(file_path).name,
            'document_id': doc.document_id,
            'program_name': get_val('program_name'),
            'organization_name': get_val('organization_name'),
            'grant_number': get_val('grant_number'),
            'total_budget': get_val('total_budget'),
            'federal_funding': get_val('federal_funding'),
            'budget_unit': get_val('budget_unit'),
            'total_reach': get_val('total_reach'),
            'reach_type': get_val('reach_type'),
            'program_description': get_val('program_description'),
            'geographic_location': get_val('geographic_location'),
            'start_date': get_val('start_date'),
            'end_date': get_val('end_date'),
            'focus_area': get_val('focus_area')
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--file")
    parser.add_argument("--directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    
    extractor = Extractor(args.model)
    
    if args.file:
        print(json.dumps(extractor.extract(args.file), indent=2))
    
    elif args.directory:
        files = list(Path(args.directory).glob("*.pdf")) + list(Path(args.directory).glob("*.PDF"))
        results = []
        
        for i, f in enumerate(sorted(set(files))):
            print(f"[{i+1}/{len(files)}] {f.name}")
            try:
                results.append(extractor.extract(str(f)))
            except Exception as e:
                results.append({'file': f.name, 'error': str(e)})
        
        if args.output:
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
