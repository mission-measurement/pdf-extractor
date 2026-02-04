# AmeriCorps PDF Extractor

Extracts structured data from PDFs using a fine-tuned vision model for low-cost, high-accuracy extraction.

## Overview

This system processes any type of PDF (applications, PPRs, scanned docs, images, handwriting) and extracts program name, budget amount, budget unit (yearly/quarterly/net), reach number, reach type (people/students/families), program description, and geography (country/state/city).

This uses fine-tuned Qwen3-VL-32B vision model trained on hand-coded data. Runs on Google Colab Pro with A100 GPU using 4-bit quantization.

## Input

Single PDF or folder of PDFs.

## Output
```
program_name: XYZ Program
budget_amount: 500000
budget_unit: yearly
reach_number: 1500
reach_type: students
description: After-school mentoring for at-risk youth
geography: Chicago, Illinois, USA
```

## How to Run
Make sure you create a folder called pdfs first with the applications and the PPRs.

1. Install dependencies: `pip install -r requirements.txt`
2. Run inference: `python inference_qwen.py --model fine_tuned_model/ --input pdfs/ --output results.csv`

For single PDF: `python inference_qwen.py --model fine_tuned_model/ --input document.pdf --output results.csv`

## Retraining (Optional)

If you need to retrain the model with new data:

1. Prepare training data: `python training_data.py --excel ground_truth.xlsx --pdf_folder pdfs/`
2. Fine-tune model: `python finetune_qwen.py --training_data training_data.json --output_model fine_tuned_model/`

## Pipeline

Document loader reads PDFs, training prep maps Excel to PDFs, fine-tuning trains model on examples, inference processes all documents, output aggregates structured data.

## Notes
This is not working very well for the reach but Laiba will continue to work on improving accuracy. 

## Key Files

AmeriCorps Tracker and Outcomes_092225.xlsx, 20260120_AmeriCorps_orgid.xlsx for hand-coded data. PDF folders for training and testing.
