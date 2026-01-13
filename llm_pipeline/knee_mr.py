
import asyncio
import pandas as pd
from pathlib import Path
from .extraction import MRExtractor

def run_knee_job(
    input_xlsx: str, 
    output_text_csv: str, 
    output_numeric_csv: str, 
    api_key: str, 
    base_url: str,
    filter_items: list = None,
    model: str = "doubao-1-5-pro-32k-250115"
):
    """
    Run Knee MRI Extraction Job.
    """
    df_in = pd.read_excel(input_xlsx)
    
    # Optional filtering by Item Name if required (as per notebook)
    if filter_items:
        df_in = df_in[df_in["检查项目"].isin(filter_items)]
    
    labeler = MRExtractor(
        api_key=api_key, 
        base_url=base_url, 
        anatomy="knee",
        model=model
        # concurrency=128 # Can be passed in
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    text_lines, numeric_lines = loop.run_until_complete(
        labeler.process_custom_dataframe(
            df_in,
            id_col="检查号",
            item_col="检查项目",
            text_cols=["检查所见", "诊断意见"]
        )
    )
    
    # Save Outputs
    with open(output_text_csv, 'w', encoding='utf-8') as f:
        f.write("Description\n")
        for line in text_lines:
            f.write(f"{line}\n")
            
    with open(output_numeric_csv, 'w', encoding='utf-8') as f:
        f.write("Label\n")
        for line in numeric_lines:
            f.write(f"{line}\n")
            
    print(f"Saved Knee Text to {output_text_csv}")
    print(f"Saved Knee Numeric to {output_numeric_csv}")

if __name__ == "__main__":
    import os
    
    # Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
    
    # Paths
    raw_excel = "./data/MR-report-clean.xlsx"
    temp_input_xlsx = "./llm_pipeline/knee_mr_test_input.xlsx"
    output_text = "./llm_pipeline/knee_mr_output_text.csv"
    output_numeric = "./llm_pipeline/knee_mr_output_numeric.csv"
    
    # Prepare Test Data
    print(f"Loading raw data from: {raw_excel}")
    if os.path.exists(raw_excel):
        df = pd.read_excel(raw_excel)
        print(f"Total rows: {len(df)}")
        
        # Filter for Knee MRI (simplistic filter for testing)
        # Using string matching on '检查项目'
        mask = df["检查项目"].astype(str).str.contains("膝", na=False)
        df_knee = df[mask].copy()
        print(f"Found {len(df_knee)} rows containing '膝' (Knee)")
        
        # Take first 100
        df_subset = df_knee.head(100)
        df_subset.to_excel(temp_input_xlsx, index=False)
        print(f"Saved 100 sample rows to {temp_input_xlsx}")
        
        # Run Job
        # We don't need 'filter_items' here because we already pre-filtered the input file.
        run_knee_job(
            input_xlsx=temp_input_xlsx,
            output_text_csv=output_text,
            output_numeric_csv=output_numeric,
            api_key=API_KEY,
            base_url=BASE_URL,
            filter_items=None,
            model=MODEL
        )
        
    else:
        print(f"Error: File not found {raw_excel}")
