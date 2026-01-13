import pandas as pd
from .extraction import LLMLabeler
import asyncio

def run_chest_job(
    input_csv: str, 
    output_csv: str, 
    report_col: str, 
    api_key: str, 
    base_url: str, 
    model: str="doubao-pro-32k-241215"
):
    """
    Run Chest CT Extraction Job (18 classes).
    Uses CT-RATE/AH-Chest schema.
    """
    df = pd.read_csv(input_csv)
    
    # Initialize Labeler with 18_class mode
    labeler = LLMLabeler(
        api_key=api_key, 
        base_url=base_url, 
        model=model, 
        mode="18_class"
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Process
    df_result = loop.run_until_complete(labeler.process_dataframe(df, report_col))
    
    # Save
    df_result.to_csv(output_csv, index=False)
    print(f"Saved Chest CT labels to {output_csv}")

if __name__ == "__main__":
    import os
    
    # 1. Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
    
    # 2. Paths
    # Input CT-RATE reports
    raw_csv = "./data/train_reports.csv"
    
    # Temporary input for the job (first 100 samples, merged columns)
    temp_input_csv = "./llm_pipeline/chest_ct_pipeline_input.csv"
    
    # Final labeled output
    output_result_csv = "./llm_pipeline/chest_ct_pipeline_output.csv"
    
    # 3. Prepare Data (Merge Findings + Impressions)
    print(f"Loading raw data from: {raw_csv}")
    if os.path.exists(raw_csv):
        df = pd.read_csv(raw_csv)
        print(f"Total rows: {len(df)}")
        
        # Take first 100
        df_subset = df.head(100).copy()
        
        # Merge report text if 'report' column missing
        if "report" not in df_subset.columns:
            # Check for CT-RATE specific columns
            if "Findings_EN" in df_subset.columns and "Impressions_EN" in df_subset.columns:
                df_subset["report"] = df_subset["Findings_EN"].fillna("") + " " + df_subset["Impressions_EN"].fillna("")
            elif "findings" in df_subset.columns and "impression" in df_subset.columns:
                 df_subset["report"] = df_subset["findings"].fillna("") + " " + df_subset["impression"].fillna("")
            else:
                # Last resort: try to find any text column or just use empty if testing structure
                print("Warning: Could not auto-detect Findings/Impressions columns. Using empty report.")
                df_subset["report"] = ""

        # Save to temp CSV as input for run_chest_job
        df_subset.to_csv(temp_input_csv, index=False)
        print(f"Created temp input file with 100 samples: {temp_input_csv}")
        
        # 4. Run Job
        run_chest_job(
            input_csv=temp_input_csv,
            output_csv=output_result_csv,
            report_col="report",
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL
        )
        
    else:
        print(f"Error: Raw CSV not found at {raw_csv}")