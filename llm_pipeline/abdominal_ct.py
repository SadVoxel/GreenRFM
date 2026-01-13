import pandas as pd
from .extraction import LLMLabeler
import asyncio

def run_abdominal_job(
    input_csv: str, 
    output_csv: str, 
    report_col: str, 
    api_key: str, 
    base_url: str, 
    model: str="doubao-pro-32k-241215"
):
    """
    Run Abdominal CT Extraction Job (30 classes).
    Uses Merlin/AH-Abd schema.
    """
    df = pd.read_csv(input_csv)
    
    # Initialize Labeler with 30_class mode
    labeler = LLMLabeler(
        api_key=api_key, 
        base_url=base_url, 
        model=model, 
        mode="30_class"
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Process
    df_result = loop.run_until_complete(labeler.process_dataframe(df, report_col))
    
    # Save
    df_result.to_csv(output_csv, index=False)
    print(f"Saved Abdominal CT labels to {output_csv}")


if __name__ == "__main__":
    import os
    
    # Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
    
    # Report Source 
    reports_source = "./data/matched_reports.csv"
    
    # Temp input/output for testing
    temp_input_csv = "./llm_pipeline/abdominal_test_input.csv"
    output_csv = "./llm_pipeline/abdominal_test_output.csv"
    
    print(f"Loading raw data from: {reports_source}")
    if os.path.exists(reports_source):
        df = pd.read_csv(reports_source)
        print(f"Total rows: {len(df)}")
        
        # Prepare Report Column (findings + impression)
        df["findings_translated"] = df["findings_translated"].fillna("")
        df["impression_translated"] = df["impression_translated"].fillna("")
        df["full_report"] = df["findings_translated"] + " " + df["impression_translated"]
        
        # Take first 50 non-empty
        mask = df["full_report"].str.len() > 10
        df_subset = df[mask].head(50)
        
        df_subset.to_csv(temp_input_csv, index=False)
        print(f"Saved 50 sample rows to {temp_input_csv}")
        
        # Run Job
        run_abdominal_job(
            input_csv=temp_input_csv,
            output_csv=output_csv,
            report_col="full_report",
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL
        )
        
    else:
        print(f"Error: File not found {reports_source}")
