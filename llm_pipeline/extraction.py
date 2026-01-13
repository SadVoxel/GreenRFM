
import asyncio
import pandas as pd
import re
from typing import List, Tuple, Optional, Any, Any
from pathlib import Path
import re

# Try import, but don't fail immediately if not installed (allows loading module for other things)
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Import labels from config to ensure consistency
try:
    from LeanRad.config.config import LABEL_COLUMNS_30, LABEL_COLUMNS_18
except ImportError:
    # Fallback to local definitions if config is missing
    LABEL_COLUMNS_30 = [
     "submucosal_edema", "renal_hypodensities", "aortic_valve_calcification", "coronary_calcification",
     "thrombosis", "metastatic_disease", "pancreatic_atrophy", "renal_cyst", "osteopenia",
     "surgically_absent_gallbladder", "atelectasis", "abdominal_aortic_aneurysm", "anasarca",
     "hiatal_hernia", "lymphadenopathy", "prostatomegaly", "biliary_ductal_dilation", "cardiomegaly",
     "splenomegaly", "hepatomegaly", "atherosclerosis", "ascites", "pleural_effusion",
     "hepatic_steatosis", "appendicitis", "gallstones", "hydronephrosis", "bowel_obstruction",
     "free_air", "fracture",
    ]

    LABEL_COLUMNS_18 = [
        "medical_material", "arterial_wall_calcification", "cardiomegaly", "pericardial_effusion",
        "coronary_artery_wall_calcification", "hiatal_hernia", "lymphadenopathy", "emphysema",
        "atelectasis", "lung_nodule", "lung_opacity", "pulmonary_fibrotic_sequela", "pleural_effusion",
        "mosaic_attenuation_pattern", "peribronchial_thickening", "consolidation", "bronchiectasis",
        "interlobular_septal_thickening"
    ]

LABELS_KNEE = [
    "Meniscus_Medial", "Meniscus_Lateral", "ACL", "PCL", "MCL", "LCL", 
    "Cartilage", "Bone_Edema", "Bone_Fracture", "Effusion/Baker", 
    "Synovitis/Burs", "PostOp", "DEG-OA", "NOR"
]

LABELS_SPINE = [
    "DEG-DIS", "DEG-BUL", "DEG-PRO", "DEG-SPO", "DEG-STE", 
    "DEG-END", "TRA-VCF", "TRA-POST", "NOR"
]

SYSTEM_PROMPT_KNEE = """
你是 骨肌系统 Knee MRI 报告解析专家。

① 标签与列序  
0 Meniscus_Medial   内侧半月板损伤  
1 Meniscus_Lateral  外侧半月板损伤  
2 ACL               前交叉韧带损伤  
3 PCL               后交叉韧带损伤  
4 MCL               内侧副韧带损伤  
5 LCL               外侧副韧带损伤  
6 Cartilage         关节软骨缺损 / 骨软骨损伤  
7 Bone_Edema        骨髓水肿 / 挫伤  
8 Bone_Fracture     骨折 / 塌陷  
9 Effusion/Baker    关节腔积液或腘窝囊肿  
10 Synovitis/Burs   滑膜炎 / 滑囊炎  
11 PostOp           术后改变 / 内固定  
12 DEG-OA           退行性骨关节病 / 骨赘  
13 NOR              正常或未见异常（若其它标签均为 0，则 NOR=1，否则 NOR=0）

② 抽取规则  
• 逐段落识别 “检查号＋检查项目”。  
• 出现即置 1；未出现置 0。  
• 同一检查项目只输出一行二进制结果。  

③ 输出  
仅 CSV，无标题或说明。  
列顺序：检查号,检查项目,Meniscus_Medial,Meniscus_Lateral,ACL,PCL,MCL,LCL,Cartilage,Bone_Edema,Bone_Fracture,Effusion/Baker,Synovitis/Burs,PostOp,DEG-OA,NOR  
示例：  
ZMR-222222,1158右侧膝关节磁共振平扫,1,0,1,0,0,0,1,1,0,1,0,0,0c

请根据上述要求，对下面给定报告文本进行解析，并仅返回 CSV 格式的数据，每个检查项目输出两行数据：
第一行为文字描述行、第二行为对应的数字 label 行, 不要输出任何其他内容,必须严格按照：文字行在前、数字行在后 的顺序输出,
禁止多输出或少输出任何内容。
"""

SYSTEM_PROMPT_SPINE = """
你是 骨肌系统 Spine MRI 报告解析专家。

① 标签与列序  
0 DEG-DIS  椎间盘退变  
1 DEG-BUL  椎间盘膨出  
2 DEG-PRO  椎间盘突出/脱出  
3 DEG-SPO  椎体滑脱  
4 DEG-STE  椎管或椎间孔狭窄  
5 DEG-END  终板改变/许莫氏结节/Modic  
6 TRA-VCF  压缩性骨折/急性骨挫伤  
7 TRA-POST 术后改变/内固定/植骨融合   
8 NOR     正常或未见异常（若其它标签均为 0，则 NOR=1，否则 NOR=0）

② 抽取规则  
• 逐段落识别 “检查号＋检查项目”。  
• 对照，出现即置 1。  
• 未出现的标签置 0。  
• 同一检查项目只输出一行二进制结果。  

③ 输出  
仅 CSV，无标题或说明。  
列顺序：检查号,检查项目,DEG-DIS,DEG-BUL,DEG-PRO,DEG-SPO,DEG-STE,DEG-END,TRA-VCF,TRA-POST,NOR  
示例：  
ZMR-111111,3119腰椎磁共振平扫,1,0,1,0,1,0,0,0,0

请根据上述要求，对下面给定报告文本进行解析，并仅返回 CSV 格式的数据，每个检查项目输出两行数据：
第一行为文字描述行、第二行为对应的数字 label 行, 不要输出任何其他内容,必须严格按照：文字行在前、数字行在后 的顺序输出,
禁止多输出或少输出任何内容。
"""

SYSTEM_PROMPT_30 = """You are a board-certified radiologist specializing in thoracic imaging.  

Task
Classify the chest/abdomen CT report for the following 30 findings, in the exact order listed:
1. submucosal_edema
2. renal_hypodensities
3. aortic_valve_calcification
4. coronary_calcification
5. thrombosis
6. metastatic_disease
7. pancreatic_atrophy
8. renal_cyst
9. osteopenia
10. surgically_absent_gallbladder
11. atelectasis
12. abdominal_aortic_aneurysm
13. anasarca
14. hiatal_hernia
15. lymphadenopathy
16. prostatomegaly
17. biliary_ductal_dilation
18. cardiomegaly
19. splenomegaly
20. hepatomegaly
21. atherosclerosis
22. ascites
23. pleural_effusion
24. hepatic_steatosis
25. appendicitis
26. gallstones
27. hydronephrosis
28. bowel_obstruction
29. free_air
30. fracture

Labeling rules
- Output a single line of 30 comma-separated values.
- Use 1 if the finding is explicitly present.
- Use 0 if the finding is explicitly ruled out or absent.
- Use -1 if the report is ambiguous, uncertain, or lacks information.

Only return the comma-separated numbers.
Example:  
0,1,-1,-1,1,-1,-1,-1,-1,-1,0,0,0,0,1,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0
"""

SYSTEM_PROMPT_18 = """You are a board-certified radiologist specializing in thoracic imaging.  

Task
Classify the chest/abdomen CT report for the following 18 findings, in the exact order listed:
1. medical material
2. arterial wall calcification
3. cardiomegaly
4. pericardial effusion
5. coronary artery wall calcification
6. hiatal hernia
7. lymphadenopathy
8. emphysema
9. atelectasis
10. lung nodule
11. lung opacity
12. pulmonary fibrotic sequela
13. pleural effusion
14. mosaic attenuation pattern
15. peribronchial thickening
16. consolidation
17. bronchiectasis
18. interlobular septal thickening

Labeling rules
- Output a single line of 18 comma-separated values.
- Use 1 if the finding is explicitly present.
- Use 0 if the finding is explicitly ruled out or absent.
- Use -1 if the report is ambiguous, uncertain, or lacks information.

Only return the comma-separated numbers.
Example:  
0,1,-1,-1,1,-1,-1,-1,-1,-1,0,0,0,0,1,0,0,0
"""

class LLMLabeler:
    """
    Efficient Scalability: Asynchronous pipeline for extracting labels from reports.
    Corresponds to 'llm_pipeline' requirements.
    """
    def __init__(self, 
                 api_key: str, 
                 base_url: str, 
                 model: str = "doubao-pro-32k-241215",
                 concurrency: int = 24,
                 max_retry: int = 3,
                 mode: str = "30_class"):
        
        if AsyncOpenAI is None:
            raise ImportError("Please install `openai` package: pip install openai")
            
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retry = max_retry
        self.backoff_base = 2
        
        if mode == "30_class":
            self.system_prompt = SYSTEM_PROMPT_30
            self.labels = LABEL_COLUMNS_30
            self.num_classes = 30
        elif mode == "18_class":
            self.system_prompt = SYSTEM_PROMPT_18
            self.labels = LABEL_COLUMNS_18
            self.num_classes = 18
        else:
            raise ValueError(f"Unknown mode: {mode}")

    async def classify_one(self, row_index: int, report: str) -> Tuple[int, Optional[List[int]]]:
        async with self.sem:
            for attempt in range(1, self.max_retry + 1):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": f"\n\n{report}"}
                        ],
                        stream=False,
                    )
                    content = resp.choices[0].message.content.strip()
                    
                    # Parse tokens (robust to "0, 1, ..." spacing)
                    tokens = [tok.strip() for tok in content.split(",")]
                    
                    # Basic validation
                    if len(tokens) != self.num_classes:
                        # Try to handle cases where LLM outputs "1, 0, ... (Explanation)"
                         if len(tokens) > self.num_classes:
                              tokens = tokens[:self.num_classes]
                         else:
                            raise ValueError(f"Expected {self.num_classes} tokens, got {len(tokens)}")
                        
                    if any(tok not in {"0", "1", "-1"} for tok in tokens):
                         raise ValueError(f"Invalid tokens found: {content}")

                    labels = [int(tok) for tok in tokens]
                    return row_index, labels

                except Exception as e:
                    # print(f"[WARN] index={row_index} attempt={attempt}/{self.max_retry}: {e}")
                    if attempt == self.max_retry:
                        return row_index, None
                    await asyncio.sleep(self.backoff_base ** attempt)
        return row_index, None

    async def process_dataframe(self, df: pd.DataFrame, report_col: str) -> pd.DataFrame:
        """
        Processes a pandas DataFrame, adding label columns.
        """
        # Initialize columns
        for col in self.labels:
            if col not in df.columns:
                df[col] = pd.NA
                
        tasks = []
        # Filter for rows that need processing or process all? 
        # For now, process all where report exists.
        for idx, row in df.iterrows():
            report = str(row[report_col]) if pd.notna(row[report_col]) else ""
            if len(report.strip()) < 5: # Skip empty or too short
                continue
            tasks.append(asyncio.create_task(self.classify_one(idx, report)))
            
        print(f"Starting classification for {len(tasks)} reports...")
        results = await asyncio.gather(*tasks)
        
        success_count = 0
        for row_index, labels in results:
            if labels is None:
                continue
            success_count += 1
            for col, value in zip(self.labels, labels):
                df.at[row_index, col] = value
                
        print(f"Processed {len(tasks)} reports, {success_count} succeeded.")
        return df

def is_numeric_row(line: str) -> bool:
    """
    Check if a CSV row is primarily numeric (ignoring first 2 columns).
    """
    if not line or line.strip() == "":
        return False
        
    tokens = [t.strip() for t in line.split(",")]
    if len(tokens) <= 2:
        return False
    
    # Check from 3rd column onwards (index 2)
    data_tokens = tokens[2:]
    # Allow ints, floats, and -1
    numeric_tokens = [t for t in data_tokens if re.fullmatch(r"[-+]?\d+(\.\d+)?", t)]
    
    # If all data tokens are numeric and there is at least one data token
    return len(numeric_tokens) == len(data_tokens) and len(data_tokens) > 0

class MRExtractor:
    def __init__(self, 
                 api_key: str, 
                 base_url: str, 
                 model: str = "doubao-pro-32k-241215",
                 concurrency: int = 24,
                 max_retry: int = 3,
                 anatomy: str = "knee"):
        
        if AsyncOpenAI is None:
            raise ImportError("Please install `openai` package: pip install openai")
            
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retry = max_retry
        self.backoff_base = 2
        self.anatomy = anatomy
        
        if anatomy == "knee":
            self.system_prompt = SYSTEM_PROMPT_KNEE
        elif anatomy == "spine":
            self.system_prompt = SYSTEM_PROMPT_SPINE
        else:
            raise ValueError(f"Unknown anatomy: {anatomy}")

    async def parse_one(self, report_text: str, chk_no: str, item_name: str) -> Tuple[List[str], List[str]]:
        # Returns (text_lines, numeric_lines)
        if not report_text: return [], []

        async with self.sem:
            for attempt in range(1, self.max_retry + 1):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": report_text}
                        ],
                        stream=False
                    )
                    content = resp.choices[0].message.content.strip()
                    if not content:
                         raise ValueError("Empty response from LLM")

                    parts = [seg for seg in content.splitlines() if seg.strip()]
                    
                    text_lines = []
                    numeric_lines = []

                    # Process pairs
                    if len(parts) % 2 != 0:
                        raise ValueError(f"Expected even number of lines, got {len(parts)}")
                    
                    for i in range(0, len(parts), 2):
                        line1, line2 = parts[i].strip(), parts[i + 1].strip()
                        
                        is_l1_num = is_numeric_row(line1)
                        is_l2_num = is_numeric_row(line2)

                        if is_l1_num and is_l2_num:
                             raise ValueError(f"Both lines numeric: {line1} / {line2}")
                        if not is_l1_num and not is_l2_num:
                             raise ValueError(f"Neither line numeric: {line1} / {line2}")
                        
                        # Ensure Text -> Numeric order
                        if is_l1_num and not is_l2_num:
                            text_l, num_l = line2, line1
                        else:
                            text_l, num_l = line1, line2
                            
                        if is_numeric_row(text_l): raise ValueError("Text line identified as numeric")
                        if not is_numeric_row(num_l): raise ValueError("Numeric line identified as text")

                        text_lines.append(text_l)
                        numeric_lines.append(num_l)
                    
                    return text_lines, numeric_lines

                except Exception as e:
                    # print(f"Error {chk_no}: {e}")
                    if attempt == self.max_retry:
                        return [], []
                    await asyncio.sleep(self.backoff_base ** attempt)
        return [], []

    async def process_custom_dataframe(self, df: pd.DataFrame, 
                                       id_col: str = "检查号", 
                                       item_col: str = "检查项目", 
                                       text_cols: List[str] = ["检查所见", "诊断意见"],
                                       filter_ids: List[str] = None):
        tasks = []
        for idx, row in df.iterrows():
            chk_no = str(row.get(id_col, ""))
            if filter_ids is not None and chk_no not in filter_ids:
                continue
            
            item_name = str(row.get(item_col, ""))
            
            # Construct report
            parts = [f"检查号：{chk_no}", f"检查项目：{item_name}"]
            for c in text_cols:
                val = str(row.get(c, "")) if pd.notna(row.get(c)) else ""
                parts.append(f"{c}：{val}")
            
            report_text = "\\n".join(parts)
            
            tasks.append(asyncio.create_task(self.parse_one(report_text, chk_no, item_name)))

        print(f"Starting extraction for {len(tasks)} items ({self.anatomy})...")
        results = await asyncio.gather(*tasks)
        
        all_text = []
        all_numeric = []
        for t_lines, n_lines in results:
            all_text.extend(t_lines)
            all_numeric.extend(n_lines)
            
        print(f"Extraction complete: {len(all_numeric)} valid records found.")
        return all_text, all_numeric

