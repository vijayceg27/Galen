# Survey Data Crosstab Generator

Automated tool to generate comprehensive crosstabs from survey data with intelligent segmentation, question type detection, and professional formatting.

## Overview

Reads survey data from Excel and automatically generates banner-style crosstabs with:
- Overall statistics + segmented breakdowns side-by-side
- Smart question type detection (categorical, Likert scales, numeric inputs, array questions, multi-select)
- Sub-question handling for multi-part questions
- Answer value mapping for labeled responses
- Professional formatting with color-coded headers and percentage formatting
- Standardized layout across all question types

**Segment Variables:** SAMPLE_TYPE, S2, SETTING

## Key Features

✅ **Banner-Style Layout** - All segments in one table  
✅ **Smart Question Detection** - Handles different question types automatically  
✅ **Answer Value Mapping** - Shows "5 = Alabama" instead of just "5"  
✅ **Array Questions** - Separate crosstabs for each sub-question  
✅ **Multi-Select Questions** - Each option analyzed separately  
✅ **Standardized Layout** - Consistent column alignment across all questions  
✅ **Percentage Formatting** - Shows 18.7% instead of 18.7  
✅ **Column Order** - Percentage/Mean before (n) for easier scanning  
✅ **Layout Tab Order** - Questions appear in same order as Layout tab  
✅ **Non-Destructive** - Creates a copy, original unchanged  
✅ **Timestamped Output** - Never overwrites previous results  

## Requirements

- Python 3.8+
- Excel file with **Data** tab (raw responses) and **Layout** tab (question metadata)

### Layout Tab Structure
- **Column B**: Column Label (matches Data tab headers)
- **Column G**: Sub Question Text
- **Column H**: Answer Text
- **Column I**: Question ID
- **Column N onwards**: Answer Values (e.g., "5 = Alabama", "6 = Alaska", etc.)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Interactive Mode:**
```bash
python generate_crosstabs.py
```

**Command Line:**
```bash
python generate_crosstabs.py "path/to/file.xlsx"
python generate_crosstabs.py "input.xlsx" "output.xlsx"
```

**Output:** `OriginalName_with_Crosstabs_YYYYMMDD_HHMMSS.xlsx`

## Question Type Detection

### 1. Single-Select Questions
Shows labeled responses with percentages
- **Answer value mapping**: "5 = Alabama" instead of just "5"
- **Segment mapping**: "SAMPLE_TYPE_1 = Onlist" instead of just "1"
- Percentage shown before (n) for easier scanning
- Empty first column for alignment

### 2. Multi-Select Questions (e.g., A4)
Each option gets its own crosstab
- Shows 0 = No, 1 = Yes distribution per option
- Answer value mapping applied to both responses and segments

### 3. Array Questions (e.g., A6, S7)
Separate crosstabs for each sub-question
- **Smart text selection**: Uses Answer Text when sub-question text is generic
- Example: S7 shows "Glioblastoma" instead of "# of patients"
- All answer options with labels

### 4. Numeric Input Questions (e.g., S3)
Shows mean values and sample sizes
- True numeric inputs with no predefined options
- Mean displayed in (%) column for consistency

### 5. Questions with Sub-Questions (e.g., S5, S11)
Combined table with all sub-questions
- Sub-Question column identifies each part
- Consistent layout across question types

## Output Format

**Standardized Layout:**
- Questions WITHOUT sub-questions: Empty column A, data starts column B
- Questions WITH sub-questions: Sub-Question in column A, data starts column B
- **Column order**: Percentage/Mean BEFORE (n) counts
- **Percentage formatting**: Shows 18.7% not 18.7
- **Questions ordered**: Same order as Layout tab

## Output Examples

**Single-Select with Answer Labels (State - S1):**
```
        | Response        | Overall (%) | Overall (n) | SAMPLE_TYPE_1 = Onlist (%) | SAMPLE_TYPE_1 = Onlist (n)
        | 5 = Alabama     | 18.7%       | 14          | 11.1%                      | 2
        | 6 = Alaska      | 2.7%        | 2           | 0.0%                       | 0
        | Total           | 100.0%      | 75          | 100.0%                     | 18
```

**Array Question (Familiarity - A6):**
```
Question A6: How familiar are you with each...
  Vorasidenib
        | Response                                      | Overall (%) | Overall (n)
        | 3 = Familiar with it, but not yet planning   | 4.0%        | 3
        | 4 = Planning to use, but have not yet had... | 18.7%       | 14
        | Total                                         | 100.0%      | 75
  
  Olutasidenib
        | Response                                      | Overall (%) | Overall (n)
        | 2 = Heard of it, but don't know anything...  | 1.3%        | 1
        | Total                                         | 100.0%      | 75
```

**Numeric with Sub-Questions (Years - S3):**
```
        | Metric | Overall (%) | Overall (n) | SAMPLE_TYPE_1 = Onlist (%) | SAMPLE_TYPE_1 = Onlist (n)
        | Mean   | 15.3        | 75          | 13.8                       | 18
```

**Array with Generic Sub-Question Text (Tumor Types - S7):**
```
Question S7: Of the patients you've treated...
  Glioblastoma
        | Metric | Overall (%) | Overall (n)
        | Mean   | 97.8        | 75
  
  Anaplastic astrocytoma
        | Metric | Overall (%) | Overall (n)
        | Mean   | 69.5        | 75
```

**Combined Sub-Questions (Settings - S5):**
```
Sub-Question            | Metric | Overall (%) | Overall (n)
Academic medical center | Mean   | 25.3        | 75
Community hospital      | Mean   | 45.7        | 75
Private practice        | Mean   | 18.5        | 75
```

## Customization

**Change segments** (line 36):
```python
self.segment_questions = ['YOUR', 'SEGMENT', 'IDS']
```

## Troubleshooting

- **Column not found**: Check Column B (Layout) matches Data tab headers
- **Segment not found**: Verify SAMPLE_TYPE, S2, SETTING exist in both tabs
- **Missing crosstabs**: Ensure questions have valid responses in Data tab

## Files

- `generate_crosstabs.py` - Main script
- `requirements.txt` - Dependencies  
- `README.md` - This file

---

**Status:** ✅ Production Ready
