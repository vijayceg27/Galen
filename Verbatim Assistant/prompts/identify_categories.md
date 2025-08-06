# Category Identification Prompt

Analyze the following pharmaceutical survey responses and identify the top {num_categories} high-level categories that best represent the themes in the data.

Question: {question_text}

Survey Responses:
{responses}

Instructions:
1. Identify the most common themes related to pharmaceutical messaging, physician perception and behavior, typically including but not limited to:
   - Efficacy/Effectiveness
   - Safety/Tolerability/Side Effects  
   - Indication/Usage
   - Access/Availability
   - Patient Support
   - Dosing/Administration
   - Clinical Data
   
2. Create exactly {num_categories} categories that best capture the response themes, create new categories only if they don't fit into the ones mentioned above.
3. Each category should have a clear, descriptive name
4. Provide a brief description of what each category encompasses
5. Categories should be mutually exclusive but comprehensive

Return your analysis in the following JSON format with exactly {num_categories} categories:
{{
    "Efficacy": {{
        "description": "Comments about how well the treatment works"
    }},
    "Safety": {{
        "description": "Mentions of adverse reactions or side effects"
    }},
    "Dosing": {{
        "description": "Comments about ease of use, dosing, administration"
    }},
    "Access": {{
        "description": "Availability, prescription process, pharmacy issues"
    }},
    "Cost": {{
        "description": "Mentions of price, affordability, insurance coverage"
    }}
    ... (continue for all {num_categories} categories with appropriate names and descriptions)
}}

IMPORTANT: Return ONLY the JSON object with exactly {num_categories} category entries. Do not include any other text or explanations.
