# Quote Extraction and Scoring Prompt

Analyze the following responses categorized under "{category_name}" and score each response based on relevance, clarity, and insight value for pharmaceutical market research.

Category: {category_name}

Responses:
{responses}

Scoring Criteria (0-100 scale):
1. **Relevance (0-30 points)**: How well does the response fit the category theme?
2. **Clarity (0-25 points)**: Is the response clear, specific, and well-articulated?
3. **Insight Value (0-25 points)**: Does it provide actionable insights for pharma research?
4. **Professional Perspective (0-20 points)**: Does it reflect healthcare professional viewpoint?

Additional Factors:
- Specific examples or details (+5 bonus points)
- Unique or unexpected insights (+5 bonus points)
- Actionable recommendations (+5 bonus points)

Return ALL responses with scores in JSON format, sorted by score (highest first):
{{
    "scored_quotes": [
        {{
            "response_id": "response_index_or_id",
            "quote": "exact response text",
            "total_score": 85,
            "score_breakdown": {{
                "relevance": 28,
                "clarity": 22,
                "insight_value": 20,
                "professional_perspective": 15
            }},
            "bonus_points": 5,
            "rationale": "Brief explanation of why this response scored high/low and what makes it valuable"
        }}
    ],
    "category_summary": {{
        "total_responses": 0,
        "average_score": 0,
        "top_themes": ["theme1", "theme2", "theme3"]
    }}
}}

IMPORTANT: Score ALL responses provided, not just the top ones. Sort by total_score in descending order.
