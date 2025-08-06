"""
LLM Service for Text Response Analysis and Categorization
Handles category identification and response categorization using LiteLLM
"""

import os
import json
import re
import litellm
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st
from pydantic import BaseModel

class LiteLLMProxyConfig(BaseModel):
    """Configuration for LiteLLM Standard Proxy mode"""
    api_key: str  # LiteLLM API key
    base_url: str  # LiteLLM proxy base URL
    model_name: str  # Alias model name (e.g., lite_llm_alias/model-name)
    temperature: float = 0.1  # Lower temperature for consistent categorization
    max_tokens: Optional[int] = 2000
    timeout: int = 60

class LiteLLMConfig:
    """LiteLLM Standard Proxy configuration manager"""
    
    # Supported model aliases for LiteLLM Standard Proxy
    SUPPORTED_MODELS = {
        "openai": [
            "openai/o4-mini",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-nano",
            "openai/gpt-4o-mini"
        ],
        "gemini": [
            "gemini/gemini-2.5-flash-lite",
            "gemini/gemini-2.5-flash"
        ]
        # "anthropic": [
        #     "anthropic/claude-4-sonnet-20250514"
        # ]
    }
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiteLLMConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        self._initialized = True
        self.setup_litellm()
    
    def setup_litellm(self):
        """Configure LiteLLM settings for Standard Proxy mode"""
        # Set LiteLLM logging level
        litellm.set_verbose = os.getenv("LITELLM_VERBOSE", "False").lower() == "true"
        
        # Configure timeout
        litellm.request_timeout = int(os.getenv("LITELLM_TIMEOUT", "60"))
        
        # Configure number of retries
        litellm.num_retries = int(os.getenv("LITELLM_RETRIES", "3"))
    
    def get_current_config(self) -> LiteLLMProxyConfig:
        """Get the current LiteLLM configuration from environment variables"""
        # Get API key with fallback to OPENAI_API_KEY for backward compatibility
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY or OPENAI_API_KEY environment variable is required")
        
        # Get base URL (required for LiteLLM Standard Proxy)
        base_url = os.getenv("LLM_API_BASE")
        if not base_url:
            raise ValueError("LLM_API_BASE environment variable is required for LiteLLM Standard Proxy")
        
        # Get model name with default
        model_name = os.getenv("LLM_MODEL_NAME", "openai/gpt-4.1-mini")
        
        # Get temperature with default
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        
        # Get max tokens (optional)
        max_tokens = os.getenv("LLM_MAX_TOKENS")
        
        # Get timeout with default
        timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        
        return LiteLLMProxyConfig(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens else None,
            timeout=timeout
        )
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get a dictionary of supported models by provider"""
        return self.SUPPORTED_MODELS

# Global configuration instance (lazy initialization)
_litellm_config = None

def get_litellm_config():
    """Get the global LiteLLM configuration instance (lazy initialization)"""
    global _litellm_config
    if _litellm_config is None:
        _litellm_config = LiteLLMConfig()
    return _litellm_config

class LazyLiteLLMConfig:
    """Lazy loading wrapper for LiteLLM config"""
    def __getattr__(self, name):
        return getattr(get_litellm_config(), name)
    
    def __call__(self):
        return get_litellm_config()

# Create a global instance for backward compatibility
litellm_config = LazyLiteLLMConfig()

class PromptManager:
    """Manages loading and formatting of prompt templates"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        
    def load_prompt(self, prompt_name: str, **kwargs) -> str:
        """Load and format a prompt template"""
        try:
            prompt_path = self.prompts_dir / f"{prompt_name}.md"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                
            # Format the prompt with any provided kwargs
            if kwargs:
                return prompt.format(**kwargs)
            return prompt
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template '{prompt_name}' not found in {self.prompts_dir}")
        except Exception as e:
            raise Exception(f"Error loading prompt '{prompt_name}': {str(e)}")

class LLMService:
    def __init__(self, api_key: str = None, prompts_dir: str = "prompts"):
        """
        Initialize LLM service with LiteLLM configuration and prompt manager
        
        Args:
            api_key: API key (if None, will use LLM_API_KEY or OPENAI_API_KEY environment variable)
            prompts_dir: Directory containing prompt template files
        """
        # Initialize prompt manager
        self.prompt_manager = PromptManager(prompts_dir)
        
        # Set environment variables if API key is provided
        if api_key:
            if api_key.startswith("sk-") and not os.getenv("OPENAI_API_KEY"):
                # If it looks like an OpenAI key and OPENAI_API_KEY is not set
                os.environ["OPENAI_API_KEY"] = api_key
            elif not os.getenv("LLM_API_KEY"):
                # Otherwise use as LiteLLM API key if not already set
                os.environ["LLM_API_KEY"] = api_key
        
        # Initialize LiteLLM configuration
        self.litellm_config = get_litellm_config()
        try:
            self.config = self.litellm_config.get_current_config()
        except Exception as e:
            st.error(f"Failed to initialize LiteLLM configuration: {str(e)}")
            raise
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, messages: List[Dict], max_tokens: int = 4000) -> str:
        """
        Make API call using the OpenAI client with LiteLLM proxy
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Maximum number of tokens to generate (not used in current implementation)
            
        Returns:
            str: The generated response content
            
        Raises:
            Exception: If the API call fails after retries
        """
        try:
            # Get the current configuration
            config = self.litellm_config.get_current_config()
            
            # Import the OpenAI client
            import openai
            
            # Configure the OpenAI client to use the LiteLLM proxy
            client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url if config.base_url else None
            )
            
            # Make the API call with the correct model name format
            # Remove the 'openai/' prefix if present
            model_name = config.model_name.replace('openai/', '') if config.model_name.startswith('openai/') else config.model_name
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            
            # Extract the response content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("Unexpected response format from LLM API")
                
        except Exception as e:
            st.error(f"LLM API Error: {str(e)}")
            st.error(f"Model: {getattr(config, 'model_name', 'N/A')}")
            st.error(f"API Base: {getattr(config, 'base_url', 'N/A')}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            raise

    def identify_categories(self, responses: List[str], question_text: str, num_categories: int = 10) -> Dict:
        """
        Identify categories from ALL responses using the identify_categories prompt template
        
        Args:
            responses: List of response strings
            question_text: The survey question text
            num_categories: Number of categories to identify (default: 10)
            
        Returns:
            Dictionary with categories and their descriptions
        """
        try:
            # Load the prompt template
            prompt = self.prompt_manager.load_prompt(
                "identify_categories",
                question_text=question_text,
                responses="\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)]),
                num_categories=num_categories
            )
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": "You are an expert at analyzing survey responses and identifying meaningful categories."},
                {"role": "user", "content": prompt}
            ]
            
            # Call the LLM
            response = self._call_llm(messages)
            
            # Parse the JSON response
            try:
                categories = json.loads(response)
                return categories
            except json.JSONDecodeError:
                st.error("Failed to parse LLM response as JSON. Using default categories.")
                return self._get_default_categories()
                
        except Exception as e:
            st.error(f"Error identifying categories: {str(e)}")
            return self._get_default_categories()
    
    def export_categories_csv(self, categories: Dict, include_subcategories: bool = True) -> str:
        """
        Export categories in CSV format required by admin portal
        
        Args:
            categories: Dictionary with categories from identify_categories
            include_subcategories: Whether to create subcategories for detailed analysis
            
        Returns:
            CSV string in format: parent_bucket_name,bucket_name,is_active
        """
        csv_lines = ["parent_bucket_name,bucket_name,is_active"]
        
        for category_name, category_info in categories.items():
            if isinstance(category_info, dict) and 'description' in category_info:
                description = category_info['description']
            else:
                description = str(category_info)
            
            # Add main category
            csv_lines.append(f"{category_name},{category_name},1")
            
            # Add subcategories if requested
            if include_subcategories:
                subcategories = self._generate_subcategories(description)
                for subcat in subcategories:
                    csv_lines.append(f"{category_name},{subcat},1")
        
        return "\n".join(csv_lines)
    
    def _generate_subcategories(self, description: str) -> List[str]:
        """
        Generate subcategories from category description
        """
        # Simple subcategory generation based on common patterns
        subcategories = []
        
        # Add positive/negative sentiment subcategories
        subcategories.extend([
            f"Positive",
            f"Negative",
            f"Neutral"
        ])
        
        return subcategories[:3]  # Limit to 3 subcategories
    
    def process_admin_portal_csv(self, csv_file_path: str) -> Dict:
        """
        Process CSV output from admin portal to extract categorized responses
        
        Args:
            csv_file_path: Path to CSV file from admin portal
            
        Returns:
            Dictionary with categorized responses and available categories
        """
        try:
            import pandas as pd
            
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Expected columns: users_waves,question_id,voice_state,responses,responses_original,is_local_outlier,[category columns...]
            required_columns = ['users_waves', 'question_id', 'responses']
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Find category columns (columns with binary 0/1 values)
            category_columns = []
            for col in df.columns:
                if col not in required_columns + ['voice_state', 'responses_original', 'is_local_outlier']:
                    # Check if column contains only 0/1 values
                    unique_values = df[col].dropna().unique()
                    if set(unique_values).issubset({0, 1, '0', '1'}):
                        category_columns.append(col)
            
            # Extract categorized responses
            categorized_responses = {}
            for category in category_columns:
                # Get responses where this category is marked as 1
                category_responses = df[df[category].isin([1, '1'])]['responses'].tolist()
                categorized_responses[category] = category_responses
            
            return {
                'categorized_responses': categorized_responses,
                'available_categories': category_columns,
                'total_responses': len(df),
                'dataframe': df
            }
            
        except Exception as e:
            st.error(f"Error processing admin portal CSV: {str(e)}")
            raise
    
    def extract_top_quotes_from_csv(self, admin_portal_data: Dict, selected_category: str = None) -> Dict:
        """
        Extract and score quotes from admin portal categorized responses
        
        Args:
            admin_portal_data: Data from process_admin_portal_csv
            selected_category: Specific category to extract quotes for (optional)
            
        Returns:
            Dictionary with scored quotes per category
        """
        try:
            categorized_responses = admin_portal_data['categorized_responses']
            categories_to_process = [selected_category] if selected_category else admin_portal_data['available_categories']
            
            scored_quotes = {}
            
            for category in categories_to_process:
                if category in categorized_responses:
                    responses = categorized_responses[category]
                    
                    if len(responses) == 0:
                        scored_quotes[category] = {
                            "scored_quotes": [],
                            "category_summary": {
                                "total_responses": 0,
                                "average_score": 0,
                                "top_themes": []
                            }
                        }
                        continue
                    
                    # Load the prompt template for quote extraction and scoring
                    try:
                        prompt = self.prompt_manager.load_prompt(
                            "extract_quotes",
                            category_name=category,
                            responses="\n".join([f"{i+1}. {response}" for i, response in enumerate(responses)])
                        )
                    except FileNotFoundError:
                        # Fallback prompt if template doesn't exist
                        prompt = f"""Score and rank the following responses for the category "{category}" based on relevance, clarity, insight value, and professional perspective (0-100 scale):

{chr(10).join([f"{i+1}. {response}" for i, response in enumerate(responses)])}

Return all responses with scores in JSON format, sorted by score (highest first)."""
                    
                    # Create messages for the LLM
                    messages = [
                        {"role": "system", "content": "You are an expert at scoring and ranking survey responses for pharmaceutical market research based on relevance, clarity, insight value, and professional perspective."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Call the LLM
                    response = self._call_llm(messages)
                    
                    # Parse the response
                    try:
                        quotes_data = json.loads(response)
                        
                        # Ensure we have the expected structure
                        if 'scored_quotes' in quotes_data:
                            scored_quotes[category] = quotes_data
                        else:
                            # Handle legacy format or unexpected structure
                            scored_quotes[category] = {
                                "scored_quotes": quotes_data if isinstance(quotes_data, list) else [],
                                "category_summary": {
                                    "total_responses": len(responses),
                                    "average_score": 0,
                                    "top_themes": []
                                }
                            }
                            
                    except json.JSONDecodeError:
                        # Fallback: create scored entries for all responses
                        fallback_quotes = []
                        for i, resp in enumerate(responses):
                            fallback_quotes.append({
                                "response_id": f"response_{i+1}",
                                "quote": resp,
                                "total_score": 50,  # Neutral score
                                "score_breakdown": {
                                    "relevance": 15,
                                    "clarity": 12,
                                    "insight_value": 12,
                                    "professional_perspective": 11
                                },
                                "bonus_points": 0,
                                "rationale": "Fallback scoring - manual review recommended"
                            })
                        
                        scored_quotes[category] = {
                            "scored_quotes": fallback_quotes,
                            "category_summary": {
                                "total_responses": len(responses),
                                "average_score": 50,
                                "top_themes": ["Manual review needed"]
                            }
                        }
            
            return scored_quotes
            
        except Exception as e:
            st.error(f"Error extracting and scoring quotes: {str(e)}")
            return {}
    
    def extract_top_quotes(self, categorized_responses: Dict, categories: Dict, responses_df) -> Dict:
        """
        Legacy method - Extract top 3 most compelling quotes for each category
        
        Args:
            categorized_responses: Results from categorize_responses_batch
            categories: Category definitions
            responses_df: Original responses dataframe
            
        Returns:
            Dictionary with top quotes per category
        """
        # This is a legacy method for backward compatibility
        return self.extract_top_quotes_from_csv({
            'categorized_responses': categorized_responses,
            'available_categories': list(categories.keys())
        })
    
    def _get_default_categories(self) -> Dict:
        """
        Return default pharmaceutical categories if LLM fails
        """
        return {
            "Efficacy": {"description": "Comments about how well the treatment works"},
            "Side Effects": {"description": "Mentions of adverse reactions or side effects"},
            "Convenience": {"description": "Comments about ease of use, dosing, administration"},
            "Cost": {"description": "Mentions of price, affordability, insurance coverage"},
            "Quality of Life": {"description": "Impact on daily activities and overall well-being"},
            "Healthcare Provider": {"description": "Comments about doctors, nurses, or medical staff"},
            "Access": {"description": "Availability, prescription process, pharmacy issues"},
            "Comparison": {"description": "Comparisons with other treatments or medications"},
            "General Satisfaction": {"description": "Overall positive or negative sentiment"},
            "Other": {"description": "Responses that don't fit other categories"}
        }
