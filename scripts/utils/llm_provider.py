#!/usr/bin/env python3
import logging
from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path

# Import LLM client libraries
import openai
from openai import OpenAI, AzureOpenAI
import anthropic

# Import the TokenTracker
from .token_tracker import TokenTracker

class LLMProvider:
    """Handle different LLM provider interactions with token usage tracking."""
    
    def __init__(self, config: Dict, model_key: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration dictionary containing model settings
            model_key: Optional model key to use (for deepinfra provider)
        """
        self.config = config
        self.provider = config['model_settings']['active_provider']
        
        # Handle the case for deepinfra with multiple models
        if self.provider == 'deepinfra':
            provider_config = config['model_settings']['deepinfra']
            
            # Use the provided model_key or fall back to the active_model in config
            active_model_key = model_key or provider_config.get('active_model', 'qwen')
            
            # Ensure the model key exists
            if active_model_key not in provider_config['models']:
                logging.warning(f"Model key '{active_model_key}' not found in deepinfra models. "
                              f"Falling back to '{provider_config['active_model']}'.")
                active_model_key = provider_config['active_model']
                
            # Get the specific model configuration
            self.model_config = provider_config['models'][active_model_key]
            self.model_key = active_model_key
        # Handle the case for ollama with multiple models
        elif self.provider == 'ollama':
            provider_config = config['model_settings']['ollama']
            
            # Use the provided model_key or fall back to the active_model in config
            active_model_key = model_key or provider_config.get('active_model', 'llama3')
            
            # Ensure the model key exists
            if active_model_key not in provider_config['models']:
                logging.warning(f"Model key '{active_model_key}' not found in ollama models. "
                              f"Falling back to '{provider_config['active_model']}'.")
                active_model_key = provider_config['active_model']
                
            # Get the specific model configuration
            self.model_config = provider_config['models'][active_model_key]
            self.model_key = active_model_key
        else:
            # For other providers, use the traditional structure
            self.model_config = config['model_settings'][self.provider]
            self.model_key = None
            
        self.client = self._initialize_client()
        
        # Initialize the token tracker
        self.token_tracker = TokenTracker()

    def _initialize_client(self) -> Any:
        """
        Initialize the appropriate LLM client based on the active provider.
        
        Returns:
            The initialized client object
        """
        if self.provider == 'deepinfra':
            api_key = self.model_config.get('api_key')
            if not api_key:
                raise ValueError(f"DeepInfra API key is missing for model {self.model_key}. Please add it to your config file.")
            
            return OpenAI(
                api_key=api_key,
                base_url=self.model_config['base_url']
            )
        elif self.provider == 'openai':
            api_key = self.model_config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key is missing. Please add it to your config file.")
            
            return OpenAI(api_key=api_key)
        elif self.provider == 'claude':
            api_key = self.model_config.get('api_key')
            if not api_key:
                raise ValueError("Claude API key is missing. Please add it to your config file.")
            
            return anthropic.Anthropic(api_key=api_key)
        elif self.provider == 'azureopenai':
            api_key = self.model_config.get('api_key')
            if not api_key:
                raise ValueError("Azure OpenAI API key is missing. Please add it to your config file.")
            
            return AzureOpenAI(
                azure_endpoint=self.model_config['azure_endpoint'],
                api_key=api_key,
                api_version=self.model_config.get('api_version', "2024-08-01-preview")
            )
        elif self.provider == 'ollama':
            api_key = self.model_config.get('api_key')
            if not api_key:
                logging.warning("Ollama API key is missing, using 'ollama' as default.")
                api_key = 'ollama'
            
            return OpenAI(
                base_url=self.model_config['base_url'],
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_completion(self, system_prompt: str, user_input: str) -> Tuple[str, Dict[str, int]]:
        """
        Generate a completion based on the provider and return a tuple:
            (content, token_usage)
        where token_usage is a dict with keys 'input_tokens' and 'output_tokens'.
        
        Args:
            system_prompt: The system prompt to use
            user_input: The user input to process
            
        Returns:
            Tuple[str, Dict[str, int]]: The generated content and token usage information
        """
        response = None
        content = ""
        
        try:
            if self.provider in ['deepinfra', 'openai', 'azureopenai', 'ollama']:
                response = self.client.chat.completions.create(
                    model=self.model_config['model_name'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=self.model_config['max_tokens'],
                    temperature=self.model_config.get('temperature', 0),
                )
                content = response.choices[0].message.content

            elif self.provider == 'claude':
                response = self.client.messages.create(
                    model=self.model_config['model_name'],
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=self.model_config['max_tokens'],
                    temperature=self.model_config.get('temperature', 0)
                )
                content = response.content[0].text

        except Exception as e:
            logging.error(f"Error generating completion with {self.provider} "
                        f"(model: {self.model_config['model_name']}): {e}")
            raise
        
        # Extract or estimate token usage
        token_usage = self.token_tracker.extract_tokens_from_response(
            response, 
            self.provider,
            system_prompt=system_prompt,
            user_input=user_input,
            response_text=content,
            model_name=self.model_config['model_name']
        )
        
        return content, token_usage
           
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict with provider, model_key, and model_name
        """
        return {
            'provider': self.provider,
            'model_key': self.model_key,
            'model_name': self.model_config['model_name']
        }


# ==============================================================================
# Factory Function for Easy Initialization
# ==============================================================================

def get_llm_provider(
    config_path: str,
    provider_name: Optional[str] = None,
    model_alias: Optional[str] = None
) -> LLMProvider:
    """
    Factory function to create and return an LLMProvider instance.
    
    Args:
        config_path: Path to the LLM configuration JSON file
        provider_name: Optional provider name to override config (e.g., 'deepinfra', 'openai')
        model_alias: Optional model alias to use (e.g., 'qwen', 'llama3')
        
    Returns:
        Initialized LLMProvider instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If provider or model configuration is invalid
        
    Example:
        >>> provider = get_llm_provider(
        ...     config_path='config_llm.json',
        ...     provider_name='deepinfra',
        ...     model_alias='qwen'
        ... )
        >>> response, tokens = provider.generate_completion(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_input="Hello!"
        ... )
    """
    # Load configuration
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"LLM config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Override active provider if specified
    if provider_name:
        if provider_name not in ['deepinfra', 'openai', 'claude', 'azureopenai', 'ollama']:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        config['model_settings']['active_provider'] = provider_name
        logging.info(f"Overriding active provider to: {provider_name}")
    
    # Create and return provider
    provider = LLMProvider(config, model_key=model_alias)
    
    model_info = provider.get_model_info()
    logging.info(
        f"Initialized LLM provider: {model_info['provider']} / "
        f"{model_info['model_key'] or 'default'} / {model_info['model_name']}"
    )
    
    return provider


def validate_llm_config(config_path: str) -> Dict[str, Any]:
    """
    Validate LLM configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict with validation results
        
    Example:
        >>> results = validate_llm_config('config_llm.json')
        >>> if results['valid']:
        ...     print("Config is valid!")
    """
    results = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'providers_found': []
    }
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            results['errors'].append(f"Config file not found: {config_path}")
            return results
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check structure
        if 'model_settings' not in config:
            results['errors'].append("Missing 'model_settings' key in config")
            return results
        
        model_settings = config['model_settings']
        
        if 'active_provider' not in model_settings:
            results['errors'].append("Missing 'active_provider' in model_settings")
            return results
        
        active_provider = model_settings['active_provider']
        
        # Check each provider
        for provider in ['deepinfra', 'openai', 'claude', 'azureopenai', 'ollama']:
            if provider in model_settings:
                results['providers_found'].append(provider)
                
                provider_config = model_settings[provider]
                
                # Check for API key
                if provider in ['deepinfra', 'ollama']:
                    # These have multiple models
                    if 'models' not in provider_config:
                        results['warnings'].append(f"{provider}: Missing 'models' section")
                    else:
                        for model_key, model_config in provider_config['models'].items():
                            if not model_config.get('api_key'):
                                results['warnings'].append(
                                    f"{provider}/{model_key}: Missing API key"
                                )
                else:
                    if not provider_config.get('api_key'):
                        results['warnings'].append(f"{provider}: Missing API key")
        
        # Check active provider exists
        if active_provider not in results['providers_found']:
            results['errors'].append(
                f"Active provider '{active_provider}' not configured"
            )
        
        # If no errors, mark as valid
        if not results['errors']:
            results['valid'] = True
        
    except json.JSONDecodeError as e:
        results['errors'].append(f"Invalid JSON: {e}")
    except Exception as e:
        results['errors'].append(f"Validation error: {e}")
    
    return results