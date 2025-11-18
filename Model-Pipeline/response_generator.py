"""
Response Generator using Open-Source LLM
Generates contextual responses based on sentiment analysis
"""
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Hugging Face Transformers for open-source LLMs
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
        BitsAndBytesConfig
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Please install transformers: pip install transformers torch")

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates review responses using open-source LLMs
    Supports multiple models with different capabilities
    """
    
    # Available open-source models (ordered by size/performance)
    AVAILABLE_MODELS = {
        'microsoft/DialoGPT-medium': {
            'type': 'conversational',
            'size': 'medium',
            'description': 'Good for conversational responses'
        },
        'google/flan-t5-base': {
            'type': 'text2text',
            'size': 'base',
            'description': 'Efficient instruction-following model'
        },
        'google/flan-t5-large': {
            'type': 'text2text',
            'size': 'large',
            'description': 'Better quality, larger model'
        },
        'facebook/blenderbot-400M-distill': {
            'type': 'conversational',
            'size': 'medium',
            'description': 'Specialized in dialogue'
        },
        'mistralai/Mistral-7B-Instruct-v0.1': {
            'type': 'instruct',
            'size': 'large',
            'description': 'High quality but requires GPU',
            'requires_gpu': True
        }
    }
    
    def __init__(self, model_name='google/flan-t5-base', device='auto'):
        """
        Initialize the response generator
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: 'cuda', 'cpu', or 'auto'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Response templates by sentiment
        self.response_templates = {
            'positive': {
                'intro': [
                    "Thank you so much for your wonderful review!",
                    "We're thrilled to hear about your positive experience!",
                    "Your kind words mean the world to us!"
                ],
                'closing': [
                    "We look forward to serving you again soon!",
                    "Thank you for choosing us!",
                    "We can't wait to welcome you back!"
                ]
            },
            'neutral': {
                'intro': [
                    "Thank you for taking the time to share your feedback.",
                    "We appreciate your honest review.",
                    "Thank you for your feedback."
                ],
                'closing': [
                    "We hope to serve you better in the future.",
                    "Please don't hesitate to reach out if you have any concerns.",
                    "We value your input and will work to improve."
                ]
            },
            'negative': {
                'intro': [
                    "We sincerely apologize for your disappointing experience.",
                    "We're very sorry to hear about the issues you encountered.",
                    "Thank you for bringing this to our attention, and we apologize for falling short."
                ],
                'closing': [
                    "Please contact us directly so we can make this right.",
                    "We'd love the opportunity to restore your faith in our service.",
                    "Your satisfaction is our priority, and we'd like to resolve this for you."
                ]
            }
        }
    
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self, use_8bit=False):
        """
        Load the LLM model and tokenizer
        
        Args:
            use_8bit: Use 8-bit quantization for large models (reduces memory)
        """
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            model_info = self.AVAILABLE_MODELS.get(self.model_name, {})
            model_type = model_info.get('type', 'text2text')
            
            # Configure for different model types
            if model_type == 'text2text':
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map='auto' if self.device == 'cuda' else None
                )
                
            elif model_type == 'conversational':
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
            elif model_type == 'instruct':
                # For larger models like Mistral
                if use_8bit and self.device == 'cuda':
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map='auto'
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map='auto' if self.device == 'cuda' else None
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Move to device if not using device_map
            if self.device == 'cpu' and hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            # Create pipeline for easy generation
            self.generator = pipeline(
                'text-generation' if model_type != 'text2text' else 'text2text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_prompt(self, review_text: str, sentiment: str, 
                     business_category: str = None, rating: int = None) -> str:
        """
        Create a prompt for the LLM based on review and sentiment
        
        Args:
            review_text: The customer review
            sentiment: Predicted sentiment (positive/neutral/negative)
            business_category: Type of business (Restaurant/Hotel/Retail)
            rating: Customer rating (1-5)
        """
        # Build context
        context_parts = []
        
        if business_category:
            context_parts.append(f"Business Type: {business_category}")
        if rating:
            context_parts.append(f"Customer Rating: {rating}/5")
        context_parts.append(f"Sentiment: {sentiment}")
        
        context = " | ".join(context_parts)
        
        # Create prompt based on model type
        if 'flan' in self.model_name.lower():
            # Flan-T5 style prompt
            prompt = f"""Generate a professional business response to this customer review.
Context: {context}
Review: "{review_text}"
Instructions: Write a personalized, empathetic response that:
1. Acknowledges the customer's feedback
2. {"Expresses gratitude and reinforces positive aspects" if sentiment == 'positive' 
   else "Apologizes and offers to resolve issues" if sentiment == 'negative'
   else "Thanks them and shows commitment to improvement"}
3. Is concise (2-3 sentences) and professional
Response:"""
        
        elif 'mistral' in self.model_name.lower():
            # Mistral instruction format
            prompt = f"""[INST] You are a professional customer service representative responding to reviews.
Context: {context}
Customer Review: "{review_text}"
Generate a professional, empathetic response that is 2-3 sentences long. 
{"Express gratitude and encourage repeat business." if sentiment == 'positive'
  else "Apologize sincerely and offer to resolve the issue." if sentiment == 'negative'
  else "Thank them for feedback and show commitment to improvement."}[/INST]"""
        
        else:
            # Generic format
            prompt = f"""Business Response Generator
{context}
Customer Review: {review_text}
Professional Response:"""
        
        return prompt
    
    def generate_response(self, review_text: str, sentiment: str,
                         business_category: str = None, rating: int = None,
                         use_template: bool = True,
                         max_length: int = 150,
                         temperature: float = 0.7) -> str:
        """
        Generate a response to a review
        
        Args:
            review_text: The customer review
            sentiment: Predicted sentiment
            business_category: Type of business
            rating: Customer rating
            use_template: Whether to use response templates
            max_length: Maximum response length
            temperature: Generation temperature (0-1, higher = more creative)
        """
        if not self.generator:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create prompt
        prompt = self.create_prompt(review_text, sentiment, business_category, rating)
        
        try:
            # Generate response
            if 'flan' in self.model_name.lower():
                # T5 models
                result = self.generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1
                )
                generated_text = result[0]['generated_text']
            
            else:
                # Causal LM models
                result = self.generator(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                # Extract only the generated part
                generated_text = result[0]['generated_text'].replace(prompt, '').strip()
            
            # Add template wrapping if requested
            if use_template:
                generated_text = self._add_template_wrapper(generated_text, sentiment)
            
            # Clean up the response
            generated_text = self._clean_response(generated_text)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback to template-based response
            return self._get_fallback_response(sentiment)
    
    def _add_template_wrapper(self, generated_text: str, sentiment: str) -> str:
        """Add intro and closing based on sentiment"""
        import random
        
        templates = self.response_templates.get(sentiment, self.response_templates['neutral'])
        
        # Sometimes add intro/closing for variety
        if random.random() > 0.3:  # 70% chance to add wrapper
            intro = random.choice(templates['intro'])
            closing = random.choice(templates['closing'])
            return f"{intro} {generated_text} {closing}"
        
        return generated_text
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the generated response"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper capitalization
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Limit length to avoid overly long responses
        sentences = text.split('. ')
        if len(sentences) > 4:
            text = '. '.join(sentences[:4]) + '.'
        
        return text
    
    def _get_fallback_response(self, sentiment: str) -> str:
        """Get a template-based fallback response"""
        import random
        
        fallback_responses = {
            'positive': "Thank you for your wonderful review! We're delighted to hear about your positive experience and look forward to serving you again soon.",
            'neutral': "Thank you for taking the time to share your feedback. We value your input and continuously strive to improve our service.",
            'negative': "We sincerely apologize for your disappointing experience. Please contact us directly so we can address your concerns and make things right."
        }
        
        return fallback_responses.get(sentiment, fallback_responses['neutral'])
    
    def batch_generate_responses(self, reviews: List[Dict], batch_size: int = 8) -> List[str]:
        """
        Generate responses for multiple reviews
        
        Args:
            reviews: List of dicts with 'text', 'sentiment', etc.
            batch_size: Number of reviews to process at once
        """
        responses = []
        
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            
            for review in batch:
                response = self.generate_response(
                    review.get('text', ''),
                    review.get('sentiment', 'neutral'),
                    review.get('business_category'),
                    review.get('rating')
                )
                responses.append(response)
        
        return responses
    
    def save_model_cache(self, cache_dir: Path = None):
        """Save model to local cache for faster loading"""
        if cache_dir is None:
            cache_dir = MODEL_DIR / 'llm_cache'
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model and self.tokenizer:
            model_cache = cache_dir / self.model_name.replace('/', '_')
            self.model.save_pretrained(model_cache)
            self.tokenizer.save_pretrained(model_cache)
            logger.info(f"Model cached at {model_cache}")

def test_response_generator():
    """Test the response generator with sample reviews"""
    
    # Sample reviews with different sentiments
    test_reviews = [
        {
            'text': "The food was absolutely amazing and the service was outstanding!",
            'sentiment': 'positive',
            'business_category': 'Restaurant',
            'rating': 5
        },
        {
            'text': "The room was okay but nothing special. Average experience overall.",
            'sentiment': 'neutral',
            'business_category': 'Hotel',
            'rating': 3
        },
        {
            'text': "Terrible service, cold food, and overpriced. Very disappointed.",
            'sentiment': 'negative',
            'business_category': 'Restaurant',
            'rating': 1
        }
    ]
    
    # Initialize generator
    generator = ResponseGenerator(model_name='google/flan-t5-base')
    generator.load_model()
    
    # Generate responses
    print("\n" + "="*60)
    print("TESTING RESPONSE GENERATOR")
    print("="*60)
    
    for review in test_reviews:
        print(f"\n📝 Review: {review['text']}")
        print(f"😊 Sentiment: {review['sentiment']}")
        print(f"⭐ Rating: {review['rating']}/5")
        print(f"🏢 Category: {review['business_category']}")
        
        response = generator.generate_response(
            review['text'],
            review['sentiment'],
            review['business_category'],
            review['rating']
        )
        
        print(f"💬 Generated Response: {response}")
        print("-"*60)

if __name__ == "__main__":
    test_response_generator()