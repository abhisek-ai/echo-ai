# """
# Complete Inference Pipeline for EchoAI
# Combines sentiment analysis and response generation
# """
# import joblib
# import json
# import logging
# from typing import Dict, List, Optional, Union
# from datetime import datetime
# import pandas as pd
# from pathlib import Path

# from config import *
# from response_generator import ResponseGenerator

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class EchoAIInference:
#     """
#     Complete inference pipeline for review analysis and response generation
#     """
    
#     def __init__(self, 
#                  sentiment_model_path: Path = None,
#                  vectorizer_path: Path = None,
#                  llm_model: str = 'google/flan-t5-base'):
#         """
#         Initialize the inference pipeline
        
#         Args:
#             sentiment_model_path: Path to trained sentiment model
#             vectorizer_path: Path to TF-IDF vectorizer
#             llm_model: Name of the LLM model to use
#         """
#         self.sentiment_model_path = sentiment_model_path or BEST_MODEL_PATH
#         self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
#         self.llm_model_name = llm_model
        
#         self.sentiment_model = None
#         self.vectorizer = None
#         self.response_generator = None
        
#         self.sentiment_labels = ['negative', 'neutral', 'positive']
        
#         # Track performance metrics
#         self.inference_stats = {
#             'total_processed': 0,
#             'successful': 0,
#             'failed': 0,
#             'avg_confidence': 0
#         }
    
#     def load_models(self, load_llm: bool = True):
#         """
#         Load all required models
        
#         Args:
#             load_llm: Whether to load the LLM (can be skipped for sentiment-only)
#         """
#         logger.info("Loading models for inference...")
        
#         # Load sentiment analysis model
#         try:
#             self.sentiment_model = joblib.load(self.sentiment_model_path)
#             self.vectorizer = joblib.load(self.vectorizer_path)
#             logger.info("✓ Sentiment model loaded successfully")
#         except Exception as e:
#             logger.error(f"Failed to load sentiment model: {e}")
#             raise
        
#         # Load LLM for response generation
#         if load_llm:
#             try:
#                 self.response_generator = ResponseGenerator(self.llm_model_name)
#                 self.response_generator.load_model()
#                 logger.info("✓ Response generation model loaded successfully")
#             except Exception as e:
#                 logger.error(f"Failed to load LLM: {e}")
#                 logger.warning("Continuing without response generation")
#                 self.response_generator = None
    
#     def predict_sentiment(self, text: str) -> Dict:
#         """
#         Predict sentiment for a single review
        
#         Args:
#             text: Review text
            
#         Returns:
#             Dictionary with sentiment prediction and confidence
#         """
#         if not self.sentiment_model or not self.vectorizer:
#             raise ValueError("Sentiment model not loaded. Call load_models() first.")
        
#         try:
#             # Vectorize the text
#             text_tfidf = self.vectorizer.transform([text])
            
#             # Get prediction
#             prediction = self.sentiment_model.predict(text_tfidf)[0]
#             sentiment_label = self.sentiment_labels[prediction]
            
#             # Get confidence if available
#             confidence = None
#             if hasattr(self.sentiment_model, 'predict_proba'):
#                 probabilities = self.sentiment_model.predict_proba(text_tfidf)[0]
#                 confidence = float(max(probabilities))
                
#                 # Get probability for each class
#                 class_probabilities = {
#                     label: float(prob) 
#                     for label, prob in zip(self.sentiment_labels, probabilities)
#                 }
#             else:
#                 class_probabilities = {}
            
#             return {
#                 'sentiment': sentiment_label,
#                 'sentiment_score': int(prediction),
#                 'confidence': confidence,
#                 'probabilities': class_probabilities
#             }
            
#         except Exception as e:
#             logger.error(f"Error predicting sentiment: {e}")
#             raise
    
#     def generate_response(self, 
#                          text: str, 
#                          sentiment: str = None,
#                          business_category: str = None,
#                          rating: int = None,
#                          auto_detect_sentiment: bool = True) -> str:
#         """
#         Generate a response for a review
        
#         Args:
#             text: Review text
#             sentiment: Sentiment (if None, will be predicted)
#             business_category: Type of business
#             rating: Customer rating
#             auto_detect_sentiment: Whether to predict sentiment if not provided
            
#         Returns:
#             Generated response text
#         """
#         # Predict sentiment if not provided
#         if sentiment is None and auto_detect_sentiment:
#             sentiment_result = self.predict_sentiment(text)
#             sentiment = sentiment_result['sentiment']
        
#         if not self.response_generator:
#             logger.warning("Response generator not available")
#             return self._get_template_response(sentiment)
        
#         try:
#             response = self.response_generator.generate_response(
#                 text, 
#                 sentiment,
#                 business_category,
#                 rating
#             )
#             return response
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             return self._get_template_response(sentiment)
    
#     def _get_template_response(self, sentiment: str) -> str:
#         """Fallback template responses"""
#         templates = {
#             'positive': "Thank you for your positive feedback! We're delighted to hear about your experience and look forward to serving you again.",
#             'neutral': "Thank you for taking the time to share your feedback. We value your input and are always working to improve our service.",
#             'negative': "We sincerely apologize for your experience. Your feedback is important to us, and we'd like to make things right. Please contact us directly."
#         }
#         return templates.get(sentiment, templates['neutral'])
    
#     def process_review(self, 
#                        review: Union[str, Dict],
#                        generate_response: bool = True) -> Dict:
#         """
#         Process a single review through the complete pipeline
        
#         Args:
#             review: Review text or dict with review data
#             generate_response: Whether to generate a response
            
#         Returns:
#             Complete analysis results
#         """
#         # Parse input
#         if isinstance(review, str):
#             review_text = review
#             metadata = {}
#         else:
#             review_text = review.get('text', review.get('review', ''))
#             metadata = {
#                 k: v for k, v in review.items() 
#                 if k not in ['text', 'review']
#             }
        
#         # Start processing
#         result = {
#             'input': review_text,
#             'metadata': metadata,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         try:
#             # Step 1: Sentiment Analysis
#             sentiment_result = self.predict_sentiment(review_text)
#             result['sentiment_analysis'] = sentiment_result
            
#             # Step 2: Response Generation (if requested)
#             if generate_response:
#                 response = self.generate_response(
#                     review_text,
#                     sentiment_result['sentiment'],
#                     metadata.get('business_category'),
#                     metadata.get('rating')
#                 )
#                 result['generated_response'] = response
            
#             # Update stats
#             self.inference_stats['total_processed'] += 1
#             self.inference_stats['successful'] += 1
#             if sentiment_result.get('confidence'):
#                 self.inference_stats['avg_confidence'] = (
#                     (self.inference_stats['avg_confidence'] * 
#                      (self.inference_stats['successful'] - 1) +
#                      sentiment_result['confidence']) / 
#                     self.inference_stats['successful']
#                 )
            
#             result['status'] = 'success'
            
#         except Exception as e:
#             logger.error(f"Error processing review: {e}")
#             result['status'] = 'failed'
#             result['error'] = str(e)
#             self.inference_stats['failed'] += 1
        
#         return result
    
#     def process_batch(self, 
#                      reviews: List[Union[str, Dict]],
#                      generate_responses: bool = True,
#                      save_results: bool = True) -> List[Dict]:
#         """
#         Process multiple reviews in batch
        
#         Args:
#             reviews: List of reviews (text or dicts)
#             generate_responses: Whether to generate responses
#             save_results: Whether to save results to file
            
#         Returns:
#             List of processed results
#         """
#         logger.info(f"Processing batch of {len(reviews)} reviews...")
        
#         results = []
#         for i, review in enumerate(reviews, 1):
#             if i % 10 == 0:
#                 logger.info(f"Processing review {i}/{len(reviews)}")
            
#             result = self.process_review(review, generate_responses)
#             results.append(result)
        
#         # Save results if requested
#         if save_results:
#             self._save_batch_results(results)
        
#         # Print summary
#         self._print_batch_summary(results)
        
#         return results
    
#     def _save_batch_results(self, results: List[Dict]):
#         """Save batch processing results"""
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         output_file = RESULTS_DIR / f'inference_results_{timestamp}.json'
        
#         with open(output_file, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         logger.info(f"Results saved to {output_file}")
    
#     def _print_batch_summary(self, results: List[Dict]):
#         """Print summary of batch processing"""
#         successful = sum(1 for r in results if r.get('status') == 'success')
#         failed = len(results) - successful
        
#         sentiments = [
#             r['sentiment_analysis']['sentiment'] 
#             for r in results 
#             if 'sentiment_analysis' in r
#         ]
        
#         print("\n" + "="*60)
#         print("BATCH PROCESSING SUMMARY")
#         print("="*60)
#         print(f"Total Processed: {len(results)}")
#         print(f"Successful: {successful}")
#         print(f"Failed: {failed}")
        
#         if sentiments:
#             print("\nSentiment Distribution:")
#             for sentiment in self.sentiment_labels:
#                 count = sentiments.count(sentiment)
#                 percentage = (count / len(sentiments)) * 100
#                 print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
#         if self.inference_stats['avg_confidence'] > 0:
#             print(f"\nAverage Confidence: {self.inference_stats['avg_confidence']:.3f}")
        
#         print("="*60)
    
#     def interactive_mode(self):
#         """
#         Interactive mode for testing individual reviews
#         """
#         print("\n" + "🤖"*30)
#         print("     ECHOAI INTERACTIVE MODE")
#         print("🤖"*30)
#         print("\nEnter reviews to analyze and generate responses.")
#         print("Type 'quit' to exit, 'stats' for statistics.")
#         print("-"*60)
        
#         while True:
#             try:
#                 # Get user input
#                 print("\n📝 Enter a review (or command):")
#                 user_input = input("> ").strip()
                
#                 # Check for commands
#                 if user_input.lower() == 'quit':
#                     print("👋 Goodbye!")
#                     break
#                 elif user_input.lower() == 'stats':
#                     print(f"\n📊 Statistics:")
#                     print(f"  Processed: {self.inference_stats['total_processed']}")
#                     print(f"  Successful: {self.inference_stats['successful']}")
#                     print(f"  Failed: {self.inference_stats['failed']}")
#                     print(f"  Avg Confidence: {self.inference_stats['avg_confidence']:.3f}")
#                     continue
#                 elif not user_input:
#                     continue
                
#                 # Get optional metadata
#                 print("\n📋 Optional info (press Enter to skip):")
#                 category = input("  Business category (Restaurant/Hotel/Retail): ").strip() or None
#                 rating_str = input("  Rating (1-5): ").strip()
#                 rating = int(rating_str) if rating_str.isdigit() else None
                
#                 # Process review
#                 review_data = {
#                     'text': user_input,
#                     'business_category': category,
#                     'rating': rating
#                 }
                
#                 result = self.process_review(review_data, generate_response=True)
                
#                 # Display results
#                 print("\n" + "="*60)
#                 print("ANALYSIS RESULTS")
#                 print("="*60)
                
#                 if result['status'] == 'success':
#                     sentiment_data = result['sentiment_analysis']
#                     print(f"\n😊 Sentiment: {sentiment_data['sentiment'].upper()}")
#                     print(f"📊 Confidence: {sentiment_data.get('confidence', 'N/A'):.3f}")
                    
#                     if sentiment_data.get('probabilities'):
#                         print("\n📈 Probabilities:")
#                         for label, prob in sentiment_data['probabilities'].items():
#                             bar = '█' * int(prob * 20)
#                             print(f"  {label:8} [{bar:20}] {prob:.3f}")
                    
#                     if 'generated_response' in result:
#                         print(f"\n💬 Generated Response:")
#                         print(f"  {result['generated_response']}")
#                 else:
#                     print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
#             except KeyboardInterrupt:
#                 print("\n👋 Goodbye!")
#                 break
#             except Exception as e:
#                 print(f"❌ Error: {e}")
#                 continue

# def main():
#     """Main function to demonstrate the inference pipeline"""
#     import argparse
    
#     parser = argparse.ArgumentParser(description='EchoAI Inference Pipeline')
#     parser.add_argument('--mode', choices=['interactive', 'batch', 'demo'], 
#                        default='demo', help='Running mode')
#     parser.add_argument('--input', type=str, help='Input file for batch mode')
#     parser.add_argument('--llm', type=str, default='google/flan-t5-base',
#                        help='LLM model to use')
#     parser.add_argument('--no-response', action='store_true',
#                        help='Skip response generation')
#     args = parser.parse_args()
    
#     # Initialize pipeline
#     pipeline = EchoAIInference(llm_model=args.llm)
#     pipeline.load_models(load_llm=not args.no_response)
    
#     if args.mode == 'interactive':
#         # Interactive mode
#         pipeline.interactive_mode()
        
#     elif args.mode == 'batch':
#         # Batch mode
#         if not args.input:
#             print("Error: --input required for batch mode")
#             return
        
#         # Load reviews from file
#         if args.input.endswith('.json'):
#             with open(args.input, 'r') as f:
#                 reviews = json.load(f)
#         elif args.input.endswith('.csv'):
#             df = pd.read_csv(args.input)
#             reviews = df.to_dict('records')
#         else:
#             print("Error: Input file must be JSON or CSV")
#             return
        
#         # Process batch
#         results = pipeline.process_batch(
#             reviews, 
#             generate_responses=not args.no_response
#         )
        
#     else:
#         # Demo mode
#         demo_reviews = [
#             {
#                 'text': "Absolutely loved the ambiance and the food was to die for! Best restaurant in town!",
#                 'business_category': 'Restaurant',
#                 'rating': 5
#             },
#             {
#                 'text': "The service was slow and the food was cold. Very disappointing experience.",
#                 'business_category': 'Restaurant', 
#                 'rating': 2
#             },
#             {
#                 'text': "Nice place, decent food. Nothing special but not bad either.",
#                 'business_category': 'Restaurant',
#                 'rating': 3
#             }
#         ]
        
#         print("\n🎭 DEMO MODE - Processing sample reviews")
#         print("="*60)
        
#         for i, review in enumerate(demo_reviews, 1):
#             print(f"\n📝 Review {i}:")
#             print(f"   {review['text']}")
            
#             result = pipeline.process_review(review)
            
#             if result['status'] == 'success':
#                 print(f"😊 Sentiment: {result['sentiment_analysis']['sentiment']}")
#                 print(f"📊 Confidence: {result['sentiment_analysis'].get('confidence', 0):.3f}")
#                 if 'generated_response' in result:
#                     print(f"💬 Response: {result['generated_response']}")
#             print("-"*60)

# if __name__ == "__main__":
#     main()

#-----------------------------------------------------------------------------------------------------------------------------------------
"""
Complete Inference Pipeline for EchoAI (Without LLM)
Sentiment analysis only - template-based responses
"""
import joblib
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EchoAIInference:
    """
    Inference pipeline for sentiment analysis
    Uses template-based responses (no LLM required)
    """
    
    def __init__(self, 
                 sentiment_model_path: Path = None,
                 vectorizer_path: Path = None):
        """
        Initialize the inference pipeline
        
        Args:
            sentiment_model_path: Path to trained sentiment model
            vectorizer_path: Path to TF-IDF vectorizer
        """
        self.sentiment_model_path = sentiment_model_path or BEST_MODEL_PATH
        self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
        
        self.sentiment_model = None
        self.vectorizer = None
        
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Track performance metrics
        self.inference_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0
        }
    
    def load_models(self):
        """Load sentiment analysis model"""
        logger.info("Loading models for inference...")
        
        # Load sentiment analysis model
        try:
            self.sentiment_model = joblib.load(self.sentiment_model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def predict_sentiment(self, text: str) -> Dict:
        """
        Predict sentiment for a single review
        
        Args:
            text: Review text
            
        Returns:
            Dictionary with sentiment prediction and confidence
        """
        if not self.sentiment_model or not self.vectorizer:
            raise ValueError("Sentiment model not loaded. Call load_models() first.")
        
        try:
            # Vectorize the text
            text_tfidf = self.vectorizer.transform([text])
            
            # Get prediction
            prediction = self.sentiment_model.predict(text_tfidf)[0]
            sentiment_label = self.sentiment_labels[prediction]
            
            # Get confidence if available
            confidence = None
            if hasattr(self.sentiment_model, 'predict_proba'):
                probabilities = self.sentiment_model.predict_proba(text_tfidf)[0]
                confidence = float(max(probabilities))
                
                # Get probability for each class
                class_probabilities = {
                    label: float(prob) 
                    for label, prob in zip(self.sentiment_labels, probabilities)
                }
            else:
                class_probabilities = {}
            
            return {
                'sentiment': sentiment_label,
                'sentiment_score': int(prediction),
                'confidence': confidence,
                'probabilities': class_probabilities
            }
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            raise
    
    def generate_response(self, 
                         text: str, 
                         sentiment: str = None,
                         business_category: str = None,
                         rating: int = None) -> str:
        """
        Generate a template-based response for a review
        
        Args:
            text: Review text
            sentiment: Sentiment (if None, will be predicted)
            business_category: Type of business
            rating: Customer rating
            
        Returns:
            Generated response text
        """
        # Predict sentiment if not provided
        if sentiment is None:
            sentiment_result = self.predict_sentiment(text)
            sentiment = sentiment_result['sentiment']
        
        # Get template-based response
        return self._get_template_response(sentiment, business_category, rating)
    
    def _get_template_response(self, sentiment: str, business_category: str = None, rating: int = None) -> str:
        """Generate template-based responses"""
        
        # Category-specific phrases
        category_phrases = {
            'Restaurant': 'dining experience',
            'Hotel': 'stay with us',
            'Retail': 'shopping experience',
            'Healthcare': 'visit to our facility',
            'Automotive': 'service experience'
        }
        
        experience_phrase = category_phrases.get(business_category, 'experience')
        
        templates = {
            'positive': [
                f"Thank you so much for your wonderful review! We're thrilled to hear about your positive {experience_phrase}. We look forward to serving you again soon!",
                f"We're delighted that you had such a great {experience_phrase}! Your kind words mean the world to us. Thank you for choosing us!",
                f"What fantastic feedback! We're so happy you enjoyed your {experience_phrase}. We can't wait to welcome you back!"
            ],
            'neutral': [
                f"Thank you for taking the time to share your feedback about your {experience_phrase}. We appreciate your honest review and are always working to improve our service.",
                f"We value your input regarding your {experience_phrase}. Thank you for bringing this to our attention. We hope to serve you better in the future.",
                f"Thank you for your feedback. We're committed to enhancing our service and appreciate you sharing your thoughts about your {experience_phrase}."
            ],
            'negative': [
                f"We sincerely apologize for your disappointing {experience_phrase}. Your feedback is important to us, and we'd like to make things right. Please contact us directly so we can address your concerns.",
                f"We're very sorry to hear about the issues you encountered during your {experience_phrase}. This is not the standard we hold ourselves to. We'd love the opportunity to restore your faith in our service.",
                f"Thank you for bringing this to our attention, and we apologize for falling short during your {experience_phrase}. Your satisfaction is our priority, and we'd like to resolve this for you. Please reach out to us."
            ]
        }
        
        # Select response based on rating if available
        response_templates = templates.get(sentiment, templates['neutral'])
        
        if rating:
            if rating == 5:
                return response_templates[0]  # Most enthusiastic
            elif rating == 1:
                return response_templates[2]  # Most apologetic
        
        # Default: use middle template
        return response_templates[1]
    
    def process_review(self, 
                       review: Union[str, Dict],
                       generate_response: bool = True) -> Dict:
        """
        Process a single review through the complete pipeline
        
        Args:
            review: Review text or dict with review data
            generate_response: Whether to generate a response
            
        Returns:
            Complete analysis results
        """
        # Parse input
        if isinstance(review, str):
            review_text = review
            metadata = {}
        else:
            review_text = review.get('text', review.get('review', ''))
            metadata = {
                k: v for k, v in review.items() 
                if k not in ['text', 'review']
            }
        
        # Start processing
        result = {
            'input': review_text,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Sentiment Analysis
            sentiment_result = self.predict_sentiment(review_text)
            result['sentiment_analysis'] = sentiment_result
            
            # Step 2: Response Generation (if requested)
            if generate_response:
                response = self.generate_response(
                    review_text,
                    sentiment_result['sentiment'],
                    metadata.get('business_category'),
                    metadata.get('rating')
                )
                result['generated_response'] = response
            
            # Update stats
            self.inference_stats['total_processed'] += 1
            self.inference_stats['successful'] += 1
            if sentiment_result.get('confidence'):
                self.inference_stats['avg_confidence'] = (
                    (self.inference_stats['avg_confidence'] * 
                     (self.inference_stats['successful'] - 1) +
                     sentiment_result['confidence']) / 
                    self.inference_stats['successful']
                )
            
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing review: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            self.inference_stats['failed'] += 1
        
        return result
    
    def process_batch(self, 
                     reviews: List[Union[str, Dict]],
                     generate_responses: bool = True,
                     save_results: bool = True) -> List[Dict]:
        """
        Process multiple reviews in batch
        
        Args:
            reviews: List of reviews (text or dicts)
            generate_responses: Whether to generate responses
            save_results: Whether to save results to file
            
        Returns:
            List of processed results
        """
        logger.info(f"Processing batch of {len(reviews)} reviews...")
        
        results = []
        for i, review in enumerate(reviews, 1):
            if i % 10 == 0:
                logger.info(f"Processing review {i}/{len(reviews)}")
            
            result = self.process_review(review, generate_responses)
            results.append(result)
        
        # Save results if requested
        if save_results:
            self._save_batch_results(results)
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _save_batch_results(self, results: List[Dict]):
        """Save batch processing results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = RESULTS_DIR / f'inference_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _print_batch_summary(self, results: List[Dict]):
        """Print summary of batch processing"""
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful
        
        sentiments = [
            r['sentiment_analysis']['sentiment'] 
            for r in results 
            if 'sentiment_analysis' in r
        ]
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if sentiments:
            print("\nSentiment Distribution:")
            for sentiment in self.sentiment_labels:
                count = sentiments.count(sentiment)
                percentage = (count / len(sentiments)) * 100
                print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        if self.inference_stats['avg_confidence'] > 0:
            print(f"\nAverage Confidence: {self.inference_stats['avg_confidence']:.3f}")
        
        print("="*60)
    
    def interactive_mode(self):
        """
        Interactive mode for testing individual reviews
        """
 
        print("     ECHOAI INTERACTIVE MODE (Template Responses)")
     
        print("\nEnter reviews to analyze and generate responses.")
        print("Type 'quit' to exit, 'stats' for statistics.")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                print("\nEnter a review (or command):")
                user_input = input("> ").strip()
                
                # Check for commands
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'stats':
                    print(f"\nStatistics:")
                    print(f"  Processed: {self.inference_stats['total_processed']}")
                    print(f"  Successful: {self.inference_stats['successful']}")
                    print(f"  Failed: {self.inference_stats['failed']}")
                    print(f"  Avg Confidence: {self.inference_stats['avg_confidence']:.3f}")
                    continue
                elif not user_input:
                    continue
                
                # Get optional metadata
                print("\nOptional info (press Enter to skip):")
                category = input("  Business category (Restaurant/Hotel/Retail): ").strip() or None
                rating_str = input("  Rating (1-5): ").strip()
                rating = int(rating_str) if rating_str.isdigit() else None
                
                # Process review
                review_data = {
                    'text': user_input,
                    'business_category': category,
                    'rating': rating
                }
                
                result = self.process_review(review_data, generate_response=True)
                
                # Display results
                print("\n" + "="*60)
                print("ANALYSIS RESULTS")
                print("="*60)
                
                if result['status'] == 'success':
                    sentiment_data = result['sentiment_analysis']
                    print(f"\nSentiment: {sentiment_data['sentiment'].upper()}")
                    if sentiment_data.get('confidence'):
                        print(f"Confidence: {sentiment_data['confidence']:.3f}")
                    
                    if sentiment_data.get('probabilities'):
                        print("\nProbabilities:")
                        for label, prob in sentiment_data['probabilities'].items():
                            bar = '*' * int(prob * 20)
                            print(f"  {label:8} [{bar:20}] {prob:.3f}")

                    if 'generated_response' in result:
                        print(f"\nGenerated Response:")
                        print(f"  {result['generated_response']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

def main():
    """Main function to demonstrate the inference pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EchoAI Inference Pipeline (No LLM)')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'demo'], 
                       default='demo', help='Running mode')
    parser.add_argument('--input', type=str, help='Input file for batch mode')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EchoAIInference()
    pipeline.load_models()
    
    if args.mode == 'interactive':
        # Interactive mode
        pipeline.interactive_mode()
        
    elif args.mode == 'batch':
        # Batch mode
        if not args.input:
            print("Error: --input required for batch mode")
            return
        
        # Load reviews from file
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                reviews = json.load(f)
        elif args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            reviews = df.to_dict('records')
        else:
            print("Error: Input file must be JSON or CSV")
            return
        
        # Process batch
        results = pipeline.process_batch(reviews, generate_responses=True)
        
    else:
        # Demo mode
        demo_reviews = [
            {
                'text': "Absolutely loved the ambiance and the food was to die for! Best restaurant in town!",
                'business_category': 'Restaurant',
                'rating': 5
            },
            {
                'text': "The service was slow and the food was cold. Very disappointing experience.",
                'business_category': 'Restaurant', 
                'rating': 2
            },
            {
                'text': "Nice place, decent food. Nothing special but not bad either.",
                'business_category': 'Restaurant',
                'rating': 3
            }
        ]
        
        print("\nDEMO MODE - Processing sample reviews")
        print("="*60)
        
        for i, review in enumerate(demo_reviews, 1):
            print(f"\nReview {i}:")
            print(f"   {review['text']}")
            
            result = pipeline.process_review(review)
            
            if result['status'] == 'success':
                print(f"Sentiment: {result['sentiment_analysis']['sentiment']}")
                if result['sentiment_analysis'].get('confidence'):
                    print(f"Confidence: {result['sentiment_analysis']['confidence']:.3f}")
                if 'generated_response' in result:
                    print(f"Response: {result['generated_response']}")
            print("-"*60)

if __name__ == "__main__":
    main()