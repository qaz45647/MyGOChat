from typing import Dict, List, Tuple
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

class MyGOChat:
    """
    MyGoChat is a dialogue prediction class that returns relevant quotes and image paths
    based on input text using a pre-trained transformer model.
    """
    
    def __init__(
        self,
        model_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'),        
        label_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'Label_Path.json'),
        max_length: int = 512
    ) -> None:
        """
        Initialize the MyGoChat instance.
        
        Args:
            model_path (str): Path to the pre-trained model directory
            label_path (str): Path to the JSON file containing quotes and image paths
            max_length (int): Maximum sequence length for input text
        
        Raises:
            FileNotFoundError: If model_path or label_path doesn't exist
            RuntimeError: If CUDA is available but fails to initialize
        """
        self.logger = self._setup_logger()
        self.max_length = max_length
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label path not found: {label_path}")
            
        # Load label mapping
        self.idx_to_label = self._load_label_mapping(model_path)
        self.quotes_data = self._load_quotes_data(label_path)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize model and tokenizer
        self.model, self.tokenizer = self._initialize_model(model_path)
        
        self.logger.info("MyGoChat initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('MyGoChat')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_label_mapping(self, model_path: str) -> Dict[int, str]:
        """Load and process label mapping from JSON."""
        mapping_path = os.path.join(model_path, 'label_mapping.json')
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            return {int(k): v for k, v in mapping['idx_to_label'].items()}
        except Exception as e:
            self.logger.error(f"Failed to load label mapping: {e}")
            raise
    
    def _load_quotes_data(self, label_path: str) -> Dict[str, Dict[str, str]]:
        """Load quotes and image paths from JSON."""
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {item['title']: {
                'quote': item['title'],
                'image_url': item['Image_Path']
            } for item in data}
        except Exception as e:
            self.logger.error(f"Failed to load quotes data: {e}")
            raise
    
    def _setup_device(self) -> torch.device:
        """Setup and validate torch device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")
        return device
    
    def _initialize_model(self, model_path: str) -> tuple:
        """Initialize the transformer model and tokenizer."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise

    def _get_top_predictions(self, probabilities: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top k predictions with their probabilities.
        
        Args:
            probabilities (torch.Tensor): Probability distribution over labels
            k (int): Number of top predictions to return
            
        Returns:
            List[Tuple[str, float]]: List of (label, probability) tuples
        """
        top_k = torch.topk(probabilities, k=min(k, len(self.idx_to_label)))
        return [
            (self.idx_to_label[idx.item()], prob.item())
            for idx, prob in zip(top_k.indices[0], top_k.values[0])
        ]
    
    def chat_with_candidates(self, text: str, k: int = 5) -> Dict:
        """
        Process input text and return top k predictions with corresponding quotes and image URLs.
        
        Args:
            text (str): Input text to process
            k (int): Number of top predictions to return
            
        Returns:
            Dict: Dictionary containing:
                - 'top_prediction': Dict with 'quote' and 'image_url' for the best match
                - 'candidates': List of dicts with 'label', 'probability', 'quote', and 'image_url'
            
        Raises:
            ValueError: If input text is empty or too long
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform prediction
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get top k predictions
            top_predictions = self._get_top_predictions(probabilities, k)
            
            # Prepare response
            candidates = []
            for label, prob in top_predictions:
                quote_data = self.quotes_data.get(label, {
                    'quote': 'No matching quote found',
                    'image_url': ''
                })
                candidates.append({
                    'label': label,
                    'probability': round(prob * 100, 2),  # Convert to percentage
                    'quote': quote_data['quote'],
                    'image_url': quote_data['image_url']
                })
            
            response = {
                'top_prediction': candidates[0],
                'candidates': candidates
            }
            
            self.logger.debug(f"Input: {text}, Top predictions: {candidates}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in chat_with_candidates: {e}")
            raise
    
    def chat(self, text: str) -> Dict[str, str]:
        """
        Process input text and return corresponding quote and image URL.
        
        Args:
            text (str): Input text to process
            
        Returns:
            Dict[str, str]: Dictionary containing 'quote' and 'image_url'
            
        Raises:
            ValueError: If input text is empty or too long
        """
        response = self.chat_with_candidates(text, k=1)
        return {
            'quote': response['top_prediction']['quote'],
            'image_url': response['top_prediction']['image_url']
        }
