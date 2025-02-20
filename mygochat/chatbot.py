from typing import Dict
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

        self.model_path = model_path
        self.label_path = label_path

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
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            # Predict label
            predicted_label = self._predict(text)
            
            # Get corresponding quote and image
            response = self.quotes_data.get(predicted_label, {
                'quote': 'No matching quote found',
                'image_url': ''
            })
            
            self.logger.debug(f"Input: {text}, Predicted label: {predicted_label}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise
    
    def _predict(self, text: str) -> str:
        """
        Predict the label for input text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted label
        """
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
                predicted_idx = torch.argmax(probabilities, dim=1).item()
        
        return self.idx_to_label[predicted_idx]