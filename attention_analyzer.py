import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaModel
from torch import nn
import json
from typing import List, Dict, Tuple
import os
import random
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PoliticalSpeechClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', output_attentions=True)
        
        # Unfreeze more layers since we have more data
        for param in self.roberta.encoder.layer[-8:].parameters():
            param.requires_grad = True
        
        hidden_size = self.roberta.config.hidden_size
        
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.emotional_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self.political_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Collect attention weights
        attention_weights = outputs.attentions
        
        # Use mean pooling instead of just [CLS] token
        token_embeddings = outputs.last_hidden_state
        attention_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_expanded, 1)
        sum_mask = torch.clamp(attention_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Get shared features
        shared_features = self.shared_features(pooled_output)
        
        # Get task-specific predictions
        emotional_logits = self.emotional_classifier(shared_features)
        political_logits = self.political_classifier(shared_features)
        
        return emotional_logits, political_logits, attention_weights

class AttentionAnalyzer:
    def __init__(self, 
                 issue_model_path: str,
                 axis_model_path: str,
                 device: str = None):
        """Initialize with both model paths"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading models...")
        self.issue_model = BertForSequenceClassification.from_pretrained(
            issue_model_path,
            output_attentions=True
        )
        self.issue_model.to(self.device)
        self.issue_model.eval()
        
        self.axis_model = self.load_axis_model(axis_model_path)
        self.axis_model.to(self.device)
        self.axis_model.eval()
        
        # Load tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained(issue_model_path)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
    def load_axis_model(self, model_path: str):
        """Load the axis prediction model"""
        model_state = torch.load(model_path, map_location=self.device)
        model = PoliticalSpeechClassifier()
        model.load_state_dict(model_state['model_state_dict'])
        return model
    
    def get_issue_model_attention(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Get attention weights from issue classification model"""
        encoding = self.bert_tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.issue_model(**inputs)
            
            # Get attention weights from last layer
            attention = outputs.attentions[-1]
            attention = attention.mean(dim=1) # Average over attention heads
            
            # Get tokens for visualization
            tokens = self.bert_tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            
        return attention[0], tokens
    
    def get_axis_model_attention(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Get attention weights from axis classification model"""
        encoding = self.roberta_tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in encoding.items()}
            outputs = self.axis_model(**inputs)
            
            # Get attention weights
            attention_weights = outputs[2]
            attention = attention_weights[-1] # Get last layer
            attention = attention.mean(dim=1) # Average over attention heads
            
            # Get tokens for visualization
            tokens = self.roberta_tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
            
        return attention[0], tokens

    def plot_attention_heatmap(self, attention: torch.Tensor, tokens: List[str], 
                                title: str, save_path: str):
            """Create attention heatmap visualization"""
            # Convert attention to numpy and get relevant tokens
            attention_matrix = attention.cpu().numpy()
            
            # Remove padding tokens and special tokens
            mask = [t not in ['[PAD]', '[CLS]', '[SEP]', '<s>', '</s>', '<pad>'] for t in tokens]
            filtered_attention = attention_matrix[mask][:, mask]
            filtered_tokens = [self.clean_token(t) for i, t in enumerate(tokens) if mask[i]]

            
            # Create heatmap
            plt.figure(figsize=(15, 10))
            sns.heatmap(filtered_attention, 
                    xticklabels=filtered_tokens,
                    yticklabels=filtered_tokens,
                    cmap='YlOrRd')
            
            plt.title(title, fontsize=16)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
    
    def plot_attention_summary(self, attention: torch.Tensor, tokens: List[str],
                             title: str, save_path: str):
        """Create summary of attention distribution across text"""
        # Get average attention received by each token
        attention_scores = attention.mean(dim=0).cpu().numpy()
        
        # Remove padding and special tokens
        mask = [t not in ['[PAD]', '[CLS]', '[SEP]', '<s>', '</s>', '<pad>'] for t in tokens]
        filtered_scores = attention_scores[mask]
        filtered_tokens = [self.clean_token(t) for i, t in enumerate(tokens) if mask[i]]
        
        # Create bar plot
        plt.figure(figsize=(15, 6))
        positions = np.arange(len(filtered_tokens))
        plt.bar(positions, filtered_scores)
        
        # Add token labels
        plt.xticks(positions[::5], filtered_tokens[::5], rotation=45, ha='right')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Position in Text', fontsize=14)
        plt.ylabel('Average Attention', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def analyze_speech(self, speech_text: str, save_dir: str):
        """Analyze attention patterns for both models on a single speech"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Analyze issue model attention
        issue_attention, issue_tokens = self.get_issue_model_attention(speech_text)
        self.plot_attention_heatmap(
            issue_attention, 
            issue_tokens,
            'Issue Classification Model Attention',
            f"{save_dir}/issue_attention_heatmap.png"
        )
        self.plot_attention_summary(
            issue_attention,
            issue_tokens,
            'Issue Classification Model - Attention Distribution',
            f"{save_dir}/issue_attention_summary.png"
        )
        
        # Analyze axis model attention
        axis_attention, axis_tokens = self.get_axis_model_attention(speech_text)
        self.plot_attention_heatmap(
            axis_attention,
            axis_tokens,
            'Political/Emotional Axis Model Attention',
            f"{save_dir}/axis_attention_heatmap.png"
        )
        self.plot_attention_summary(
            axis_attention,
            axis_tokens,
            'Political/Emotional Axis Model - Attention Distribution',
            f"{save_dir}/axis_attention_summary.png"
        )
    def clean_token(self, token: str) -> str:
        """Clean up token for display"""
        # Remove the Ġ prefix (used by RoBERTa)
        if token.startswith('Ġ'):
            return token[1:]
        return token
    
def main():
    ISSUE_MODEL_PATH = "issue_classifier_eval/model/saved_issue_model"
    AXIS_MODEL_PATH = "large-training-output/model_artifacts_20241202_142615/model.pt"
    
    # Initialize analyzer
    analyzer = AttentionAnalyzer(ISSUE_MODEL_PATH, AXIS_MODEL_PATH)
    
    # Load example speeches
    with open("issue_classifier_eval/model/speeches_111_gpt_topic_labels.json", 'r') as f:
        speeches = json.load(f)
    
    # Filter speeches by length
    appropriate_speeches = []
    for speech_id, speech_data in speeches.items():
        word_count = len(speech_data['speech'].split())
        if 200 <= word_count <= 400:
            appropriate_speeches.append((speech_id, speech_data['speech']))
    
    # Take first 3 appropriate speeches
    selected_speeches = appropriate_speeches[:3]
    
    # Create output directory
    os.makedirs('attention_analysis', exist_ok=True)
    
    # Analyze selected speeches
    for i, (speech_id, speech_text) in enumerate(selected_speeches, 1):
        print(f"\nAnalyzing speech {i}...")
        print(f"Speech ID: {speech_id}")
        print(f"Word count: {len(speech_text.split())}")
        analyzer.analyze_speech(
            speech_text,
            f'attention_analysis/speech_{i}'
        )

if __name__ == "__main__":
    main()