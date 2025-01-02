import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import random
from attention_analyzer import AttentionAnalyzer  # Import function from existing file

class AggregateAttentionAnalyzer:
    def __init__(self, issue_model_path: str, axis_model_path: str):
        """Initialize analyzer with model paths"""
        self.attention_analyzer = AttentionAnalyzer(issue_model_path, axis_model_path)
        
    def load_and_filter_speeches(self, file_paths: List[str], n_speeches: int) -> List[Tuple[str, str]]:
        """Load and filter speeches from multiple files"""
        all_appropriate_speeches = []
        
        # Calculate how many speeches to take from each file
        speeches_per_file = n_speeches // len(file_paths)
        remaining_speeches = n_speeches % len(file_paths)
        
        for file_path in file_paths:
            # Load speeches from file
            with open(file_path, 'r') as f:
                speeches = json.load(f)
            
            # Filter speeches by length
            appropriate_speeches = []
            for speech_id, speech_data in speeches.items():
                word_count = len(speech_data['speech'].split())
                if 200 <= word_count <= 400:
                    appropriate_speeches.append((speech_id, speech_data['speech']))
            
            # Determine number of speeches to take from this file
            n_to_take = speeches_per_file + (1 if remaining_speeches > 0 else 0)
            remaining_speeches -= 1
            
            # Randomly sample speeches
            sampled_speeches = random.sample(appropriate_speeches, min(n_to_take, len(appropriate_speeches)))
            all_appropriate_speeches.extend(sampled_speeches)
        
        return all_appropriate_speeches
    
    def normalize_attention_patterns(self, attention: torch.Tensor, tokens: List[str]) -> np.ndarray:
        """Normalize attention patterns to a fixed length for aggregation"""
        # Remove special tokens
        mask = [t not in ['[PAD]', '[CLS]', '[SEP]', '<s>', '</s>', '<pad>'] for t in tokens]
        filtered_attention = attention.cpu().numpy()[mask]
        
        # Convert to relative positions (0 to 1)
        n_positions = 100  # Fixed number of positions
        attention_scores = filtered_attention.mean(axis=0)  # Average attention received by each token
        
        # Resample to fixed number of positions
        positions = np.linspace(0, len(attention_scores)-1, n_positions, dtype=int)
        normalized_attention = attention_scores[positions]
        
        return normalized_attention
    
    def analyze_aggregate_patterns(self, speeches: List[Tuple[str, str]], save_dir: str):
        """Analyze attention patterns across multiple speeches"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize arrays for aggregating attention patterns
        n_positions = 100
        issue_attention_aggregate = np.zeros(n_positions)
        axis_attention_aggregate = np.zeros(n_positions)
        
        print("Analyzing speeches...")
        for speech_id, speech_text in tqdm(speeches):
            # Get attention patterns
            issue_attention, issue_tokens = self.attention_analyzer.get_issue_model_attention(speech_text)
            axis_attention, axis_tokens = self.attention_analyzer.get_axis_model_attention(speech_text)
            
            # Normalize and aggregate
            issue_normalized = self.normalize_attention_patterns(issue_attention, issue_tokens)
            axis_normalized = self.normalize_attention_patterns(axis_attention, axis_tokens)
            
            issue_attention_aggregate += issue_normalized
            axis_attention_aggregate += axis_normalized
        
        # Average the aggregates
        issue_attention_aggregate /= len(speeches)
        axis_attention_aggregate /= len(speeches)
        
        # Plot aggregate patterns
        self.plot_aggregate_patterns(
            issue_attention_aggregate,
            axis_attention_aggregate,
            len(speeches),
            save_dir
        )
    
    def plot_aggregate_patterns(self,
                            issue_attention: np.ndarray,
                            axis_attention: np.ndarray,
                            n_speeches: int,
                            save_dir: str):
        """Create visualizations of aggregate attention patterns"""
        start_idx = int(len(issue_attention) * 0.03)
        
        # Create x positions excluding first 3%
        x_positions = np.linspace(3, 100, len(issue_attention)-start_idx)
        
        # Plot issue model patterns
        plt.figure(figsize=(15, 6))
        plt.plot(x_positions, issue_attention[start_idx:], label='Average Attention', linewidth=2)
        
        # Add trend line
        window = 5
        smoothed = np.convolve(issue_attention[start_idx:], np.ones(window)/window, mode='valid')
        x_smooth = x_positions[window-1:]
        plt.plot(x_smooth, smoothed, label='Smoothed Trend', linewidth=3, alpha=0.7)
        
        plt.title(f'Issue Classification Model - Aggregate Attention Pattern\n(Averaged over {n_speeches} speeches)', 
                fontsize=16)
        plt.xlabel('Relative Position in Speech (%)', fontsize=14)
        plt.ylabel('Average Attention', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/aggregate_issue_attention.png")
        plt.close()
        
        # Plot axis model patterns
        plt.figure(figsize=(15, 6))
        plt.plot(x_positions, axis_attention[start_idx:], label='Average Attention', linewidth=2)
        
        # Add trend line
        smoothed = np.convolve(axis_attention[start_idx:], np.ones(window)/window, mode='valid')
        plt.plot(x_smooth, smoothed, label='Smoothed Trend', linewidth=3, alpha=0.7)
        
        plt.title(f'Political/Emotional Axis Model - Aggregate Attention Pattern\n(Averaged over {n_speeches} speeches)', 
                fontsize=16)
        plt.xlabel('Relative Position in Speech (%)', fontsize=14)
        plt.ylabel('Average Attention', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/aggregate_axis_attention.png")
        plt.close()
        
        # Save the raw data
        np.savez(f"{save_dir}/aggregate_attention_data.npz",
                issue_attention=issue_attention,
                axis_attention=axis_attention)

def main():
    # Model paths
    ISSUE_MODEL_PATH = "issue_classifier_eval/model/saved_issue_model"
    AXIS_MODEL_PATH = "large-training-output/model_artifacts_20241202_142615/model.pt"
    
    # Speech files to analyze
    SPEECH_FILES = [
        "issue_classifier/outputs/speeches_104_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_105_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_106_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_107_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_108_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_109_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_110_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_111_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_112_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_113_gpt_topic_labels.json",
        "issue_classifier/outputs/speeches_114_gpt_topic_labels.json"
    ]
    
    # Initialize analyzer
    analyzer = AggregateAttentionAnalyzer(ISSUE_MODEL_PATH, AXIS_MODEL_PATH)
    
    # Load and analyze speeches
    n_speeches = 500
    speeches = analyzer.load_and_filter_speeches(SPEECH_FILES, n_speeches)
    
    print(f"Found {len(speeches)} appropriate speeches")
    analyzer.analyze_aggregate_patterns(speeches, 'aggregate_attention_analysis')

if __name__ == "__main__":
    main()