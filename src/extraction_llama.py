import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from typing import Dict, List, Tuple
import re
import random
import gc

class LlamaActivationExtractor:
    def __init__(self, model_name: str, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="eager"  # Required for attention extraction
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda" and hasattr(self.model, 'to'):
            self.model.to(self.device)

        self.model.eval()

        # Storage for both activations and attention weights
        self.activations = {}
        self.attention_weights = {}
        self.hooks = []

        self._register_hooks()
        print(f"Model loaded. Registered {len(self.hooks)} hooks (MLP + Attention)")

    def _register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if hasattr(hidden_states, 'shape') and len(hidden_states.shape) == 3:
                    activation = hidden_states.detach().cpu().numpy()
                    self.activations[name] = activation

            return hook

        def get_attention_weights(name):
            def hook(module, input, output):
                if output is not None and isinstance(output, tuple):
                    if len(output) >= 2 and output[1] is not None:
                        attention_weights = output[1].detach().cpu().numpy()
                        if len(attention_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                            self.attention_weights[name] = attention_weights
            return hook

        # Find transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            print("Could not find transformer layers!")
            return

        # Register hooks for both MLP activations and attention weights
        for i, layer in enumerate(layers):
            # MLP activation hook (whole layer)
            mlp_hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
            self.hooks.append(mlp_hook)

            # Attention weights hook
            if hasattr(layer, 'self_attn'):
                attn_hook = layer.self_attn.register_forward_hook(get_attention_weights(f'attn_layer_{i}'))
                self.hooks.append(attn_hook)

    def extract_full_sequence_activations(self, prompt: str):
        self.activations.clear()
        self.attention_weights.clear()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Forward pass with attention extraction
            outputs = self.model(
                **inputs,
                output_attentions=True,  # Force attention computation
                return_dict=True,
                use_cache=False
            )

            # Debug: Print attention shapes
            print(f"Extracted {len(self.attention_weights)} attention matrices")
            for key, attn in self.attention_weights.items():
                print(f"  {key}: {attn.shape}")
            
            print("=== DEBUG: After hooks in extract_full_sequence_activations ===")
            for key, attn in self.attention_weights.items():
                print(f"  HOOK: {key}: {attn.shape}")
            print("=" * 60)

            # Also capture attention from model output as backup
            # if hasattr(outputs, 'attentions') and outputs.attentions:
            #     for i, attn in enumerate(outputs.attentions):
            #         if attn is not None:
            #             attn_np = attn.detach().cpu().numpy()
            #             key = f"attn_layer_{i}_direct"
            #             self.attention_weights[key] = attn_np

        return self.activations.copy(), self.attention_weights.copy()

    def create_few_shot_prompt(self, problem: str) -> str:
        """Create a few-shot prompt with examples to improve model performance"""

        few_shot_examples = """Here are some examples of how to solve problems:

Example 1:
Problem: Boris has 24 books and he donates a fourth of his books to the library. 
Cameron has 30 books and he donates a third of his books to the library. 
After donating their books, how many books in total do Boris and Cameron have together?
The answer is 38.

Example 2:
Problem: Two twins sisters Sita and Geeta were standing back to back and suddenly they started running in opposite direction for 4 miles and then turn to left and run for another 3 miles. 
What is the distance between the the two twins when they stop ?
The answer is 10.

Example 3:
Problem: Gerald had 20 toy cars. He donated 1/4 of his toy cars to an orphanage. How many toy cars does Gerald have left?
The answer is 15.

Now solve this problem:
Problem: """

        return few_shot_examples + problem + "\nSolution: \n"

    def find_problem_start(self, original_prompt: str, generation_prompt: str) -> int:
        """Find where the actual problem starts in the generation prompt"""
        marker_text = "Problem: " + original_prompt.split('.')[0]
        
        # Find text position
        text_pos = generation_prompt.find(marker_text)
        if text_pos == -1:
            # Try just looking for the original prompt
            text_pos = generation_prompt.find(original_prompt.split('.')[0])
            if text_pos == -1:
                return 0
        
        # Convert text position to token index
        text_before_problem = generation_prompt[:text_pos]
        tokens_before = self.tokenizer.encode(text_before_problem, add_special_tokens=False)
        return len(tokens_before)

    def slice_data(self, start_idx: int, end_idx: int):
        """Slice both attention and activations"""
        if start_idx == 0:
            return
        
        # Slice attention matrices
        for key, attention in list(self.attention_weights.items()):
            if len(attention.shape) == 4:
                self.attention_weights[key] = attention[:, :, start_idx:end_idx, start_idx:end_idx]
        
        # Slice activations
        for key, activation in list(self.activations.items()):
            if len(activation.shape) == 3:
                self.activations[key] = activation[:, start_idx:end_idx, :]

    def generate_with_clean_attention_capture(self, original_prompt: str, generation_prompt: str, max_new_tokens: int = 150):
        self.activations.clear()
        self.attention_weights.clear()
        
        inputs = self.tokenizer(generation_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_sequences = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )

        # Find problem start and slice data
        start_idx = self.find_problem_start(original_prompt, generation_prompt)
        self.slice_data(start_idx, len(generated_sequences[0]))
        
        # Return tokens for metadata
        tokens = self.tokenizer.convert_ids_to_tokens(generated_sequences[0][start_idx:])
        token_ids = generated_sequences[0][start_idx:].cpu().numpy().tolist()
        
        return generated_sequences, tokens, token_ids

    def extract_numerical_answer(self, text: str) -> str:
        # Look for "the answer is" pattern first
        answer_pattern = r'the answer is\s*(\d+(?:\.\d+)?)'  # Allow decimal numbers
        answer_match = re.search(answer_pattern, text.lower())
        if answer_match:
            return answer_match.group(1)

        # Look for "answer:" pattern
        answer_pattern2 = r'answer:\s*(\d+(?:\.\d+)?)'
        answer_match2 = re.search(answer_pattern2, text.lower())
        if answer_match2:
            return answer_match2.group(1)

        # Fallback to last number if no clear answer pattern
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            return numbers[-1]
        return "No numerical answer found"

    def process_single_prompt(self, prompt: str, prompt_name: str, row_index: int, seed: int = 37, use_few_shot: bool = True):
        torch.manual_seed(seed)
        
        generation_prompt = self.create_few_shot_prompt(prompt) if use_few_shot else prompt
        generated_sequence, sliced_tokens, sliced_token_ids = self.generate_with_clean_attention_capture(
            prompt, generation_prompt, 200
        )

        # Extract answer
        full_text = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
        if use_few_shot:
            marker = "Problem: " + prompt
            start = full_text.find(marker)
            answer_text = full_text[start + len(marker):].strip() if start != -1 else ""
        else:
            answer_text = full_text[len(generation_prompt):].strip()

        # Label data
        labeled_activations = {f"row_{row_index}_{prompt_name.lower()}_{k}": v 
                            for k, v in self.activations.items()}
        labeled_attention = {f"row_{row_index}_{prompt_name.lower()}_{k}": v 
                            for k, v in self.attention_weights.items()}

        # Metadata
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(prompt, add_special_tokens=True)
        )
        
        token_metadata = {
            'prompt_name': prompt_name,
            'row_index': row_index,
            'input_tokens': input_tokens,
            'sliced_tokens': sliced_tokens,
            'numerical_answer': self.extract_numerical_answer(answer_text),
            'original_prompt': prompt
        }

        return answer_text, self.extract_numerical_answer(answer_text), labeled_activations, labeled_attention, token_metadata

    def process_dataset(self, df: pd.DataFrame, seed: int = 42, batch_size: int = 1, use_few_shot: bool = True):
        results_list = []
        all_activations = []
        all_attention_weights = []
        all_token_metadata = []

        total_rows = len(df)
        print(f"Processing {total_rows} rows with {'few-shot' if use_few_shot else 'simple'} prompting")

        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end]

            for local_idx, (idx, row) in enumerate(batch_df.iterrows()):
                print(f"Processing row {local_idx + 1}/{total_rows} (original index: {idx})")

                row_result = {
                    'Index': row.get('Index', idx),
                    'Source': row.get('Source', ''),
                    'Category': row.get('Category', ''),
                    'Difficulty_Level': row.get('Difficulty_Level', ''),
                    'Prompt_1': row.get('Prompt_1', ''),
                    'Prompt_2': row.get('Prompt_2', ''),
                    'Prompt_3': row.get('Prompt_3', '')
                }

                row_activations = {}
                row_attention_weights = {}
                row_token_metadata = {}

                for prompt_col in ['Prompt_1', 'Prompt_2', 'Prompt_3']:
                    if prompt_col in df.columns and pd.notna(row[prompt_col]) and str(row[prompt_col]).strip():
                        prompt_text = str(row[prompt_col]).strip()

                        try:
                            full_output, numerical_answer, activations, attention_weights, token_metadata = self.process_single_prompt(
                                prompt_text, prompt_col, idx, seed=seed, use_few_shot=use_few_shot
                            )

                            row_result[f"{prompt_col}_Prediction_Full"] = full_output
                            row_result[f"{prompt_col}_Prediction_Extracted"] = numerical_answer

                            row_activations.update(activations)
                            row_attention_weights.update(attention_weights)
                            row_token_metadata[prompt_col] = token_metadata

                        except Exception as e:
                            print(f"Error processing {prompt_col}: {str(e)}")
                            row_result[f"{prompt_col}_Prediction_Full"] = "ERROR"
                            row_result[f"{prompt_col}_Prediction_Extracted"] = "ERROR"
                            row_token_metadata[prompt_col] = {"error": str(e)}

                    else:
                        row_result[f"{prompt_col}_Prediction_Full"] = "No prompt"
                        row_result[f"{prompt_col}_Prediction_Extracted"] = "No prompt"

                results_list.append(row_result)
                all_activations.append(row_activations)
                all_attention_weights.append(row_attention_weights)
                all_token_metadata.append(row_token_metadata)

                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        results_df = pd.DataFrame(results_list)
        print(f"Processing complete. Results shape: {results_df.shape}")

        return results_df, all_activations, all_attention_weights, all_token_metadata

    def save_results(self, results_df: pd.DataFrame, all_activations: List[Dict],
                    all_attention_weights: List[Dict], all_token_metadata: List[Dict],
                    output_dir: str = "llama_output"):
        os.makedirs(output_dir, exist_ok=True)

        # Save main results
        results_df.to_csv(f"{output_dir}/results_llama.csv", index=False)
        print(f"Results saved to {output_dir}/results_llama.csv")

        # Save MLP activations in separate folder
        mlp_dir = f"{output_dir}/activations"
        os.makedirs(mlp_dir, exist_ok=True)

        activation_files_created = 0
        for row_idx, row_activations in enumerate(all_activations):
            if not row_activations:
                continue

            row_dir = f"{mlp_dir}/row_{row_idx}"
            os.makedirs(row_dir, exist_ok=True)

            prompt_activations = {}
            for activation_key, activation_array in row_activations.items():
                parts = activation_key.split('_')
                if len(parts) >= 4:
                    prompt_part = f"{parts[2]}_{parts[3]}"
                    layer_part = '_'.join(parts[4:])

                    if prompt_part not in prompt_activations:
                        prompt_activations[prompt_part] = {}
                    prompt_activations[prompt_part][layer_part] = activation_array

            for prompt_name, prompt_layers in prompt_activations.items():
                prompt_dir = f"{row_dir}/{prompt_name}"
                os.makedirs(prompt_dir, exist_ok=True)

                for layer_name, activation_array in prompt_layers.items():
                    filename = f"{layer_name}.npy"
                    filepath = f"{prompt_dir}/{filename}"
                    np.save(filepath, activation_array)
                    activation_files_created += 1

        print(f"Saved {activation_files_created} MLP activation files")

        # Save attention weights in separate folder
        attn_dir = f"{output_dir}/attention"
        os.makedirs(attn_dir, exist_ok=True)

        attention_files_created = 0
        for row_idx, row_attention_weights in enumerate(all_attention_weights):
            if not row_attention_weights:
                continue

            row_dir = f"{attn_dir}/row_{row_idx}"
            os.makedirs(row_dir, exist_ok=True)

            prompt_attentions = {}
            for attention_key, attention_array in row_attention_weights.items():
                parts = attention_key.split('_')
                # Handle different naming patterns for full sequence attention
                if "generation_step" in attention_key or "final_full_sequence" in attention_key or "clean_target" in attention_key:
                    # For generation step attention: row_X_prompt_Y_generation_step_Z_attn_layer_W
                    if len(parts) >= 6:
                        prompt_part = f"{parts[2]}_{parts[3]}"
                        layer_part = '_'.join(parts[4:])
                else:
                    # For regular attention: row_X_prompt_Y_attn_layer_Z
                    if len(parts) >= 5:
                        prompt_part = f"{parts[2]}_{parts[3]}"
                        layer_part = '_'.join(parts[4:])

                if 'prompt_part' in locals():
                    if prompt_part not in prompt_attentions:
                        prompt_attentions[prompt_part] = {}
                    prompt_attentions[prompt_part][layer_part] = attention_array

            for prompt_name, prompt_layers in prompt_attentions.items():
                prompt_dir = f"{row_dir}/{prompt_name}"
                os.makedirs(prompt_dir, exist_ok=True)

                for layer_name, attention_array in prompt_layers.items():
                    filename = f"{layer_name}.npy"
                    filepath = f"{prompt_dir}/{filename}"
                    np.save(filepath, attention_array)
                    attention_files_created += 1

        print(f"Saved {attention_files_created} attention weight files")

        # Save token metadata
        with open(f"{output_dir}/token_metadata_llama.json", 'w') as f:
            json.dump(all_token_metadata, f, indent=2)

        # Save separate summaries - MLP Activations Summary
        activation_summary = {}
        total_activations = 0

        for row_idx, row_activations in enumerate(all_activations):
            if not row_activations:
                continue

            row_summary = {}
            for key, activation in row_activations.items():
                parts = key.split('_')
                if len(parts) >= 4:
                    prompt_part = f"{parts[2]}_{parts[3]}"
                    layer_part = '_'.join(parts[4:])

                    if prompt_part not in row_summary:
                        row_summary[prompt_part] = {}

                    row_summary[prompt_part][layer_part] = {
                        'shape': list(activation.shape),
                        'dtype': str(activation.dtype),
                        'sequence_length': activation.shape[1] if len(activation.shape) > 1 else 1,
                        'hidden_dim': activation.shape[2] if len(activation.shape) > 2 else activation.shape[-1],
                        'file_path': f"activations/row_{row_idx}/{prompt_part}/{layer_part}.npy"
                    }
                    total_activations += 1

            if row_summary:
                activation_summary[f"row_{row_idx}"] = row_summary

        with open(f"{output_dir}/activation_summary_llama.json", 'w') as f:
            json.dump(activation_summary, f, indent=2)

        # Attention Weights Summary (updated for full sequence)
        attention_summary = {}
        total_attentions = 0

        for row_idx, row_attentions in enumerate(all_attention_weights):
            if not row_attentions:
                continue

            row_summary = {}
            for key, attention in row_attentions.items():
                parts = key.split('_')

                # Handle different naming patterns
                if "generation_step" in key or "final_full_sequence" in key or "clean_target" in key:
                    if len(parts) >= 6:
                        prompt_part = f"{parts[2]}_{parts[3]}"
                        layer_part = '_'.join(parts[4:])
                else:
                    if len(parts) >= 5:
                        prompt_part = f"{parts[2]}_{parts[3]}"
                        layer_part = '_'.join(parts[4:])

                if 'prompt_part' in locals():
                    if prompt_part not in row_summary:
                        row_summary[prompt_part] = {}

                    row_summary[prompt_part][layer_part] = {
                        'shape': list(attention.shape),
                        'dtype': str(attention.dtype),
                        'batch_size': attention.shape[0] if len(attention.shape) > 0 else 1,
                        'num_heads': attention.shape[1] if len(attention.shape) > 1 else 1,
                        'sequence_length': attention.shape[3] if len(attention.shape) > 3 else 1,
                        'attention_type': 'clean_target' if 'clean_target' in key else ('full_sequence' if ('generation_step' in key or 'final_full_sequence' in key) else 'prompt_only'),
                        'file_path': f"attention/row_{row_idx}/{prompt_part}/{layer_part}.npy"
                    }
                    total_attentions += 1

            if row_summary:
                attention_summary[f"row_{row_idx}"] = row_summary

        with open(f"{output_dir}/attention_summary_llama.json", 'w') as f:
            json.dump(attention_summary, f, indent=2)

        print(f"\nResults saved to: {output_dir}/")
        print(f"Files created: {activation_files_created + attention_files_created}")
        print(f"  MLP activations: {total_activations}")
        print(f"  Attention weights: {total_attentions}")
        print("Folder structure:")
        print("  activations/row_X/prompt_Y/layer_Z.npy")
        print("  attention/row_X/prompt_Y/clean_target_attn_layer_Z.npy")

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.attention_weights.clear()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

def select_test_rows(df: pd.DataFrame, num_rows: int = 5, method: str = "first"):
    if method == "first":
        test_df = df.head(num_rows)
    elif method == "last":
        test_df = df.tail(num_rows)
    elif method == "random":
        test_df = df.sample(n=min(num_rows, len(df)), random_state=42)
    else:
        test_df = df.head(num_rows)

    print(f"Selected {len(test_df)} rows for processing")
    return test_df

def main():
    extractor = LlamaActivationExtractor(
        model_name="meta-llama/Llama-3.2-1B" #______________Model___________#
    )

    try:
        print("Loading dataset...")
        df = pd.read_csv('/cluster/scratch/ekozachenko/experiments/dataset_final.csv')  #______Dataset___________#
        print(f"Dataset loaded: {len(df)} rows")
######################################
        # Test mode settings
        TEST_MODE = False # Write False for full dataset inference
        CHUNK_MODE = True      # Add this
        CHUNK_SIZE = 300       # Process 300 rows at a time  
        CHUNK_START = int(os.environ.get('CHUNK_START', 0))        # Change this for each run
        NUM_TEST_ROWS = 2
        TEST_METHOD = "first"
        USE_FEW_SHOT = True  # Set to False to disable few-shot prompting
 ##########################################
        if CHUNK_MODE:
            chunk_end = min(CHUNK_START + CHUNK_SIZE, len(df))
            df = df.iloc[CHUNK_START:chunk_end]
            output_folder = f"chunk_{CHUNK_START}_{chunk_end-1}_output"
            print(f"Processing chunk: rows {CHUNK_START} to {chunk_end-1} ({len(df)} rows)")
        elif TEST_MODE:
            df = select_test_rows(df, num_rows=NUM_TEST_ROWS, method=TEST_METHOD)
            output_folder = f"test_output_{NUM_TEST_ROWS}rows"
        else:
            output_folder = "/cluster/scratch/ekozachenko/experiments/full_output_llama"

        print(f"Processing {'test' if TEST_MODE else 'full'} dataset with {'few-shot' if USE_FEW_SHOT else 'simple'} prompting")

        results_df, all_activations, all_attention_weights, all_token_metadata = extractor.process_dataset(
            df, seed=42, batch_size=1, use_few_shot=USE_FEW_SHOT
        )

        print("Saving results...")
        extractor.save_results(results_df, all_activations, all_attention_weights, all_token_metadata, output_folder)

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        extractor.cleanup()
        print("Done!")

if __name__ == "__main__":
    main()