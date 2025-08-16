import torch
import os
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner as SAETrainingRunner,
    StandardTrainingSAEConfig,
)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Create SAE config with required parameters
    sae_config = StandardTrainingSAEConfig(
        d_in=2048,
        d_sae=4096,
        l1_coefficient=2.5,
        device=device,
        dtype="float32",
    )
    print("SAE config created successfully")

    # SAE Training Configuration
    cfg = LanguageModelSAERunnerConfig(
        sae=sae_config,
        # Model and hook
        model_name="meta-llama/Llama-3.2-1B",
        hook_name="blocks.8.hook_resid_post",
        # Dataset
        dataset_path="monology/pile-uncopyrighted",
        streaming=True,
        is_dataset_tokenized=False,
        context_size=128,
        # Training
        training_tokens=2000000,
        train_batch_size_tokens=4096,
        store_batch_size_prompts=8,
        n_batches_in_buffer=32,
        # Learning parameters
        lr=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        # Hardware and misc
        device=device,
        seed=42,
        dtype="float32",
        verbose=True,
    )

    print("Starting SAE training...")
    sae_trainer = SAETrainingRunner(cfg)
    sae = sae_trainer.run()
    print("Training completed!")

if __name__ == "__main__":
    main()