# MATH Dataset Post-training

Post-training experiments on the MATH dataset (Hendrycks et al.) for mathematical reasoning using policy gradient methods.

## Experiments

### Supervised Fine-tuning (SFT)
Fine-tune base models on MATH training data with YAML-based configuration system and multi-device training.

### Expert Iteration (EI) 
Iterative self-improvement by filtering correct solutions. Samples multiple responses from the model, filters correct ones using r1_zero grading, and fine-tunes on the filtered dataset.

### Baseline Evaluation
Evaluate pretrained models on MATH validation set using standardised evaluation pipeline.


## Attribution

- MATH dataset experiments based on Hendrycks et al.
- Grader function from Stanford CS336 (originally understand-r1-zero)
- Prompt templates from Stanford CS336
