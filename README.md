# MATH Dataset Post-training

Post-training experiments on the MATH dataset (Hendrycks et al.) for mathematical reasoning using policy gradient methods.

## Experiments

### Supervised Fine-tuning (SFT)
Fine-tune base models on MATH training data with YAML-based configuration system and multi-device training.

### Expert Iteration (EI) 
Policy gradient training with generated solutions. Samples multiple responses, computes rewards, and performs gradient updates using r1_zero grading.

### Baseline Evaluation
Evaluate pretrained models on MATH validation set using standardised evaluation pipeline.


## Attribution

- MATH dataset experiments based on Hendrycks et al.
- Grader function from Stanford CS336 (originally understand-r1-zero)
- Prompt templates from Stanford CS336
