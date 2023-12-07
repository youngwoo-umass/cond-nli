
This is implementation for the PAT model introduced in the EMNLP 2023 paper Conditional Natural Langauge Inference.

# Environment

The codes are tested with working directory as "code" and PYTHONPATH=src

pip install -r requirements.xt

# Training and execution

Step 1. Generate training data ()
* bash 1_tfrecord_gen.sh

Step 2. Run Training
* bash 2_train.sh

Step 3. Run Cond-NLI Inference
* bash 3_cond_inf.sh

Step 4. Run evaluation
* bash 4_cond_eval.sh

