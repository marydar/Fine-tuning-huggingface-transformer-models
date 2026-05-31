# Fine tune huggingface transformer models on different tasks

A collection of hands‑on notebooks demonstrating how to fine‑tune Hugging Face Transformer models on 7 core NLP tasks: text classification (CoLA), masked & causal language modeling, extractive question answering, summarization, translation, multiple choice, and token classification (NER). Each notebook includes dataset preprocessing, training, and evaluation using the Trainer API or custom PyTorch loops.

# Task 1 — Text Classification(Cola)
Predict whether a given English sentence is grammatically acceptable or not (binary classification). CoLA (Corpus of Linguistic Acceptability) is a standard benchmark for linguistic competence.

## Model checkpoint

distilbert-base-uncased

## Sample Data + AFter preprocessing
![Sample Data](/Text_classification/images/data_sample.png)
![pre Data](/Text_classification/images/preprocess_data.png)

### Preprocessing data
- Tokenize each sentence with the same tokenizer used by the pre-trained model.

- Add a classification token ([CLS] for BERT‑style models).

- Pad/truncate all sentences to the same maximum length.

-  Convert labels (0/1) into tensor format.

-   Create a Dataset object with input_ids, attention_mask, and labels.

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 5
- Weight Decay: 0.01

## Evaluation Metrics
![eval Data](/Text_classification/images/eval_logs.png)

## Training Curves
![loss curve](/Text_classification/images/loss_curves.png)
![acc curve](/Text_classification/images/accuracy_curve.png)

## Sample Predictions
- LABEL_1: acceptable
- LABEL_0: unacceptable

![e.g](/Text_classification/images/Screenshot%20from%202026-05-30%2019-51-27.png)

# Task 2 — Language Modeling
MLM (Masked Language Modeling): The model learns to predict randomly masked tokens in a sentence (e.g., I love [MASK] → NLP).

CLM (Causal Language Modeling): The model predicts the next token autoregressively (GPT‑style), used for text generation.

## Model checkpoint (MLM)

distilroberta-base

## Sample Data + AFter preprocessing

![Sample Data](/Language_modeling/images/data2.png)
![pre Data](/Language_modeling/images/pre.png)

### Preprocessing data
- For MLM:

    - Tokenize raw text.

    - Randomly mask 15% of tokens (replace with [MASK]).

    - Keep original tokens as labels.

- For CLM:

    - Tokenize and concatenate texts with an eos_token between them.

    -  Split into blocks of fixed length (e.g., 512 tokens).

    -   Shift input right by 1 token to create labels (input: [a, b, c], labels: [b, c, eos]).

    -   Use Hugging Face DataCollatorForLanguageModeling for dynamic masking/blocking.

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Weight Decay: 0.01

## Training loss and accuracy
![eval Data](/Language_modeling/images/train.png)

## Evaluation Metrics
![pre Data](/Language_modeling/images/eval.png)


## Training Curves
![pre Data](/Language_modeling/images/losses.png)

## Sample Predictions
![example Data](/Language_modeling/images/test.png)

## Model checkpoint (CLM)

distilgpt2

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Weight Decay: 0.01

## Training loss and accuracy
![eval Data](/Language_modeling/images/clm-train.png)

## Evaluation Metrics
![pre Data](/Language_modeling/images/clm-eval.png)


## Training Curves
![pre Data](/Language_modeling/images/clm-plot.png)

## Sample Predictions
![example Data](/Language_modeling/images/clm-test.png)



# Task 3 — Question Answering
Extract an answer span from a given context paragraph based on a question (extractive QA). Example: context = "Paris is the capital of France.", question = "What is the capital?" → answer = "Paris".

## Model checkpoint

distilbert-base-uncased


## Sample Data + AFter preprocessing
![Sample Data](/Question_answering/images/data.png)
##### example
![Sample Data](/Question_answering/images/example.png)
![pre Data](/Question_answering/images/pre.png)

### Preprocessing data
- Pair each question with its context.

- Tokenize the pair (question + [SEP] + context).

- Map the answer start/end character positions to token positions.

- Truncate long contexts (with stride to keep answer possible).

- Labels are start_positions and end_positions (integers)

## Training Arguments

- Learning Rate: 5e-5
- Batch Size: 16
- Epochs: 3
- Weight Decay: 0.01


# Task 4 — Multiple choice
Given a context and several answer options (usually 2–4), pick the correct one. Example: context + "What comes next?" with options A, B, C.

## Model checkpoint

distilbert-base-uncased


## Sample Data + AFter preprocessing
![Sample Data](/Multiple_choice/images/data.png)
![pre Data](/Multiple_choice/images/preprocess.png)

### Preprocessing data
-   For each example, create N input pairs: (context, option_i) for i=1..N.

-   Tokenize each pair independently.

-   Pad/truncate to same length across all options.

-   Labels = index of the correct option (0..N-1).

-   Model outputs logits over N choices.

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 5
- Weight Decay: 0.01

## Training loss and accuracy
![eval Data](/Multiple_choice/images/train.png)


## Evaluation Metrics
![eval Data](/Multiple_choice/images/eval.png)




# Task 5 — Token Classification(NER)
Assign a label to each token in a sentence (e.g., B-PER, I-LOC, O). Used for Named Entity Recognition (NER): detecting people, locations, organizations, etc.
## Model

distilbert-base-uncased


## Sample Data + AFter preprocessing
![Sample Data](/Token_classification/images/sample_data.png)

![pre Data](/Token_classification/images/preprocess.png)

### Preprocessing data
-    Tokenize sentence with is_split_into_words=False (or True if pre‑tokenized).

-   Align word‑level labels to subword tokens:

    -   First subword gets the word’s label.

    -    Subsequent subwords get -100 (ignored in loss).

-    Convert label strings to IDs using label2id mapping.

-    Pad/truncate to max length, using -100 for padding in labels.

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Weight Decay: 0.01

## Evaluation Metrics
![eval Data](/Token_classification/images/eval_logs.png)

## Training Curves
![loss curve](/Token_classification/images/loss_curves.png)
![acc curve](/Token_classification/images/acc-curve.png)

## Sample Predictions

![e.g](/Token_classification/images/example.png)

# Task 6 — Text Translation
Convert text from a source language (e.g., English) to a target language (e.g., French). Also a sequence‑to‑sequence task.

## Model checkpoint

Helsinki-NLP/opus-mt-en-ro


## Sample Data + AFter preprocessing
![Sample Data](/Translation/images/data.png)
![pre Data](/Translation/images/pre.png)

### Preprocessing data
-    Tokenize source sentence with tokenizer for source language (or multilingual tokenizer).

-   Tokenize target sentence separately with same tokenizer (for models like mT5, M2M100).

-    Set appropriate max_length for source and target (usually source longer).

-    Labels = target input_ids.

-    Add language tokens if required (e.g., "2en", "2fr" for some models).

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 1
- Weight Decay: 0.01


# Task 7 — Text Summarization
Generate a shorter, coherent summary of a longer text (e.g., news article → headline/abstract). Sequence‑to‑sequence task (encoder‑decoder models like BART, T5).
## Model checkpoint

T5-small

## Sample Data + AFter preprocessing
![Sample Data](/Summarization/images/data.png)

![pre Data](/Summarization/images/pre.png)

### Preprocessing data
-    Tokenize the source document and target summary separately.

-    Set max_source_length (e.g., 1024) and max_target_length (e.g., 256).

-    Use truncation=True and padding=True.

-    Return input_ids (source) and labels (target summary tokens).

-    During training, decoder inputs are labels shifted right; model handles it internally.

## Training Arguments

- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 1
- Weight Decay: 0.01




