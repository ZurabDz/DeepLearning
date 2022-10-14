from transformers import AlbertForMaskedLM, AlbertConfig
from transformers import AlbertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from itertools import chain


max_length = 256
# tokenizer vocab size is 30_000 Model's torch.embedding can be index at max embedding_dim-1 (embedding_dim is model's vocabsize hence 30_001)
vocab_size = 30_001
truncate_longer_samples = False


# TODO: what's wrong with rust that creates infinite recursion
# def _convert_token_to_id_with_added_voc(self, token: str) -> int:
#         index = self._tokenizer.token_to_id(token)
#         if index is None:
#             # return self.unk_token_id
#             return self.vocab_size
#         return index

tokenizer = AlbertTokenizerFast(tokenizer_file='tokenizer.json', model_max_length=max_length)


def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples['text'], truncation=True, padding="max_length",
                   max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples['text'], return_special_tokens_mask=True)


encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation

config = AlbertConfig.from_pretrained('albert-base-v2')
config.vocab_size=vocab_size
model = AlbertForMaskedLM(config=config)

dataset = load_dataset('text', data_files={'train': ['wikitext-103-v1/wikitext-103/wiki.train.tokens'],
 'valid': ['wikitext-103-v1/wikitext-103/wiki.valid.tokens']})


train_dataset = dataset["train"].map(encode, batched=True, num_proc=12)
test_dataset = dataset["valid"].map(encode, batched=True, num_proc=12)


if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

else:
  # remove other columns, and remain them as Python lists
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])




from itertools import chain
# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
# grabbed from: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result


if not truncate_longer_samples:
  train_dataset = train_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}", num_proc=8)
  test_dataset = test_dataset.map(group_texts, batched=True,
                                  desc=f"Grouping texts in chunks of {max_length}", num_proc=8)
  # convert them from lists to torch tensors
  train_dataset.set_format("torch")
  test_dataset.set_format("torch")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)


from config import ds_config

training_args = TrainingArguments(
    output_dir='output',          
    evaluation_strategy="steps",    
    overwrite_output_dir=True,      
    num_train_epochs=60,            
    per_device_train_batch_size=13, 
    gradient_accumulation_steps=8,  
    per_device_eval_batch_size=6,  
    logging_steps=250,             
    save_steps=250,
    load_best_model_at_end=True,  
    save_total_limit=8,
    # no_cuda=True
    fp16=True,
    fp16_opt_level='O2',
    deepspeed=ds_config,
    half_precision_backend='apex'
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()