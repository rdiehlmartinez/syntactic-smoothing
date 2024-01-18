
# typing imports
from typing import Any, Dict

from transformers import DataCollatorForLanguageModeling

class StackedCollator:
    def __init__(self, collators):
        self.collators = collators

    def __call__(self, *args, **kwargs):
        batch = {}
        for collator_name, collator in self.collators.items():
            batch.update(collator(*args, **kwargs))
        return batch

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, task_unit_name, unmask_probability=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_unit_name = task_unit_name
        self.unmask_probability = unmask_probability

    def torch_call(self, *args):
        """
        Prepares data for the masked language modeling task.

        NOTE: The Datacollators should always return a batch with the following keys:
        {
            input_ids_{task_unit_name}
            labels_{task_unit_name}
            pos_tags
        }
        """
        batch: Dict[str, Any] = super().torch_call(*args)
        assert "labels" in batch

        batch[f"input_ids_{self.task_unit_name}"] = batch["input_ids"]
        batch[f"labels_mlm_{self.task_unit_name}"] = batch["labels"]
        del batch["labels"]  # each task has its own labels
        del batch["input_ids"]  # each task has its own input_ids
        return batch

    # We override this function to allow us to adjust the probability of unmasking
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # Typical MLM objective is 80% mask, 10% random, 10% original
        # Here we do 90-self.unmask_probability mask, 10% random, self.unmask_probability original
        keep_mask_prob = 0.9 - self.unmask_probability
        random_prob = 0.1
        remainder_prob = random_prob / (self.unmask_probability + random_prob)

        # keep_mask_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, keep_mask_prob)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word. If self.unmask_probability is 0, this is all remaining masked tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, remainder_prob)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (self.unmask_probability) we keep the masked input tokens unchanged.
        return inputs, labels

