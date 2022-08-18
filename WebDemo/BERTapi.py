from curses import qiflush
import json
import pandas as pd
from tqdm import tqdm
from IPython.display import clear_output
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
#from datasets import load_dataset, load_from_disk, Dataset


class BERT:
    def __init__(self, model_path, device='cpu', n_best=10, max_length=512, stride=256, no_answer=False):
      """Initializes PyTorch Question Answering Prediction
      It's best to leave use the default values.
      Args:
          model: Fine-tuned torch model
          tokenizer: Transformers tokenizer
          device (torch.device): Running device
          n_best (int): Number of best possible answers
          max_length (int): Tokenizer max length
          stride (int): Tokenizer stride
          no_answer (bool): If True, model can return "no answer"
      """
      self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
      self.model = self.model.eval()
      self.tokenizer = AutoTokenizer.from_pretrained(model_path)
      self.device = device
      self.max_length = max_length
      self.stride = stride
      self.no_answer = no_answer
      self.n_best = n_best


    def __call__(self, question, context, batch_size=1, answer_max_len=100):
        """Creates model prediction

        Args:
            questions (list): Question strings
            contexts (list): Contexts strings
            batch_size (int): Batch size
            answer_max_len (int): Sets the longests possible length for any answer

        Returns:
            dict: The best prediction of the model
                (e.g {0: {"text": str, "score": int}})
        """
        questions, contexts = [question], [context]
        answer_max_len = 100
        n = len(contexts)
        tokens = self.tokenizer(questions, contexts, add_special_tokens=True,
                            return_token_type_ids=True, return_tensors="pt", padding=True,
                            return_offsets_mapping=True, truncation="only_second",
                            max_length=self.max_length, stride=self.stride)

        start_logits, end_logits = [], []
        for i in tqdm(range(0, n-batch_size+1, batch_size)):
            with torch.no_grad():
                out = self.model(tokens['input_ids'][i:i+batch_size].to(self.device),
                            tokens['attention_mask'][i:i+batch_size].to(self.device),
                            tokens['token_type_ids'][i:i+batch_size].to(self.device))

            start_logits.append(out.start_logits)
            end_logits.append(out.end_logits)

        tokens, starts, ends = tokens, torch.stack(start_logits).view(n, -1), torch.stack(end_logits).view(n, -1)
        start_indexes = starts.argsort(dim=-1, descending=True)[:, :self.n_best]
        end_indexes = ends.argsort(dim=-1, descending=True)[:, :self.n_best]

        preds = {}
        for i, (c, q) in enumerate(zip(contexts, questions)):
            min_null_score = starts[i][0] + ends[i][0]
            start_context = tokens['input_ids'][i].tolist().index(self.tokenizer.sep_token_id)

            offset = tokens['offset_mapping'][i]
            valid_answers = []
            for start_index in start_indexes[i]:

                if start_index<start_context:
                    continue
                for end_index in end_indexes[i]:

                    if (start_index >= len(offset) or end_index >= len(offset)
                        or offset[start_index] is None or offset[end_index] is None):
                        continue

                    if end_index < start_index or (end_index-start_index+1) > answer_max_len:
                        continue

                    start_char = offset[start_index][0]
                    end_char = offset[end_index][1]
                    valid_answers.append({"score": (starts[i][start_index] + ends[i][end_index]).item(),
                                        "text": c[start_char: end_char],
                                        "loc": [start_char , end_char]})
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                best_answer = {"text": "", "score": min_null_score,"loc": [torch.tensor(0) , torch.tensor(0)]}

            if self.no_answer:
                preds[i] = best_answer if best_answer["score"] >= min_null_score else {"text": "", "score": min_null_score
                                                                                        ,"loc": [torch.tensor(0) , torch.tensor(0)]}
            else:
                preds[i] = best_answer

        return preds[0]["text"].strip(), preds
