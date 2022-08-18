from BERTapi import BERT
from transformers import T5ForConditionalGeneration, AutoTokenizer

class Ensemble:

  def __init__(self,device='cpu', n_best=10, max_length=512, stride=256, no_answer=True):
    self.device = device

    # ParsBERT
    self.model_path_ParsBERT = "sepiosky/ParsBERT_QA"
    self.predictor_ParsBERT = BERT(self.model_path_ParsBERT, device, n_best, max_length, stride, no_answer)

    # mBERT
    self.model_path_mBERT = "sepiosky/MBERT_QA"
    self.predictor_mBERT = BERT(self.model_path_mBERT, device, n_best, max_length, stride, no_answer)

    #alBERT
    self.model_path_alBERT = "sepiosky/ALBERT_QA"
    self.predictor_alBERT = BERT(self.model_path_mBERT, device, n_best, max_length, stride, no_answer)

    #ParsT%
    self.model_path_ParsT5 = "sepiosky/ParsT5_QA"
    self.model_ParsT5 = T5ForConditionalGeneration.from_pretrained(self.model_path_ParsT5)
    self.tokenizer_ParsT5 = AutoTokenizer.from_pretrained(self.model_path_ParsT5)
    self.model_ParsT5.to(device)


  def __call__(self, question, context, batch_size=1):

      _, preds_ParsBERT = self.predictor_ParsBERT(question, context, batch_size=1)
      _, preds_mBERT = self.predictor_mBERT(question, context, batch_size=1)
      _, preds_alBERT = self.predictor_alBERT(question, context, batch_size=1)

      input = 'متن: ' + context + '، پرسش: ' + question
      input_ids_ParsT5 = self.tokenizer_ParsT5.encode(input, return_tensors='pt').to(self.device)
      output_ids_ParsT5 = self.model_ParsT5.generate(input_ids_ParsT5, max_length=150, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
      output_ParsT5 = ' '.join([self.tokenizer_ParsT5.decode(id) for id in output_ids_ParsT5])
      pred_ParsT5 = output_ParsT5.replace('<pad>', '').replace('</s>', '').strip()

      # votes
      preds_scores = [preds_ParsBERT[0], preds_mBERT[0] , preds_alBERT[0] ]
      preds_text = [preds_ParsBERT[0]["text"], preds_mBERT[0]["text"] , preds_alBERT[0]["text"] , pred_ParsT5]
      resutls = {}
      for pred in preds_text:
        if pred.strip() not in list(resutls.keys()):
          resutls[pred.strip()] = 1
        else:
          resutls[pred.strip()] += 1

      high_voted = sorted(resutls)[0]
      votes = resutls[high_voted]
      if votes==1 :
        high_voted = sorted(preds_scores , key=lambda x: x["score"], reverse=True)[0]['text']

      return high_voted, _
