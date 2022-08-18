from BERTapi import BERT

MODEL_PATH = "sepiosky/ParsBERT_QA"

class ParsBERT(BERT):
    def __init__(self, device='cpu', n_best=10, max_length=512, stride=256, no_answer=False):
        super().__init__(MODEL_PATH, device, n_best, max_length, stride, no_answer)

    def __call__(self, question, context, batch_size=1, answer_max_len=100):
        return super().__call__(question, context, batch_size, answer_max_len)
