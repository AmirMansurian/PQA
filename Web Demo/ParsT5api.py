from transformers import T5ForConditionalGeneration, AutoTokenizer

MODEL_PATH = "sepiosky/ParsT5_QA"

class ParsT5:
    def __init__(self, device='cpu', n_best=10, max_length=512, stride=256, no_answer=False):
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
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        self.model = self.model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self.no_answer = no_answer
        self.n_best = n_best


    def __call__(self, question, context, batch_size=1, n_best=20,stride=256,answer_max_len = 100,no_answer=True):
        input = 'متن: ' + context + '، پرسش: ' + question

        input_ids = self.tokenizer.encode(input, return_tensors='pt')
        output_ids = self.model.generate(input_ids, max_length=150, num_beams=2, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
        output = ' '.join([self.tokenizer.decode(id) for id in output_ids])
        return output.replace('<pad>', '').replace('</s>', '').strip()