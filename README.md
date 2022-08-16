# PQA
Persian QesutionAnswering

Contributors : Amir Mansourian - Elnaz Rahmati - Zeinab Taghavi - Sepehr Ghobadi - Vida Ramezanian

Documentation: https://docs.google.com/document/d/1GRqZdRClQduQxV5ld0vCRSdLCTcN85yAr-RdA8X2tGw/edit?usp=sharing

Trained models: (put the links to trained models here)
<br/>
**ParsBERT**: https://drive.google.com/drive/u/1/folders/10j3KSd0zu4eM94yNKH7B-4grd5PhkeT_
<br/>
XLM-RoBERTa:
<br/>
**ParsT5**: The model was too big to finetune on colab, so for just one epoch, I had to split training data to 4 batches and train the model in four steps. accuracy after one epoch was 0.462 and model didn't converge so better results can be achieved if we had access to more computational resources. You can find a tutorial on how to use this model in this path: Models/ParsT5_test.ipynb and finally link to model: https://drive.google.com/drive/folders/1iKHu4Wr8_5MNysVfBd8PhROzANSQyBOm?usp=sharing



### Results

|   Model  |  EM  | F1 | 
|:----------:|:---------:|:------------:|
| ParsBERT | 0.1896853146853147 |    0.3427832896026578   | 
| ParsT5 | 0.3173076923076923 |    0.38419642354479394     |  
| XLM-RoBERTa | ? |     ?     |  
