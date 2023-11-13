# PQA
Persian QesutionAnswering

[Web demo for testing models](https://sepiosky-pqa-webdemoweb-d9nn5x.streamlitapp.com/) 

Contributors : Amir Mansourian - Elnaz Rahmati - Zeinab Taghavi - Sepehr Ghobadi - Vida Ramezanian

Documentation: [Click here to for the paper](https://drive.google.com/file/d/1jovO0tzp3wQEpfthIY5dbDZrRF5CRcoZ/view?usp=sharing)

Trained models: 
<br/>
**ParsBERT**: [ParsBERT Model](https://drive.google.com/drive/folders/10j3KSd0zu4eM94yNKH7B-4grd5PhkeT_?usp=sharing)
<br/>
**mBERT**: [mBERT Model](https://drive.google.com/drive/folders/1Pk4U5XfXuT0zgLPCnxdZN-h4fDA_qzro?usp=sharing)
<br/>
**ALBERT**: [ALBERT Model](https://drive.google.com/drive/folders/1BXE0RxNww5aj5BvtniDy_aY6CGXdh7eS?usp=sharing)
<br/>
**ParsT5**:  [ParsT5 Model](https://drive.google.com/drive/folders/1iKHu4Wr8_5MNysVfBd8PhROzANSQyBOm?usp=sharing)
<br/>
**Ensemble**: [Ensemble Function](https://drive.google.com/drive/folders/1oORC2iodaIRunO56eBJLGJFQlOuLVv3W?usp=sharing)

### Results

|   Model  |  EM  | F1 | BLEU | BLEU1 | BLEU4 | 
|:----------:|:---------:|:------------:|:------------:|:------------:|:------------:|
| Ensemble | **0.6633** |    **0.7023**   | **0.4693**  | **0.6943**  |**0.4682**   | 
| mBERT | 0.6484 |    0.6926  | 0.4274  | 0.6832  | 0.4261  | 
| ParsBERT | 0.6233 |    0.6690  | 0.4131  | 0.6593  |0.4118   | 
| ALBERT | 0.4737 |     0.5291     |       0.3697     |      0.5156    |      0.3680    | 
| ParsT5 | 0.4532 |    0.4815   | 0.3314    | 0.4722    | 0.3313     | 
