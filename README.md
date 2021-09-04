# NLP-BERT-for-Relation-Extraction

Forth Assignment in 'NLP - Natural Languages Processing' course by Prof. Yoav Goldberg, Prof. Ido Dagan and Prof. Reut Tsarfaty at Bar-Ilan University.

**In this assignment, I implemented a Relation Extraction (RE) machine-learning based system, by using the Transfer Learning technique.**
That is, I picked up pre-trained BERT language model and fine-tuned it for few epochs on the given Relation Extraction dataset.

<ins>I chose to examine and compare two main fine-tuning techniques on BERT-Base-Uncased model with the Relation classification task using the provided dataset:</ins>
1. The first technique is to train the entire architecture. That is, to further train the entire pre-trained model together with the additional task-specific layers on the dataset and feed the final output to a sigmoid layer. In this technique, the error is backpropagated through the entire architecture and the pre-trained weights of the model are updated based on the new dataset.
2. The second technique is to train some layers while freezing others. That is, to train it partially – for example to keep the weights of the initial layers of the model frozen while retraining only the higher layers

To get to the bottom of this assignment, I tried about 8 different settings of the second finetuning technique in addition to experimenting with the first fine-tuning technique, in order to find the best fine-tuning settings of BERT for this task.

**<ins>General approach:</ins>**  
For this work, I got inspired by the paper ‘Enriching Pre-trained Language Model with Entity Information for Relation Classification’ written by Shanchan Wu and Yifan He, and I decided to try and replicate the RBERT model introduced in this paper, while making the necessary changes in order to fit it to my binary classification task and the amount of available data I have got for it, hoping to achieve good performance as reported in the paper.
(The figure below was taken from the paper but was edited by me, so that it will represent the architecture I actually implemented. I also changed some of the original notations for my convenience - I used these new notations in the code as well).
![image](https://user-images.githubusercontent.com/49911079/132099248-884faa51-6b71-416b-8092-00d8438b2d04.png)
**<ins>The final parameter settings:</ins>**
* <ins>Pretrained BERT:</ins>  bert-base-uncased
* <ins>Number of epochs:</ins> 8
* <ins>Batch size:</ins>       1 (as the dataset is very small)
* <ins>Activation:</ins>       Tanh
* <ins>Optimizer:</ins>        AdamW
* <ins>Learning rate:</ins>    2e-5
* <ins>Dropout rate:</ins>     0.2
* <ins>Loss Function:</ins>    BCELoss (binary Cross Entropy loss)

**<ins>Evaluation results:</ins>**
![image](https://user-images.githubusercontent.com/49911079/132099304-d2afd880-2747-464e-8571-f691f3236b91.png)
**<ins>Score:</ins>** 100
