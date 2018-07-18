# LDA
Topics model for extract content from a dataset composed by italian web-news (json format).


I used pyspark because the dataset that i used was really big (159k articles). For the moment i'm just using a smaller df (760 news only).

I create also a visualization of the results with pyLDAvis, that shows at the same time (interactively) what words are important for a topic and in which topics is often used a word. 


Notebook viewer: https://nbviewer.jupyter.org/github/robertaparisi/LDA/blob/master/LDAbella.ipynb
