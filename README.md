# LDA
Topics model for extract content from a dataset composed by italian web-news (json format).


I used pyspark because the dataset that i used was really big (159k articles). For the moment i'm just using a smaller df (760 news only).

I create also a visualization of the results with pyLDAvis, that shows at the same time (interactively) what words are important for a topic and in which topics is often used a word. 


The updated file are the following: 
  - myLDAfunctions.py that has inside all the functions that i created to run the model
  - myLDArunfileOLD.ipynb that has inside the command that needs to be runned in order to have the results
  
  
 In a while I will add also the Web Application that I create for this NLP model

Notebook viewer functions file : https://nbviewer.jupyter.org/github/robertaparisi/LDA/blob/master/myLDAfunct.ipynb

Notebook viewer run file: https://nbviewer.jupyter.org/github/robertaparisi/LDA/blob/master/myLDArunfileOLD.ipynb

Notebook viewer old version: https://nbviewer.jupyter.org/github/robertaparisi/LDA/blob/master/LDAbella.ipynb
