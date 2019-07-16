# FakeML
This project applies machine learning to help detect the phenomena "fake" news.

While most algorithms to detect fake news rely on a simple bag-of-words technique, in this project I implemented a new algorithm that utilized both a hashing vectorizer and passive-aggressive classifier, a type of semi-supervised learning algorithm.

This project essentially reads a dataset with articles labelled either fake or real(credit to George McIntire) and vectorizes each word of the article to a numerical index through the hashing vectorizer. Then, using these features, a passive-aggressive classifier was able to differentiate between the articles using the Hinge-Loss Function. Finally, the program prints out how well the classifier worked on choosing between fake and real articles. 

This can be modified to personal use by changing the training to testing data ratio.
