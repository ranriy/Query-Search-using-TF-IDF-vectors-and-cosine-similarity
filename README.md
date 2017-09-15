# Query-Search-using-TF-IDF-vectors-and-cosine-similarity
Analyzed a corpus containing 30 .txt files and retrieved the most relevant document for a given query using Python and NLTK

#Dataset:
presidential_debates folder contains a collection of general election presidential debates from 1960 to 2012. Each of the 30 files contains the transcript of a debate and is named by the date of the debate.

#Steps:
1) Read the 30 .txt files.
2) Tokenized the contents of the files.
3) Performed stopword removal on the obtained tokens.
4) Performed stemming on the obtained tokens.
5) Computed the TF-IDF vector for each document.
6) Given a query string, calculated the query vector.
7) Returned the document which results in the highest cosine similarity score. Constructed and used a posting list (document, TF-IDF weights) for each token in the corpus.
