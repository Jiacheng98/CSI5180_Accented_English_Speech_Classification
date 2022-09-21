Introduction
------------

The project aims to classify English accents using two deep learning models: MLP and Transformer Encoder [1]. The two models were tested on the UT-Podcast corpus [2], and the Transformer Encoder achieved a 74% improvement on the macro F1 score.

The dataset can be downloaded by running the download_dataset.py file. 

The main function can be found in main.py, and the code of MLP and Transformer Encoder can be found in MLP.py and TransformerModel.py, respectively. 

The code accent_googleAPI.py is used to call the Google API for speech to text, where the input is a WAV file and a language code that represents the accent of the WAV file.
<br />
<br />


References
------------
* [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
* [2] Hansen, J. H., & Liu, G. (2016). Unsupervised accent classification for deep data fusion of accent and language information. Speech Communication, 78, 19-33.


