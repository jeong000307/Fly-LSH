# Fly-LSH

sift_base.fvecs : decompress sift.tar.gz in http://corpus-texmex.irisa.fr

gist_base.fvecs : decompress gist.tar.gz in http://corpus-texmex.irisa.fr

glove.42B.300d.txt : decompress glove.42B.300d.zip in https://nlp.stanford.edu/projects/glove/

t10k-images.idx3-ubyte : decompress t10k-images-idx3-ubyte.gz in http://yann.lecun.com/exdb/mnist/

put these files in raw folder

python preprocessing.py

python main.py