# Created by Marek Lipert (2017). All rights reserved.
# Can be distributed under GPLv3
# See the LICENSE file for details

wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
bunzip2 test.csv.bz2
bunzip2 train.csv.bz2
mv GoogleNews-vectors-negative300.bin pretrained.bin
python prepare_data.py
python prepare_embeddings.py
python prepare_tensorflow_embeddings.py