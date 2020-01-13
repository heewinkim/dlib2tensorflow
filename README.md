# dlib2tensorflow
how to convert dlib's deep learning model to tensorflow

# description blog
https://heewinkim.blogspot.com/2020/01/convert-deeplearning-model-dlib-to.html

# quick manual

----

### download dlib
$ wget http://dlib.net/files/dlib-19.19.tar.bz2
$ tar -xvf dlib-19.19.tar.bz2

### download face recognition weight(dlib)
$ wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
$ bzip2 dlib_face_recognition_resnet_model_v1.dat.bz2 --decompress

### generate face recognition xml file

$ g++ -std=c++11 -O3 -Idlib-19.19 ./dlib-19.19/dlib/all/source.cpp -lpthread -lX11 dlib_to_xml.cpp -o dlib_to_xml

$ ./dlib_to_xml

### convert dlib to tensorflow
python main.py --xml_path ./dlib_face_recognition_resnet_model_v1.xml
