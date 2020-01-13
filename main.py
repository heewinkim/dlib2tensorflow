import os
import sys
import argparse
import tensorflow as tf
from converter.model import build_dlib_model
from converter.weights import load_weights

def main(args):
    """ Main entry point """

    # Build the model (just the graph)
    keras_model = build_dlib_model(use_bn=False)
    keras_model.summary()   

    # parse xml and load weights
    load_weights(keras_model, args.xml_path)

    # save it as h5
    keras_model.save("dlib_face_recognition_resnet_model_v1.h5")

    # save it as saved_model
	tf.saved_model.save(keras_model,'./saved_model/')


def parse_arg(argv):
    """ Parse the arguments """
    arg_paser = argparse.ArgumentParser()

    arg_paser.add_argument(
        '--xml-path',
        type=str,
        required=True,
        help='Path to the dlib recognition xml file')    

    return arg_paser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arg(sys.argv[1:]))
