import argparse


def setUpArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputpic", action="store",
                        help="add picture you want to run on", required=True)
    parser.add_argument("-sp", "--savepieces", action="store_true",
                        help="save the pieces the picture was broken up into", default=False)
    parser.add_argument("-l", "--length", action="store", type=int,
                        help="size of the length of square segments wanted in pixels", required=True)
    parser.add_argument("-sa", "--saveassembly", action="store_true",
                        help="save the assembled picture in each round", default=False)
    parser.add_argument("-a", "--showanimation", action="store_true",
                        help="show animation of picture being built", default=True)
    parser.add_argument("-k", "--use kruskal for building",
                        action="store_true")
    parser.add_argument("-p", "-use prims for building", action="store_true")
    return parser.parse_args()
