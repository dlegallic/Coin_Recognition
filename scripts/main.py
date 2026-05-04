import process
import opencvComparison as cv
import argparse

def main(algo, dataset, p1, p2, graphic, full, file_path):
    if full :
        if dataset=='test':
            imagePath= '../bases/base_test/images_test'
            jsonPath = '../bases/base_test/labels_test'
            print("Running on the test dataset...")
        elif dataset=='validation':
            imagePath= '../bases/base_validation/images_validation'
            jsonPath = '../bases/base_validation/labels_validation'
            print("Running on the validation dataset...")
        else:
            raise ValueError("dataset must be 'test' or 'validation'")
        print(process.iterate(imagePath,jsonPath, algo, p1, p2, graphic))
    else :
        if(algo=='manual'):
            print("Running single file...")
            process.processImg(file_path, graphicEnabled=True)
        elif(algo=='opencv'):
            print("Running single file...")
            cv.processImg(file_path, p1, p2, graphicEnabled=True)
        else:
            raise ValueError("algorithm must be 'manual' or 'opencv'")
       


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="A simple tool to detect coins in a given image using Hough's transform.",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-a", "--algo", metavar="ALGORITHM", nargs="?", default="manual", choices=["manual", "opencv"], 
                            help="Algorithm type.")
        parser.add_argument("--p1", nargs="?", default=150, type=int,  
                            help="Parameter for openCV.")
        parser.add_argument("--p2", nargs="?", default=30, type=int,  
                            help="Parameter for openCV.")
        parser.add_argument("-g", "--graphic", action="store_true", 
                            help="Show plot of analyzed images.")
        
        
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--full", action="store_true", help="Run on a full dataset.")
        mode.add_argument("--file", dest="file_path", help="Run on a single image file.")
        
        parser.add_argument("-d", "--dataset", metavar="DATASET", nargs="?", default="test", choices=["test", "validation"],  
                            help="Dataset to analyse.")
        
        args = parser.parse_args()
        main(args.algo, args.dataset, args.p1, args.p2, args.graphic, args.full, args.file_path)
    except KeyboardInterrupt:
        print("\nInterrupted during the process.")

