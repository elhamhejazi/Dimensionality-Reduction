

from libs.calculation import Calculation
from libs.dataSourceReader import DataSourceReader
from os import walk, getcwd
import argparse, sys
parser=argparse.ArgumentParser()
parser.add_argument('--all', help='if set to anything all will be calculated')
parser.add_argument('--i', help='Index of algorithm. If not set all will be calculated')
args=parser.parse_args()

def main():
    folder = getcwd() + '\\datasets\\'
    print("Looking in Folder: '" + folder + "'")
    filenames = next(walk(folder),
                     (None, None, []))[2]  # [] if no file

    print("All data sources will run" if args.all != None else "Only one data source will run")
    print("All algorithms will run" if args.i == None else "Algorithm {} will run".format(args.i))
    
    # Save Dataset Tran/Test for each DataSet
    calculator = Calculation(args)
    # for fileName in filenames:
    #     print("DataSource: '" + folder + fileName + "'")
    #     dataSource = DataSourceReader(folder + fileName)
    #     dataSource.getAllData( fileName)
    #     dataSource.fold(fileName)
    
    # Call Dataset Tran/Test for each DataSet --all=D1.csv To D13.csv, i== 0 To 19
    for fileName in filenames:
        print("File name in dataset: " , args.all)
        if(fileName == args.all ):
            calculator.calculate(fileName)
            if args.all == None:
                break

if __name__ == "__main__":
    main()
