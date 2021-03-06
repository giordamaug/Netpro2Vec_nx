import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import argparse
import sys,os
import time
import sys
sys.path.append('.')
from netpro2vec_nx.Netpro2Vec import Netpro2Vec
import utils
import tqdm
import csv

# import sklearn library
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold,cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

parser = argparse.ArgumentParser(description='Tesing Netpro2vec')
parser.add_argument('-i', "--input-path", metavar='<input-path>', type=str, help='input directory (default .)', required=True)
parser.add_argument('-l', "--labelfile", metavar='<labelfile>', type=str, help='label file (path))', required=True)
parser.add_argument('-E', "--embeddingdir", metavar='<embeddingdir>', type=str, default="./embeddings", help='embedding directory (default ./embeddings)', required=False)
parser.add_argument('-n', "--distributions", metavar='<distributions>', nargs='*',  type=str, default=['ndd'], help='list of distribution types (default: %(default)s)') 
parser.add_argument('-c', "--cutoffs", metavar='<cutoffs>', nargs='*',  type=float, default=[0.01], help='list of cutoffs (default: %(default)s)') 
parser.add_argument('-e', "--extractors", metavar='<extractors>', nargs='*',  type=int, default=[1], help='list of extractor indices (default: %(default)s)') 
parser.add_argument('-a', "--aggregators", metavar='<aggregators>', nargs='*',  type=int, default=[1], help='list of aggregator indices (default: %(default)s)') 
parser.add_argument('-o', "--embedfilename", metavar='<embedfilename>', type=str, help='embedding  file (default None))', required=False, default="")
parser.add_argument('-X', "--select", help="enable feature elimination (default disabled)", action='store_true')
parser.add_argument('-v', "--verbose", help="enable verobose printing (default disabled)", action='store_true')
parser.add_argument('-D', "--dump", help="enable embedding save (default disabled)", action='store_true')
parser.add_argument('-L', "--load", help="enable embedding load (default disabled)", action='store_true')
parser.add_argument('-p', "--label-position", dest='label_position', metavar='<label-position>', type=int, help='label position (default 2)', default=2, required=False)
parser.add_argument('-d', "--dimension", metavar='<dimension>', type=int, help='feature dimension (default 128)', default=128, required=False)
parser.add_argument('-x', "--extension", metavar="<extension>", type=str, default='graphml',choices=['graphml', 'edgelist'], help="file format (graphml, edgelist)) ", required=False)


def load_graphs(input_path, dataname, labels, fmt='graphml',verbose=False):
	_tqdm = tqdm.tqdm
	if not verbose: _tqdm = utils.nop 
	if os.path.isdir(input_path):
		filenames = os.listdir(input_path)
		utils.vprint("Loading " + dataname + " graphs with igraph...", verbose=verbose)
		graph_list = []
		targets = []
		for f in _tqdm(filenames):
			graph_list.append(nx.read_graphml(os.path.join(input_path, f)))
			targets += [labels[f.split('.')[0]]]
		y = np.array(targets)
		return graph_list,y 
	else:
		raise Exception("Problem opening input dir!")

def main(args):
      tm = time.time()
      # parse arguments
      args = parser.parse_args()
      if os.path.isdir(args.input_path):
        dataname = args.input_path.split("/")[-2] 
      else:
        dataname = args.input_path
      if not args.load:    # compute embeddings
        # read class labels
        labels= {}
        if args.labelfile:
            labelfile = open(args.labelfile)
            rows = list(csv.reader(labelfile, delimiter='\t'))
            for row in rows:
                 labels[row[0]] = row[args.label_position]
        tm0 = time.time()
        graphs,y = load_graphs(args.input_path, dataname, labels, fmt=args.extension,verbose=args.verbose)
        tm1 = time.time()
        if args.verbose: print("Embeddings...")
        tm2 = time.time()
        model = Netpro2Vec(dimensions=args.dimension, extractor=args.extractors,prob_type=args.distributions,
				      cut_off=args.cutoffs, agg_by=args.aggregators,verbose=args.verbose)
        model.fit(graphs)
        X = model.get_embedding()
        tm3 = time.time()
        if args.verbose: print("No. of features: " + str(X.shape[1])) 
        tm4 = time.time()
        X = X.astype(np.float)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X = scaler.fit_transform(X).astype(np.float)
        tm5 = time.time()
        if args.dump:
           if args.embedfilename == "":
               embedding_filename = os.path.join(args.embeddingdir, dataname +  "_" + args.model + ".csv")
           else:
               embedding_filename = args.embedfilename 
           if args.verbose: print("Saving embedding " + embedding_filename)
           data = pd.DataFrame(data=np.c_[X,y])
           data.to_csv(embedding_filename, index=False)
      else:
        if args.embedfilename == "":
           embedding_filename = os.path.join(args.embeddingdir, dataname +  "_" + args.model + ".csv")
        else:
           embedding_filename = args.embedfilename 
        tm0 = time.time()
        if args.verbose: print("Loading embedding " + embedding_filename)
        embedding_data = pd.read_csv(embedding_filename)
        X = embedding_data.iloc[:, :-1]
        X = np.array(X)
        y = np.array(embedding_data.iloc[:, -1])
        tm1 = tm2 = tm3 = tm4 = time.time()
        X = X.astype(np.float)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        X = scaler.fit_transform(X).astype(np.float)
        tm5 = time.time()
        if args.verbose: print("No. of features: " + str(X.shape[1]))
      tm6 = time.time()
      if args.select:
          clf = SVC(kernel="linear")
          sfm = RFECV(estimator=clf, step=10, min_features_to_select=50, cv=StratifiedKFold(random_state=0,shuffle=True,n_splits=5),scoring='accuracy')
          sfm = sfm.fit(X, y)
          X = sfm.transform(X)
          if args.verbose: print("Reduced dataset " + str(X.shape[1]))
      tm7 = time.time()
      clf = SVC(kernel='linear')
      scoring = ['accuracy', 'precision_macro', "recall_macro", "f1_macro"]
      scores_cv = cross_validate(clf, X, y, scoring=scoring, cv=RepeatedStratifiedKFold(n_splits=10 , n_repeats=10, random_state=465), return_train_score=False)
      tm8 = time.time()
      print('Acc Avg+Std:\t', (scores_cv['test_accuracy'] * 100).mean(), (scores_cv['test_accuracy'] * 100).std())
      print('Prec Avg+Std:\t', (scores_cv['test_precision_macro'] * 100).mean(),(scores_cv['test_precision_macro'] * 100).std())
      print('Recall Avg+Std:\t', (scores_cv['test_recall_macro'] * 100).mean(), (scores_cv['test_recall_macro'] * 100).std())
      print('F1 Avg+Std:\t', (scores_cv['test_f1_macro'] * 100).mean(), (scores_cv['test_f1_macro'] * 100).std())
      print('Time Load: %.2f Embed: %.2f Scaling: %.2f Elim: %.2f Validate: %.2f Tot: %.2f '%(tm1-tm0, tm3-tm2, tm5-tm4,tm7-tm6, tm8-tm7,tm8-tm))

if __name__ == "__main__":
    main(sys.argv[1:])
