import json
import torch
from query_generator import get_subgraphs
import numpy as np
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.walkers import RandomWalker
import matplotlib.pyplot as plt
from pytorch_utils import DynamicZModel
import pyodbc
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import tree, svm
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from torch_geometric.loader import DataLoader


from torch_geometric.data import Data, HeteroData
import torch.nn.functional as F
import json
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear, RGCNConv, RGATConv, HEATConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Embedding
from torch.nn import MSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import os
import random
from models import GINmodel, TripleModel
from utils import get_query_graph_data, StatisticsLoader, MyDataset, get_query_graph_data_new, ToUndirectedCustom
import time

class cardinality_estimator():
    """
    Base class for estimating cardinality for a given dataset.

    """
    def __init__(self, dataset_name, graphfile, sim_measure):
        self.dataset_name = dataset_name
        self.graphfile = graphfile
        self.sim_measure = sim_measure

        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"

        try:
            if os.path.exists(os.path.join("Datasets", dataset_name, dataset_name + "_embeddings.json")):
                #Load in memory statistics
                with open(os.path.join("Datasets", dataset_name, dataset_name + "_embeddings.json")) as f:
                    self.statistics = json.load(f)
            else:
                # Load Statistics from disk
                self.statistics = StatisticsLoader(os.path.join("Datasets", dataset_name, "statistics"))
            print("Successfully loaded statistics")
        except:
            print("No statistics found")
            exit()




    def train_GNN(self, train_data, test_data, epochs=100, train=True):
        """
        Train the model on the given train_data, or evaluate on the given test_data
        :param train_data: training data in the form of a list of query dicts
        :param test_data: test data in the form of a list of query dicts
        :param epochs: number of epochs to train for
        :param train: if True, train the model, if False, evaluate the model
        :return: None
        """
        print("Starting Training...")
        test_mae = []
        test_q_error = []
        min_q_error = 9999999
        min_mae = 9999999
        #model = GINmodel().to(self.device).double()
        model = TripleModel().to(self.device).double()

        # Start from a checkpoint
        try:
            model.load_state_dict(torch.load("model.pth"))
        except Exception as e:
            print("No checkpoint found, starting with random weights")

        print("Number of Parameters: ", sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss = MSELoss()

        if train:
            for epoch in tqdm(range(epochs)):
                epoch_loss = 0
                points_processed = 0
                i = 0

                model.train()
                for datapoint in train_data:
                    i += 1

                    # Get graph representation of query
                    data = get_query_graph_data_new(datapoint, self.statistics, self.device)

                    # Transform graph to undirected representation, with feature indicating edge direction
                    data = ToUndirectedCustom(merge=False)(data)
                    data = data.to_homogeneous()
                    data = data.to(self.device)

                    # Predict logarithm of cardinality
                    out = model(data.x.double(), data.edge_index, data.edge_type, data.edge_attr.double())

                    y = np.log(datapoint["y"])
                    y = torch.tensor(y)

                    # Calculate loss
                    l = loss(out, torch.tensor(y).to(self.device))

                    l.backward()
                    points_processed += 1
                    # Gradient Accumulation
                    if points_processed > 32:
                        optimizer.step()
                        optimizer.zero_grad()
                        points_processed = 0

                    epoch_loss += l.item()
                print(epoch_loss / len(train_data))


                # Evaluating on test set:
                abs_errors = []
                q_errors = []
                preds = []
                gts = []

                model.eval()
                for datapoint in test_data:
                    data = get_query_graph_data_new(datapoint, self.statistics, self.device)
                    data = ToUndirectedCustom(merge=False)(data)
                    data = data.to_homogeneous()
                    data = data.to(self.device)

                    out = model(data.x.double(), data.edge_index, data.edge_type, data.edge_attr.double())

                    y = datapoint["y"]

                    pred = out.detach().cpu().numpy()
                    # As model predicts logarithm, scale accordingly
                    pred = np.exp(pred)
                    preds.append(pred)
                    gts.append(y)
                    abs_errors.append(np.abs(pred - y))
                    q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
                    points_processed += 1

                # Calculate mean absolute error and q-error
                print('MAE: ', np.mean(abs_errors))
                test_mae.append(np.mean(abs_errors))
                print('Qerror: ', np.mean(q_errors))
                test_q_error.append(np.mean(q_errors))
                # Save model if it is the best so far
                if (np.mean(q_errors) < min_q_error):
                    torch.save(model.state_dict(), "model.pth")
                    min_q_error = np.mean(q_errors)
                if (np.mean(abs_errors) < min_mae):
                    torch.save(model.state_dict(), "model_mae.pth")
                    min_mae = np.mean(abs_errors)


        # Evaluation of the best model on the test set
        model.load_state_dict(torch.load("model.pth"))
        abs_errors = []
        q_errors = []
        preds = []
        gts = []
        sizes = []

        model.eval()
        # List to store execution times
        exec_times = []
        for datapoint in test_data:
            data = get_query_graph_data_new(datapoint, self.statistics, self.device)
            data = ToUndirectedCustom(merge=False)(data)
            data = data.to_homogeneous()
            data = data.to(self.device)

            # Measure execution time of model
            start = time.time()
            out = model(data.x.double(), data.edge_index, data.edge_type, data.edge_attr.double())
            end = time.time()
            exec_times.append((end - start) * 1000) # Convert to ms
            sizes.append(len(datapoint["triples"]))
            pred = out.detach().cpu().numpy()[0][0]

            y = datapoint["y"]
            pred = np.exp(pred)

            preds.append(pred)
            gts.append(y)
            y = torch.tensor(y).double()
            abs_errors.append(np.abs(pred - y))
            q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))

        print("Mean Absolute Error: ", np.mean(abs_errors))
        print("Mean Q-Error: ", np.mean(q_errors))

        print("Mean execution time: ", np.mean(exec_times))

        # Saving Raw Results to File
        np.save(os.path.join("Datasets", self.dataset_name, "Results", "preds.npy"), preds)
        np.save(os.path.join("Datasets", self.dataset_name, "Results", "gts.npy"), gts)
        np.save(os.path.join("Datasets", self.dataset_name, "Results", "sizes.npy"), sizes)
        np.save(os.path.join("Datasets", self.dataset_name, "Results", "pred_times.npy"), exec_times)



        # Plotting results
        plt.plot(gts, preds, "x")
        plt.xlabel("True Cardinality")
        plt.ylabel("Predicted Cardinality")
        plt.show()
        plt.savefig("cardinality_plot.pdf")

        plt.plot(test_q_error)
        plt.xlabel("Epoch")
        plt.ylabel("Q Error")
        plt.savefig("QError.pdf")
        plt.show()

        plt.plot(test_mae)
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.savefig("MAE.pdf")
        plt.show()


if __name__ == "__main__":
    model = cardinality_estimator("yago", None, sim_measure="cosine")

    # Loading cleaned Dataset:
    with open("Datasets/yago/path/Joined_Queries.json") as f:
        data = json.load(f)

    random.Random(4).shuffle(data)
    train_data = data[:int(0.8 * len(data))]
    test_data = data[int(0.8 * len(data)):]

    # Out of Distribution Testing
    # with open("Datasets/lubm/disjoint_train.json") as f:
    #     train_data = json.load(f)
    # with open("Datasets/lubm/disjoint_test.json") as f:
    #     test_data = json.load(f)

    # Train or evaluate the model on the dataset
    model.train_GNN(train_data, test_data, epochs=100, train=True)







