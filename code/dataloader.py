import os
from collections import defaultdict
from os.path import join
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from torch import cosine_similarity
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

def calculate_time_hours_id(timestamps):
    HOURS_IN_A_DAY = 24
    DAYS_IN_A_WEEK = 7

    timestamps_in_hours = timestamps // 3600
    hours_in_week = timestamps_in_hours % HOURS_IN_A_DAY
    days_in_week = (timestamps_in_hours // HOURS_IN_A_DAY) % DAYS_IN_A_WEEK
    time_id_week = hours_in_week * DAYS_IN_A_WEEK + days_in_week

    return time_id_week


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def c_categories(self):
        raise NotImplementedError

    @property
    def train_users(self):
        raise NotImplementedError

    @property
    def train_items(self):
        raise NotImplementedError

    @property
    def train_temporal(self):
        raise NotImplementedError

    @property
    def get_item_categories(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        raise NotImplementedError

    def getSparseGraph(self):
        raise NotImplementedError

    def getSparseTemporalGraph(self):
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """

    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData -= 1
        testData -= 1
        self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")

        # (users,users)
        self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
                                    shape=(self.n_users, self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos


    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data,
                                                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()
            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)


class Loader(BasicDataset):
    def __init__(self, config=world.config, path="../data/nyc"):
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.n_cluster = config['n_cluster']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        if not os.path.exists(f"{path}/cat.txt"):
            print("cat.txt not found. Starting clustering...")
            self._cluster_items_by_time_with_cosine(
                train_file=f"{path}/train.txt",
                time_file=f"{path}/time_train.txt",
                output_file=f"{path}/cat.txt",
                n_clusters=self.n_cluster
            )
            # time_clustering = TimeClustering(f"{path}/train.txt", f"{path}/time_train.txt", f"{path}/cat.txt", n_clusters=100)
            # time_clustering.cluster_items()
        else:
            print("cat.txt already exists. Skipping clustering.")

        self.data_loader = DataLoader(path)
        self.data_loader.get_statistics()

        self.n_user = self.data_loader.n_users
        self.m_item = self.data_loader.n_items
        self.train_user = self.data_loader.train_user
        self.train_item = self.data_loader.train_item
        self.train_time = self.data_loader.train_time
        self.c_category = self.data_loader.n_categories
        self.trainDataSize = self.data_loader.train_data_size
        self.testDataSize = self.data_loader.test_data_size
        self.item_categories = self.data_loader.item_categories
        self.item_locations = self.data_loader.item_locations
        self.cat_map = self.data_loader.cat_map

        self.graph_builder = GraphBuilder(self.data_loader, split=self.split, folds=self.folds)
        self.UserItemNet = self.graph_builder.UserItemNet
        self.Graph = None
        self.TemporalGraph = None

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

        print(f"Dataset Loaded. {self.trainDataSize} training interactions, {self.testDataSize} testing interactions.")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def c_categories(self):
        return self.c_category

    @property
    def train_users(self):
        return self.graph_builder.train_user_interaction

    @property
    def train_items(self):
        return self.graph_builder.train_item_interaction

    @property
    def train_temporal(self):
        return self.graph_builder.train_time_interaction

    @property
    def get_item_categories(self):
        return self.item_categories

    @property
    def trainDataSize(self):
        return self._trainDataSize

    def getSparseGraph(self):
        if self.Graph is None:
            print("Building Basic Graph...")
            self.Graph = self.graph_builder.build_basic_graph()
        return self.Graph

    def getSparseTemporalGraph(self):
        if self.TemporalGraph is None:
            print("Building Combined Graph...")
            self.TemporalGraph = self.graph_builder.build_combined_graph(threshold=0.0)
            # self.TemporalGraph = self.graph_builder.build_combined_graph()
        return self.TemporalGraph

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    @property
    def allPos(self):
        return self._allPos

    def __build_test(self):
        test_data = {}
        for user, item in zip(self.data_loader.test_user, self.data_loader.test_item):
            if user in test_data:
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    @trainDataSize.setter
    def trainDataSize(self, value):
        self._trainDataSize = value

    @property
    def testDict(self):
        return self.__testDict

    def _cluster_items_by_time_with_cosine(self, train_file, time_file, output_file, n_clusters=100):
        print("Loading training and time data...")

        with open(train_file, 'r') as f, open(time_file, 'r') as tf:
            train_lines = f.readlines()
            time_lines = tf.readlines()

        item_timestamps = defaultdict(list)
        seconds_in_week = 7 * 24 * 60 * 60
        for l, t in zip(train_lines, time_lines):
            l = l.strip().split()
            t = t.strip().split()

            items = list(map(int, map(float, l[1:])))
            timestamps = list(map(float, t[1:]))

            for item, timestamp in zip(items, timestamps):
                hour_in_week = int((timestamp % seconds_in_week) / 3600)
                item_timestamps[item].append(hour_in_week)

        all_items = list(item_timestamps.keys())
        time_vectors = list(item_timestamps.values())

        print("Computing Cosine Similarity matrix in batches...")

        time_array = np.array([np.bincount(tv, minlength=168) for tv in time_vectors])

        time_tensor = torch.tensor(time_array, dtype=torch.float32)

        num_vectors = time_tensor.size(0)

        batch_size = 100
        similarity_matrix = torch.zeros((num_vectors, num_vectors), dtype=torch.float32)

        for i in range(0, num_vectors, batch_size):
            for j in range(0, num_vectors, batch_size):
                batch_i = time_tensor[i:i + batch_size]
                batch_j = time_tensor[j:j + batch_size]

                sim_batch = cosine_similarity(batch_i.unsqueeze(1), batch_j.unsqueeze(0), dim=-1)

                similarity_matrix[i:i + batch_size, j:j + batch_size] = sim_batch

        print("Cosine Similarity matrix computation complete.")

        print("Clustering items using KMeans with Cosine Similarity...")
        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0
        )
        labels = clustering.fit_predict(similarity_matrix)
        print(f"Number of clusters: {n_clusters}")

        print("Writing results to file...")
        with open(output_file, 'w') as cat_file:
            for item, label in zip(all_items, labels):
                cat_file.write(f"{item} {label}\n")

        print(f"cat.txt created with {n_clusters} clusters at {output_file}")



class DataLoader:
    def __init__(self, path):
        cprint(f'Loading dataset from: [{path}]')
        self.path = path

        self.train_file = f"{path}/train.txt"
        self.test_file = f"{path}/test.txt"
        self.time_file = f"{path}/time_train.txt"
        self.category_file = f"{path}/cat.txt"
        self.location_file = f"{path}/location.txt"

        self.n_users = 0
        self.n_items = 0
        self.n_categories = 0

        self.train_data_size = 0
        self.test_data_size = 0
        self.item_categories = {}
        self.item_locations = {}

        self.train_user = []
        self.train_item = []
        self.test_user = []
        self.test_item = []
        self.train_time = []
        self._load_train_data()
        self._load_test_data()
        self._load_categories()
        self._load_item_locations()
        self.cat_map = self.load_cat_map(f"{path}/cat.txt")
        print("Data loading complete.")

    def _load_train_data(self):
        print("Loading training data...")
        with open(self.train_file, 'r') as f, open(self.time_file, 'r') as tf:
            train_lines = f.readlines()
            time_lines = tf.readlines()

            for l, t in zip(train_lines, time_lines):
                l = l.strip().split()
                t = t.strip().split()
                user = int(float(l[0]))
                items = list(map(int, map(float, l[1:])))
                timestamps = list(map(int, map(float,t[1:])))

                self.train_user.extend([user] * len(items))
                self.train_item.extend(items)
                self.train_time.extend(timestamps)

                self.n_users = max(self.n_users, user)
                self.n_items = max(self.n_items, *items)
                self.train_data_size += len(items)

    def _load_test_data(self):
        print("Loading testing data...")
        with open(self.test_file, 'r') as f:
            for line in f:
                l = line.strip().split()
                user = int(float(l[0]))
                items = list(map(int, map(float, l[1:])))
                self.test_user.extend([user] * len(items))
                self.test_item.extend(items)

                self.n_users = max(self.n_users, user)
                self.n_items = max(self.n_items, *items)
                self.test_data_size += len(items)
        self.n_users += 1
        self.n_items += 1

    def _load_categories(self):
        print("Loading item categories...")
        with open(self.category_file, 'r') as f:
            for line in f:
                item, category = map(int, line.strip().split())
                self.item_categories[item] = category
                self.n_categories = max(self.n_categories, category)
        self.n_categories += 1

    def _load_item_locations(self):
        print("Loading item locations...")
        with open(self.location_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                item_id = int(parts[0])
                lat, lon = float(parts[1]), float(parts[2])
                self.item_locations[item_id] = (lat, lon)

    def load_cat_map(self, cat_file):
        cat_map = {}
        with open(cat_file, 'r') as f:
            for line in f:
                item, cat = map(int, line.strip().split())
                cat_map[item] = cat
        return cat_map

    def get_statistics(self):
        print("===== Data Statistics =====")
        print(f"Number of Users: {self.n_users}")
        print(f"Number of Items: {self.n_items}")
        print(f"Number of Categories: {self.n_categories}")
        print(f"Training Data Size: {self.train_data_size}")
        print(f"Testing Data Size: {self.test_data_size}")
        sparsity = 1 - (self.train_data_size / (self.n_users * self.n_items))
        print(f"Data Sparsity: {sparsity:.6f}")
        print("===========================")


class GraphBuilder:
    def __init__(self, stats, split=False, folds=100000, device=world.device):
        self.stats = stats
        self.split = split
        self.folds = folds
        self.device = device
        self.Graph = None
        self.CombinedGraph = None
        self.UserItemNet = self._build_user_item_matrix()
        self.train_user_interaction = None
        self.train_item_interaction = None
        self.train_time_interaction = None

    def _build_user_item_matrix(self):
        try:
            interaction_mat = sp.load_npz(self.stats.path + '/interaction_mat.npz')
            print("Successfully loaded existing interaction matrix.")
            return interaction_mat
        except:
            print("Building User-Item matrix...")
            interaction_mat = csr_matrix(
                (np.ones(len(self.stats.train_user)), (self.stats.train_user, self.stats.train_item)),
                shape=(self.stats.n_users, self.stats.n_items)
            )
            sp.save_npz(self.stats.path + '/interaction_mat.npz', interaction_mat)
            return interaction_mat

    def build_basic_graph(self):
        print("Loading adjacency matrix...")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.stats.path + '/adj_mat.npz')
                print("Successfully loaded existing adjacency matrix.")
                norm_adj = pre_adj_mat
            except:
                print("Generating new adjacency matrix...")
                s = time()
                adj_mat = sp.dok_matrix((self.stats.n_users + self.stats.n_items,
                                         self.stats.n_users + self.stats.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()

                R = self.UserItemNet.tolil()
                adj_mat[:self.stats.n_users, self.stats.n_users:] = R
                adj_mat[self.stats.n_users:, :self.stats.n_users] = R.T

                norm_adj = self._normalize_adjacency_matrix(adj_mat)

                sp.save_npz(self.stats.path + '/adj_mat.npz', norm_adj)
                end = time()
                print(f"Graph generated in {end - s:.2f}s.")

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)
        return self.Graph

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = len(self.stats.train_user) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = len(self.stats.train_user)
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def build_combined_graph(self, threshold=0.0):
        print("Loading Combined Graph...")
        if self.CombinedGraph is None:
            train_data = list(zip(self.stats.train_user, self.stats.train_item))
            time_ids = [calculate_time_hours_id(v) for v in self.stats.train_time]

            interaction_time = {}
            for i in range(len(train_data)):
                if train_data[i] in interaction_time:
                    interaction_time[train_data[i]].append(time_ids[i])
                else:
                    interaction_time[train_data[i]] = [time_ids[i]]
            try:
                combined_graph = sp.load_npz(self.stats.path + '/combined_mat.npz')
                print("Successfully loaded existing Combined Graph.")
                norm_adj = combined_graph
            except:
                print("Generating new Combined Graph...")
                s = time()
                location_weight_matrix = self._compute_location_weight_matrix()
                # Extract nodes and their lists
                nodes = list(interaction_time.keys())
                n_pairs = len(nodes)

                # Initialize data for COO matrix
                row = []
                col = []
                data = []

                interaction_sets = {node: set(values) for node, values in interaction_time.items()}

                for i, node_a in enumerate(nodes):
                    set_a = interaction_sets[node_a]
                    for j in range(i, n_pairs):  # Only iterate from i to avoid redundant computation
                        node_b = nodes[j]
                        set_b = interaction_sets[node_b]
                        if set_a & set_b:  # Check intersection
                            weight = location_weight_matrix[node_a[1], node_b[1]]  # Use second elements from nodes
                            if weight > threshold:
                                row.append(i)
                                col.append(j)
                                data.append(weight)
                                if i != j:  # Ensure symmetry
                                    row.append(j)
                                    col.append(i)
                                    data.append(weight)

                adj_mat = sp.coo_matrix((data, (row, col)), shape=(n_pairs, n_pairs)).tocsr()

                norm_adj = self._normalize_adjacency_matrix(adj_mat)

                sp.save_npz(self.stats.path + '/combined_mat.npz', norm_adj)
                print(f"Combined Graph generated in {time() - s:.2f}s")

            # self.CombinedGraph = self._split_A_hat(norm_adj)

            self.CombinedGraph = self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)
            self.train_user_interaction = [p[0] for p in interaction_time.keys()]
            self.train_item_interaction = [p[1] for p in interaction_time.keys()]
            self.train_time_interaction = [p for p in interaction_time.values()]
            # Calculate sparsity of the final Combined Graph
            sparsity = 1 - (norm_adj.nnz / (norm_adj.shape[0] * norm_adj.shape[1]))
            print(f"Sparsity of the final Combined Graph: {sparsity:.6f}")
            print(f"Shape of the final Combined Graph: {norm_adj.shape[0]} x {norm_adj.shape[1]}")

        return self.CombinedGraph

    def _normalize_adjacency_matrix(self, adj_matrix):
        adj_matrix = adj_matrix.todok()
        rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
        rowsum[rowsum == 0] = 1e-10
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_matrix).dot(d_mat)
        return norm_adj.tocsr()

    def _compute_category_weight_matrix(self):
        print("Building Simple Category Graph...")

        # Lấy thông tin về số lượng danh mục và số lượng items
        n_categories = self.stats.n_categories
        n_items = self.stats.n_items

        # Tạo ma trận danh mục-mục
        category_item_matrix = np.zeros((n_items, n_categories))
        for item, category in self.stats.item_categories.items():
            category_item_matrix[item, category] = 1

        # Tính ma trận item-item dựa trên danh mục
        category_item_matrix = csr_matrix(category_item_matrix)
        item_similarity_matrix = category_item_matrix @ category_item_matrix.T

        # Chuyển đổi thành ma trận nhị phân: các item cùng category là 1, còn lại là 0
        item_similarity_matrix[item_similarity_matrix > 0] = 1

        print("Simple Category Graph built.")
        return csr_matrix(item_similarity_matrix)

    def _compute_location_weight_matrix(self, threshold=0.1, save_path='location_weight_matrix.npz'):
        try:
            location_graph = sp.load_npz(self.stats.path + '/' + save_path)
            print("Successfully loaded existing location weight matrix.")
        except:
            print("Computing location weight matrix...")
            distances = self._compute_item_pair_distances()
            sigma = self._compute_sigma()

            location_weights = np.exp(-distances / sigma)
            location_weights[location_weights < threshold] = 0
            location_graph = csr_matrix(location_weights)
            sp.save_npz(self.stats.path + '/' + save_path, location_graph)
            print("Location Graph built.")

        # Kiểm tra phần trăm các phần tử lớn hơn các ngưỡng
        location_weights = location_graph.toarray()  # Chuyển thành numpy array nếu cần
        thresholds = [0.9, 0.8, 0.5, 0.1]
        total_elements = location_weights.size

        for th in thresholds:
            count = np.sum(location_weights > th)
            percentage = (count / total_elements) * 100
            print(f"Percentage of elements > {th}: {percentage:.2f}%")

        return location_graph

    def _compute_sigma(self):
        user_trajectories = {}

        for user, item in zip(self.stats.train_user, self.stats.train_item):
            if user not in user_trajectories:
                user_trajectories[user] = []
            user_trajectories[user].append(item)

        all_distances = []

        def haversine_batch(coords):
            R = 6371
            latlon = np.radians(coords)
            lat1, lon1 = latlon[:, None, 0], latlon[:, None, 1]
            lat2, lon2 = latlon[None, :, 0], latlon[None, :, 1]
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            distances = R * c
            return distances[np.triu_indices_from(distances, k=1)]

        for user, items in user_trajectories.items():
            coords = np.array([self.stats.item_locations[item] for item in items if item in self.stats.item_locations])
            if len(coords) > 1:
                distances = haversine_batch(coords)
                all_distances.extend(distances)

        median = np.median(all_distances)
        sigma = -median / np.log(0.9)
        print(f"Computed Sigma (Median Distance): {sigma:.2f} km")
        return sigma

    def _compute_item_pair_distances(self):
        print("Computing item pair distances...")
        item_locations = self.stats.item_locations
        items = list(item_locations.keys())
        coords = np.array([item_locations[item] for item in items])

        R = 6371
        latlon = np.radians(coords)
        lat1, lon1 = latlon[:, None, 0], latlon[:, None, 1]
        lat2, lon2 = latlon[None, :, 0], latlon[None, :, 1]
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = R * c

        print("Item pair distances computed.")
        return distances

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        data = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(torch.stack([row, col]), data, torch.Size(coo.shape))