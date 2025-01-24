"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
from dataloader import BasicDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class SEPGCN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(SEPGCN, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.num_categories = self.dataset.c_categories

        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']

        self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
        self.embedding_category = nn.Embedding(self.num_categories, self.latent_dim*2)

        self._initialize_weights()

        self.f = nn.Sigmoid()

        self.graph = self.dataset.getSparseGraph()
        self.graphX = self.dataset.getSparseTemporalGraph()

        self.train_items = self.dataset.train_items
        self.train_users = self.dataset.train_users

        self.train_temporal = self.dataset.train_temporal
        self.item_categories = self.dataset.get_item_categories
        # Prepare item-to-category mapping
        max_item_id = max(self.item_categories.keys())
        self.item_to_category = torch.zeros(max_item_id + 1, dtype=torch.long, device=world.device)
        for item, category in self.item_categories.items():
            self.item_to_category[item] = category
        print(f"LightGCN initialized with n_layers={self.n_layers}, dropout={self.config['dropout']}")

    def _initialize_weights(self):
        init_method = self.config.get('weight_init', 'normal')

        def init_fn(tensor):
            if init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(tensor)
            elif init_method == 'normal':
                nn.init.normal_(tensor, mean=0.0, std=0.1)

        for embedding in [self.embedding_user, self.embedding_item, self.embedding_category]:
            init_fn(embedding.weight)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        cat_emb = self.embedding_category.weight

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb) #(n+m)
            users, items = torch.split(all_emb, [self.num_users, self.num_items])
            if i == 0:
                user_item_features = torch.cat([users[self.train_users], items[self.train_items]], dim=-1) + cat_emb[self.item_to_category[self.train_items]]
            else:
                user_item_features = torch.cat([users[self.train_users], items[self.train_items]], dim=-1)
            user_item_features = torch.sparse.mm(self.graphX, user_item_features)
            all_emb = self.update_node_embedding_from_edge(user_item_features, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs[1:], dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items

    def update_node_embedding_from_edge(self, edge_emb, node_emb):
        device = world.device
        train_users = torch.tensor(self.train_users, device=device)
        train_items = torch.tensor(self.train_items, device=device)
        degree = torch.cat([
            torch.bincount(train_users, minlength=self.num_users),
            torch.bincount(train_items, minlength=self.num_items)
        ]).to(device)

        updated_node_emb = torch.zeros_like(node_emb, device=device)
        updated_node_emb.index_add_(
            0,
            train_users,
            edge_emb[:, :self.latent_dim]
        )
        updated_node_emb.index_add_(
            0,
            train_items + self.num_users,
            edge_emb[:, self.latent_dim:]
        )

        updated_node_emb = updated_node_emb / (degree.unsqueeze(-1) + 1e-9)
        updated_node_emb = (updated_node_emb + node_emb) / 2.0
        return updated_node_emb

    def getEdgeTimestamps(self, edge_indices):
        user_to_item_indices = edge_indices[:, (edge_indices[0] < self.num_users) & (edge_indices[1] >= self.num_users)]
        u_ids = user_to_item_indices[0]
        i_ids = user_to_item_indices[1] - self.num_users
        flat_indices = u_ids * self.num_items + i_ids
        return self.edge_time_map_tensor[flat_indices]

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)

        neg_scores = torch.sum(users_emb * neg_emb, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
