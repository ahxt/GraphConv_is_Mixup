import logging
import torch
from utils import load_dataset, get_laplacian
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul
from typing import Callable, Optional, Tuple
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter_add
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Optional, Tuple
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_sparse import spmm
import torch.utils.data as data_utils
from torch.nn import Linear
import torch.nn.functional as F
import os
import time
import math
import random
import argparse
import numpy as np




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def arg_parse():
    parser = argparse.ArgumentParser(description="GNNisMixup")
    parser.add_argument("--dataset", type=str, default="Cora", required=True, choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--dataset_dir", type=str, default="/data/han/graph")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")  # 5e-4
    parser.add_argument("--log_screen", type=bool, default=True)
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200, help="number of training the one shot model")
    parser.add_argument(
        "--data_split",
        type=str,
        default="public",
        choices=["public", "full", "random"],
        help="public: public split for Cora, CiteSeer, PubMed; full:, ranom: randomly split 60:20:20",
    )
    parser.add_argument("--data_split_ratio", type=float, default=0.6, help="random public split ration")
    parser.add_argument("--train_mixup_type", type=str, default="nomixup", choices=["mlp", "mlp_ymixup", "gnn"])
    parser.add_argument("--dropout", type=float, default=0.2, help="input feature dropout")
    parser.add_argument("--norm_features", type=bool, default=True, help="input feature dropout")

    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--project", type=str, default="", required=True, help="input project name")
    parser.add_argument("--random_seed", type=int, default=31)
    parser.add_argument("--job_id", type=str, default="")

    parser.add_argument("--use_checkpoint", type=bool, default=False)
    parser.add_argument("--pretrained_checkpoint", type=str, default="no_pretrained_checkpoint")
    parser.add_argument("--save_dir", type=str, default="/home/grads/h/han/workspace/mlpinit/saved")
    parser.add_argument("--cuda", type=bool, default=True, required=False, help="run in cuda mode")
    parser.add_argument("--cuda_num", type=int, default=0, help="GPU number")
    parser.add_argument("--eval_steps", type=int, default=5, help="interval steps to evaluate model performance")
    parser.add_argument("--multi_label", type=bool, default=False, help="multi_label or single_label task")
    parser.add_argument("--save_gnn", type=bool, default=True)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--round", type=int, default=0)

    args = parser.parse_args()

    return args


def set_seed(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):

    fill_value = 2.0 if improved else 1.0

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        kwargs.setdefault("aggr", "add")
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), self.improved, self.add_self_loops
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class GCNConv_MLP(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):

        kwargs.setdefault("aggr", "add")
        super(GCNConv_MLP, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """"""
        x = x @ self.weight

        out = x

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class GNN(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, normalize=True):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, inter_channels, cached=True, normalize=normalize)
        self.conv2 = GCNConv(inter_channels, out_channels, cached=True, normalize=normalize)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, normalize=False):
        super(MLP, self).__init__()
        self.conv1 = GCNConv_MLP(in_channels, inter_channels, cached=True, normalize=normalize)
        self.conv2 = GCNConv_MLP(inter_channels, out_channels, cached=True, normalize=normalize)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, training=self.training, p = 0.8)
        # x = F.relu( self.conv2(x, edge_index, edge_weight) )
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class LabelPropagation(MessagePassing):
    def __init__(self, num_layers: int, alpha: float):
        super().__init__(aggr="add")
        self.num_layers = num_layers
        self.alpha = alpha

    @torch.no_grad()
    def forward(
        self,
        y: Tensor,
        edge_index: Adj,
        mask: Optional[Tensor] = None,
        edge_weight: OptTensor = None,
        post_step: Callable = lambda y: y.clamp_(0.0, 1.0),
    ) -> Tensor:
        """"""

        if y.dtype == torch.long:
            y = F.one_hot(y.view(-1)).to(torch.float)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=None, size=None)
            # out.mul_(self.alpha).add_(res)
            # out = post_step(out)

            out[mask] += y[mask]

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_layers={self.num_layers}, " f"alpha={self.alpha})"


if __name__ == "__main__":
    args = arg_parse()

    dataset = args.dataset
    dataset_dir = args.dataset_dir
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    log_screen = args.log_screen
    log_dir = args.log_dir
    num_layers = args.num_layers
    epochs = args.epochs
    dim_hidden = args.dim_hidden
    random_seed = args.random_seed
    norm_features = args.norm_features

    train_mixup_type = args.train_mixup_type
    data_split = args.data_split
    data_split_ratio = args.data_split_ratio

    set_seed(random_seed)

    if args.log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if args.log_dir != "":
        log_dir = os.path.join(args.log_dir, args.dataset)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file_name = f"/{args.exp_name}_{time.time()}.log"
        fh = logging.FileHandler(log_dir + log_file_name)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    logger.info(f"args: {args}")

    path = os.path.join(dataset_dir, dataset)

    if norm_features == True:
        transform = T.NormalizeFeatures()
    else:
        transform = None

    data = load_dataset(path, dataset, transform=transform)

    data.y = F.one_hot(data.y.squeeze())
    logger.info(f"data loaded: {data}")

    # path = osp.join('/fsx/han3/graph', dataset)
    # dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    # # dataset = Planetoid(path, dataset)
    # data = dataset[0]
    # data.y = F.one_hot( data.y.squeeze() )

    if data_split == "full":
        val_mask = data.val_mask
        test_mask = data.test_mask
        train_mask = ~(val_mask + test_mask)

    elif data_split == "public":
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    elif data_split == "random":
        train_num = math.ceil(data.x.shape[0] * data_split_ratio)
        test_num = math.ceil(data.x.shape[0] * (1 - data_split_ratio) / 2)
        val_num = data.x.shape[0] - train_num - test_num
        mask = np.array([1] * train_num + [2] * val_num + [3] * test_num)
        mask = np.random.permutation(mask)
        mask = torch.tensor(mask)

        train_mask = mask == 1
        val_mask = mask == 2
        test_mask = mask == 3

    else:
        raise NotImplementedError("Please input data_split!!")

    data = data.to(device)
    num_features = data.x.shape[1]
    num_classes = data.y.shape[1]
    num_nodes = data.x.shape[0]

    if train_mixup_type in ["mlp_ymixup", "mlp"]:
        model = MLP(in_channels=num_features, inter_channels=dim_hidden, out_channels=num_classes).to(device)
    elif train_mixup_type in ["gnn"]:
        model = GNN(in_channels=num_features, inter_channels=dim_hidden, out_channels=num_classes).to(device)
    else:
        raise NotImplementedError("Please input train_mixup_type!!")

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)

    def mixup_cross_entropy_loss(input, target, size_average=True):
        loss = -torch.sum(input * target)
        return loss / input.size(0) if size_average else loss

    def train_lp(model, data, lp_y):
        model.train()
        optimizer.zero_grad()
        y_hat = model(data)
        y_hat = y_hat[torch.sum(lp_y, 1) != 0]
        lp_y = lp_y[torch.sum(lp_y, 1) != 0]
        loss = mixup_cross_entropy_loss(y_hat, lp_y)
        loss.backward()
        optimizer.step()
        return model

    @torch.no_grad()
    def test(model, data):
        model.eval()
        accs = []
        losses = []
        logits = model(data)
        for mask in [train_mask, val_mask, test_mask]:

            loss = mixup_cross_entropy_loss(logits[mask], data.y[mask])

            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask].argmax(dim=-1, keepdim=True).squeeze()).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs + losses

    X = data.x
    Y = data.y
    Y = Y.type(torch.float32)

    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)

    _, edge_attr_norm = get_laplacian(edge_index, normalization="rw", num_nodes=num_nodes, edge_weight=None)

    Y_tmp = Y.clone()
    Y_tmp[~train_mask] = 0

    y_mix = Y_tmp
    y_mix = spmm(index=edge_index, value=edge_attr_norm, m=num_nodes, n=num_nodes, matrix=y_mix)
    y_mix = spmm(index=edge_index, value=edge_attr_norm, m=num_nodes, n=num_nodes, matrix=y_mix)
    y_mix = y_mix / (torch.sum(y_mix, 1).unsqueeze(-1) + 1e-10)

    logger.info("Mixed y_train!!")

    if train_mixup_type == "mlp_ymixup":
        lp_y = y_mix
    else:
        lp_y = Y.clone()
        lp_y[~train_mask] = 0

    best_mlp_val_acc = best_mlp_test_acc = 0
    best_gnn_val_acc = best_gnn_test_acc = 0
    best_mlp_epoch = 0
    best_gnn_epoch = 0
    for epoch in range(1, 401):

        model = train_lp(model, data, lp_y=lp_y)

        model_mlp = MLP(in_channels=num_features, inter_channels=dim_hidden, out_channels=num_classes).to(device)
        model_mlp.load_state_dict(model.state_dict())
        mlp_train_acc, mlp_val_acc, mlp_test_acc, mlp_train_loss, mlp_val_loss, mlp_test_loss = test(model_mlp, data)

        model_gnn = GNN(in_channels=num_features, inter_channels=dim_hidden, out_channels=num_classes).to(device)
        model_gnn.load_state_dict(model.state_dict())

        gnn_train_acc, gnn_val_acc, gnn_test_acc, gnn_train_loss, gnn_val_loss, gnn_test_loss = test(model_gnn, data)

        if mlp_val_acc > best_mlp_val_acc:
            best_mlp_val_acc = mlp_val_acc
            best_mlp_test_acc = mlp_test_acc
            best_mlp_epoch = epoch

            best_gnn_val_acc = gnn_val_acc
            best_gnn_test_acc = gnn_test_acc
            best_gnn_epoch = epoch

        epoch_dict = vars(args)
        epoch_dict["epoch"] = epoch
        epoch_dict["mlp_train_acc"] = mlp_train_acc
        epoch_dict["mlp_val_acc"] = mlp_val_acc
        epoch_dict["mlp_test_acc"] = mlp_test_acc
        epoch_dict["mlp_train_loss"] = mlp_train_loss
        epoch_dict["mlp_val_loss"] = mlp_val_loss
        epoch_dict["mlp_test_loss"] = mlp_test_loss
        epoch_dict["best_mlp_val_acc"] = best_mlp_val_acc
        epoch_dict["best_mlp_test_acc"] = best_mlp_test_acc
        epoch_dict["best_mlp_epoch"] = best_mlp_epoch

        epoch_dict["gnn_train_acc"] = gnn_train_acc
        epoch_dict["gnn_val_acc"] = gnn_val_acc
        epoch_dict["gnn_test_acc"] = gnn_test_acc
        epoch_dict["gnn_train_loss"] = gnn_train_loss
        epoch_dict["gnn_val_loss"] = gnn_val_loss
        epoch_dict["gnn_test_loss"] = gnn_test_loss
        epoch_dict["best_gnn_val_acc"] = best_gnn_val_acc
        epoch_dict["best_gnn_test_acc"] = best_gnn_test_acc
        epoch_dict["best_gnn_epoch"] = best_gnn_epoch

        logger.info(f"epoch_dict: {epoch_dict}")

logger.info("Experiment Done!!")
