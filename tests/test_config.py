from sparse_st_fgw.config import BestConfig


def test_best_configuration_is_locked():
    config = BestConfig()
    assert config.gnn_hidden == 64
    assert config.gnn_consistency_weight == 0.1
    assert config.gnn_norm is True
    assert config.gnn_activation is True
    assert config.gnn_dropout == 0.1
    assert config.temporal_topk == 3
    assert config.temporal_solver == "masked_fgw"
    assert config.final_knn == 64
    assert config.fgw_weight == 0.6
    assert config.temporal_epsilon == config.final_epsilon == 0.002
    assert config.rho_a == 0.005
    assert config.rho_b == 0.30
    assert config.ot_device == "cuda:0"
