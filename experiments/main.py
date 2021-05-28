import pickle
import torch
import os

from samplefree_ood.datasets import __DATASETS__, get_transform

from samplefree_ood.architectures import __ARCHITECTURES__, \
    __ARCHITECTURES_224__
from samplefree_ood.experiments import OODStreamerWithSummaries

NUM_WORKER = 4
BATCH_SIZE = 256

def run(dataset, architecture, network_path, base_db, random_state):
    excluded = "in-dms:in-dss:in-dms-aos:in-nota:nota:proj+".split(":")

    # =========================== PATH =============================== #
    latent_folder = os.path.expanduser("~/data/latent/{}_{}_{}_{}"
                                       "".format(base_db, architecture,
                                                 random_state, dataset))

    folder = os.path.expanduser("~/data/samplefree_ood/")
    uniform_path = os.path.join(folder, "{}_{}_{}_uniform_full.npy"
                                        "".format(base_db,
                                                  architecture,
                                                  random_state))

    result_path = os.path.expanduser("~/data/results/{}_{}_{}_{}"
                                     "".format(base_db, architecture,
                                               random_state, dataset))

    # ============================= RAND ================================= #
    cuda = torch.cuda.is_available()
    torch.manual_seed(random_state + 42)
    if cuda:
        torch.cuda.manual_seed(random_state + 101)

    # =========================== ARCHITECTURES =============================== #
    ARCHITECTURES = __ARCHITECTURES__

    if architecture.endswith("_small"):
        ARCHITECTURES = __ARCHITECTURES__
    if architecture.endswith("_large"):
        ARCHITECTURES = __ARCHITECTURES_224__


    # ============================== MODELS ============================== #
    print("Loading model from", network_path)

    id_db_factory = __DATASETS__[base_db]
    print(id_db_factory, id_db_factory.get_default_n_outputs())
    model = ARCHITECTURES[architecture.split("_")[0]](
        n_outputs=id_db_factory.get_default_n_outputs())

    model.load_state_dict(torch.load(network_path, map_location="cpu"))
    model.eval()

    # =========================== DATA =================================== #
    id_db = id_db_factory(shape=model.input_size)

    actual_db = base_db if dataset in {"train", "val", "test"} else dataset
    whole_dataset = __DATASETS__[actual_db].same_as(id_db)

    train_set, val_set, test_set = whole_dataset.get_ls_vs_ts()

    if dataset == "train":
        investigated_set = train_set
    elif dataset == "val":
        investigated_set = val_set
    elif dataset == "test":
        investigated_set = test_set
    else:
        investigated_set = torch.utils.data.ConcatDataset([train_set,
                                                           val_set,
                                                           test_set])

    # ============================= COMPUTATION ================================ #
    streamer = OODStreamerWithSummaries.from_uniform(id_db,
                                                     excluded=excluded,
                                                     ref_save_path=uniform_path,
                                                     num_workers=NUM_WORKER,
                                                     batch_size=BATCH_SIZE)

    whole_cache = None
    for completion, cache in streamer.stream(model, investigated_set,
                                             latent_folder,
                                             batch_size=BATCH_SIZE,
                                             num_workers=NUM_WORKER):
        whole_cache = cache
        # `completion` hook to display progress

    if whole_cache is None:
        raise RuntimeError("Nothing to save.")


    # ============================= RESULTS ================================ #
    collector = {}
    for key, value in whole_cache:
        collector[key] = value

    with open(result_path, "wb") as hdl:
        pickle.dump(collector, hdl, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="OOD dataset")
    parser.add_argument("architecture", help="Name of the architecture")
    parser.add_argument("network_path", help="Path to the learned model")
    parser.add_argument("--base_db", help="ID dataset", default="cifar10")
    parser.add_argument("--random_state", help="ID dataset", default=0, type=int)

    args = parser.parse_args()

    run(args.dataset, args.architecture, args.network_path, args.base_db, args.random_state)
