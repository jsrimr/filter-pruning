import abc
import re
import tempfile
import traceback
from typing import Tuple, Callable, Union, List, Optional

import kerassurgeon
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow.python.framework import ops

from typing import Callable, Optional, List

import numpy as np
from tensorflow.keras import models, layers
from sklearn import cluster, metrics

class BasePruning:
    _FUZZ_EPSILON = 1e-5

    def __init__(self,
                 pruning_factor: float,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Optional[Callable[[models.Model, int, int], None]],
                 nb_finetune_epochs: int,
                 nb_trained_for_epochs: int,
                 maximum_prune_iterations: int,
                 maximum_pruning_percent: float):

        self._pruning_factor = pruning_factor
        self._tmp_model_file_name = tempfile.NamedTemporaryFile().name

        self._model_compile_fn = model_compile_fn
        self._model_finetune_fn = model_finetune_fn

        self._nb_finetune_epochs = nb_finetune_epochs
        self._current_nb_of_epochs = nb_trained_for_epochs
        self._maximum_prune_iterations = maximum_prune_iterations
        self._maximum_pruning_percent = maximum_pruning_percent

        self._channel_number_bins = None
        self._pruning_factors_for_channel_bins = None

        self._original_number_of_filters = -1

        # TODO: select a subset of layers to prune
        self._prunable_layers_regex = ".*"

    def run_pruning(self, model: models.Model, prune_factor_scheduler_fn: Callable[[float, int], float] = None,
                    custom_objects_inside_model: dict = None) -> Tuple[models.Model, int]:
        self._original_number_of_filters = self._count_number_of_filters(model)

        pruning_iteration = 0

        while True:
            if prune_factor_scheduler_fn is not None:
                self._pruning_factor = prune_factor_scheduler_fn(self._pruning_factor, pruning_iteration)

            # Pruning step
            print("Running filter pruning {0}".format(pruning_iteration))
            model, pruning_dict = self._prune(model)

            # Computing statistics
            nb_of_pruned_filters = sum(pruning_dict.values())
            if nb_of_pruned_filters == 0:
                print("Number of pruned filters == 0, so pruning is stopped")
                break
            print("Number of pruned filters at this step: {0}".format(nb_of_pruned_filters))
            pruning_percent = self._compute_pruning_percent(model)
            print("Network is pruned from the original state, by {0} %".format(pruning_percent * 100))

            # Finetune step
            self._save_after_pruning(model)
            self._clean_up_after_pruning(model)
            model = self._load_back_saved_model(custom_objects_inside_model)
            self._model_compile_fn(model)
            if self._model_finetune_fn is not None:
                self._model_finetune_fn(model, self._current_nb_of_epochs,
                                        self._current_nb_of_epochs + self._nb_finetune_epochs)
            self._current_nb_of_epochs += self._nb_finetune_epochs

            # Stopping conditions
            if nb_of_pruned_filters < 1:
                print("No filters were pruned. Pruning is stopped.")
                break
            if self._maximum_pruning_percent is not None:
                if pruning_percent > self._maximum_pruning_percent:
                    print(
                        "Network pruning (currently {0} %) reached the maximum based on your definition ({1} %)".format(
                            pruning_percent * 100, self._maximum_pruning_percent * 100))
                    break
            pruning_iteration += 1

            if self._maximum_prune_iterations is not None:
                if pruning_iteration > self._maximum_prune_iterations:
                    break

        print("Pruning stopped.")
        return model, self._current_nb_of_epochs

    def define_prune_bins(self, channel_number_bins: Union[List[int], np.ndarray],
                          pruning_factors_for_bins: Union[List[float], np.ndarray]):
        if (len(channel_number_bins) - 1) != len(pruning_factors_for_bins):
            raise ValueError("While defining pruning bins, channel numbers list "
                             "should contain 1 more items than the pruning factor list")

        self._channel_number_bins = np.asarray(channel_number_bins).astype(int)
        self._pruning_factors_for_channel_bins = np.asarray(pruning_factors_for_bins).astype(float)

    def _get_pruning_factor_based_on_prune_bins(self, nb_channels: int) -> float:
        for i, pruning_factor in enumerate(self._pruning_factors_for_channel_bins):
            min_channel_number = self._channel_number_bins[i]
            max_channel_number = self._channel_number_bins[i + 1]
            if min_channel_number <= nb_channels < max_channel_number:
                return self._pruning_factors_for_channel_bins[i]
        # If we did not found any match we will return with the default pruning factor value
        print("No entry was found for a layer with channel number {0}, "
              "so returning pruning factor {1}".format(nb_channels, self._pruning_factor))
        return self._pruning_factor

    def _prune(self, model: models.Model) -> Tuple[models.Model, dict]:
        surgeon = kerassurgeon.Surgeon(model, copy=True)
        pruning_dict = dict()
        for layer in model.layers:
            if layer.__class__.__name__ == "Conv2D":
                if re.match(self._prunable_layers_regex, layer.name):
                    layer_weight_mtx = layer.get_weights()[0]
                    pruning_factor = self._pruning_factor
                    if self._pruning_factors_for_channel_bins is not None:
                        pruning_factor = self._get_pruning_factor_based_on_prune_bins(layer_weight_mtx.shape[-1])
                    filter_indices_to_prune = self.run_pruning_for_conv2d_layer(pruning_factor,
                                                                                layer,
                                                                                layer_weight_mtx)
                    # Remove selected filters from layer
                    surgeon.add_job("delete_channels", layer, channels=filter_indices_to_prune)
                    pruning_dict[layer.name] = len(filter_indices_to_prune)
        try:
            new_model = surgeon.operate()
        except Exception as e:
            print("Could not complete pruning step because got Exception: {0}".format(e))
            print(traceback.format_exc())
            return model, {k: 0 for k, _ in pruning_dict.items()}
        return new_model, pruning_dict

    @staticmethod
    def _count_number_of_filters(model: models.Model) -> int:
        nb_of_filters = 0
        for layer in model.layers:
            if layer.__class__.__name__ == "Conv2D":
                layer_weight_mtx = layer.get_weights()[0]
                _, _, _, channels = layer_weight_mtx.shape
                nb_of_filters += channels
        return nb_of_filters

    def _compute_pruning_percent(self, model: models.Model) -> float:
        nb_filters = self._count_number_of_filters(model)
        left_filters_percent = 1.0 - (nb_filters / self._original_number_of_filters)
        return left_filters_percent

    def _save_after_pruning(self, model: models.Model):
        model.save(self._tmp_model_file_name, overwrite=True, include_optimizer=True)

    @staticmethod
    def _clean_up_after_pruning(model: models.Model):
        del model
        K.clear_session()
        # tf.reset_default_graph()
        # https://stackoverflow.com/a/40782339/6128538
        ops.reset_default_graph()

    def _load_back_saved_model(self, custom_objects: dict) -> models.Model:
        model = models.load_model(self._tmp_model_file_name, custom_objects=custom_objects)
        return model

    @staticmethod
    def _apply_fuzz_to_vector(x: np.ndarray):
        # Prepare the vector element indices
        indices = np.arange(0, len(x), dtype=int)
        np.random.shuffle(indices)
        # Select the indices to be modified (always modify only N-1 values)
        nb_of_values_to_modify = np.random.randint(0, len(x) - 1)
        modify_indices = indices[:nb_of_values_to_modify]
        # Modify the selected elements of the vector
        x[modify_indices] += BasePruning._epsilon()

    @staticmethod
    def _apply_fuzz(x: np.ndarray):
        for i in range(len(x)):
            BasePruning._apply_fuzz_to_vector(x[i])

    @staticmethod
    def _epsilon():
        return BasePruning._FUZZ_EPSILON

    @staticmethod
    def _calculate_number_of_channels_to_keep(keep_factor: float, nb_of_channels: int) -> Tuple[int, int]:
        # This is the number of channels we would like to keep
        new_nb_of_channels = int(np.ceil(nb_of_channels * keep_factor))

        if new_nb_of_channels > nb_of_channels:
            # This happens when (factor > 1)
            new_nb_of_channels = nb_of_channels
        elif new_nb_of_channels < 1:
            # This happens when (factor <= 0)
            new_nb_of_channels = 1

        # Number of channels which will be removed
        nb_channels_to_remove = nb_of_channels - new_nb_of_channels

        return new_nb_of_channels, nb_channels_to_remove

    @abc.abstractmethod
    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer: layers.Conv2D, layer_weight_mtx) -> List[int]:
        raise NotImplementedError


class KMeansFilterPruning(BasePruning):
    def __init__(self,
                 clustering_factor: float,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Optional[Callable[[models.Model, int, int], None]],
                 nb_finetune_epochs: int,
                 maximum_prune_iterations: int = None,
                 maximum_pruning_percent: float = 0.9,
                 nb_trained_for_epochs: int = 0):
        super().__init__(pruning_factor=clustering_factor,
                         model_compile_fn=model_compile_fn,
                         model_finetune_fn=model_finetune_fn,
                         nb_finetune_epochs=nb_finetune_epochs,
                         nb_trained_for_epochs=nb_trained_for_epochs,
                         maximum_prune_iterations=maximum_prune_iterations,
                         maximum_pruning_percent=maximum_pruning_percent)

    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer: layers.Conv2D, layer_weight_mtx) -> List[int]:
        _, _, _, nb_channels = layer_weight_mtx.shape

        # Initialize KMeans
        nb_of_clusters, _ = self._calculate_number_of_channels_to_keep(pruning_factor, nb_channels)
        kmeans = cluster.KMeans(nb_of_clusters)

        # Fit with the flattened weight matrix
        # (height, width, input_channels, output_channels) -> (output_channels, flattened features)
        layer_weight_mtx_reshaped = layer_weight_mtx.transpose(3, 0, 1, 2).reshape(nb_channels, -1)
        # Apply some fuzz to the weights, to avoid duplicates
        self._apply_fuzz(layer_weight_mtx_reshaped)
        kmeans.fit(layer_weight_mtx_reshaped)

        # If a cluster has only a single member, then that should not be pruned
        # so that point will always be the closest to the cluster center

        closest_point_to_cluster_center_indices = metrics.pairwise_distances_argmin(kmeans.cluster_centers_,
                                                                                    layer_weight_mtx_reshaped)
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(layer_weight_mtx_reshaped)))
        channel_indices_to_keep = set(closest_point_to_cluster_center_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)

        if len(channel_indices_to_keep) > nb_of_clusters:
            print("Number of selected channels for pruning is less than expected")
            diff = len(channel_indices_to_keep) - nb_of_clusters
            print("Randomly adding {0} channels for pruning".format(diff))
            np.random.shuffle(channel_indices_to_keep)
            for i in range(diff):
                channel_indices_to_prune.append(channel_indices_to_keep.pop(i))
        elif len(channel_indices_to_keep) < nb_of_clusters:
            print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
            diff = nb_of_clusters - len(channel_indices_to_keep)
            print("Discarding {0} pruneable channels".format(diff))
            for i in range(diff):
                channel_indices_to_keep.append(channel_indices_to_prune.pop(i))

        if len(channel_indices_to_keep) != nb_of_clusters:
            raise ValueError(
                "Number of clusters {0} is not equal with the selected "
                "pruneable channels {1}".format(nb_of_clusters, len(channel_indices_to_prune)))

        return channel_indices_to_prune


class L1FilterPruning(BasePruning):
    def __init__(self,
                 clustering_factor: float,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Optional[Callable[[models.Model, int, int], None]],
                 nb_finetune_epochs: int,
                 maximum_prune_iterations: int = None,
                 maximum_pruning_percent: float = 0.9,
                 nb_trained_for_epochs: int = 0):
        super().__init__(pruning_factor=clustering_factor,
                         model_compile_fn=model_compile_fn,
                         model_finetune_fn=model_finetune_fn,
                         nb_finetune_epochs=nb_finetune_epochs,
                         nb_trained_for_epochs=nb_trained_for_epochs,
                         maximum_prune_iterations=maximum_prune_iterations,
                         maximum_pruning_percent=maximum_pruning_percent)

    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer: layers.Conv2D, layer_weight_mtx) -> List[int]:
        _, _, _, nb_channels = layer_weight_mtx.shape

        new_nb_of_channels, nb_channels_to_remove = self._calculate_number_of_channels_to_keep(pruning_factor, nb_channels)

        # Fit with the flattened weight matrix
        # (height, width, input_channels, output_channels) -> (output_channels, flattened features)
        layer_weight_mtx_reshaped = layer_weight_mtx.transpose(3, 0, 1, 2).reshape(nb_channels, -1)
        # Apply some fuzz to the weights, to avoid duplicates
        self._apply_fuzz(layer_weight_mtx_reshaped)

        """
        Q : l1_norm 이 작은 filter 의 index 를 찾는 로직을 구현해보세요
        """
        l1_norm = np.linalg.norm(layer_weight_mtx_reshaped, ord=1, axis=1)
        idxes = l1_norm.argsort()
        # Compute filter indices which can be pruned
        channel_indices_to_prune = idxes[:nb_channels_to_remove]

        # return channel_indices_to_prune
