from .trainer import Callback


class LevelController(Callback):
    """Training level controller.
    """
    def __init__(self, num_layers, epochs_per_level):
        """Initializer.
        Args:
            num_layers: int, number of the StyleALAE encoder, generator, layers.
            epochs_per_level: int, number of the epochs per training level.
        """
        super(LevelController, self).__init__()
        self.num_layers = num_layers
        self.epochs_per_level = epochs_per_level

    def interval(self):
        """Set callback interval as epoch level.
        """
        return -1

    def __call__(self, model, _, epochs):
        """Set training level of models based on epochs.
        """
        level = min(self.num_layers - 1, epochs // self.epochs_per_level)
        model.set_level(level)
