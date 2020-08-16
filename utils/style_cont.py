from .trainer import Callback


class LevelController(Callback):
    """Training level controller.
    """
    def __init__(self, num_layers, epochs_per_level):
        """Initializer.
        Args:
            num_layers: int, number of the StyleALAE encoder, generator, layers.
            epochs_per_level: Union[int, Dict[int, int]],
                number of the epochs per training level.
                If dictionary is given, it must contain key-0 as default level.
        """
        super(LevelController, self).__init__()
        self.num_layers = num_layers
        self.epochs_per_level = epochs_per_level
        self.schedule = self._uniform_schedule \
            if isinstance(self.epochs_per_level, int) \
            else self._dict_base_schedule

    def interval(self):
        """Set callback interval as epoch level.
        """
        return -1

    def __call__(self, model, _, epochs):
        """Set training level of models based on epochs.
        """
        level = self.schedule(epochs)
        model.set_level(level)

    def _uniform_schedule(self, epochs):
        """Set training level uniformly based on predefined constant.
        """
        return min(self.num_layers - 1, epochs // self.epochs_per_level)
    
    def _dict_base_schedule(self, epochs):
        """Set training level based on dictionary.
        """
        if epochs in self.epochs_per_level:
            self.last_level = self.epochs_per_level[epochs]
        return self.last_level
