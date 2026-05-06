"""
@file CombinedFusionModel.py
@brief Module for sequential fusion model (Main-SMF and Mod-SMF)

Module for sequential fusion model (Main-SMF and Mod-SMF) that combines the initially trained models
into a single ensemble that can be used to train the secondary networks.

@details
Classes:
- CombinedFusionModel - Class for sequential fusion model (Main-SMF and Mod-SMF)

@package CombinedFusionModel
"""

import torch
import numpy as np

class CombinedFusionModel:
    """
    @brief Class for sequential fusion model (Main-SMF and Mod-SMF).
    @details Class for sequential fusion model (Main-SMF and Mod-SMF) that combines the initially trained
        models into a single ensemble that can be used to train the secondary networks.
    """
    def __init__(self, model_name: str, amount: int, device: str) -> None:
        """
        @brief Initializes the model
        @param model_name: Save space of model files
            (in form: '{folder}/{model_name}_model_main_{rolling_window}_')
        @param amount: Amount of individual models per rolling window
            (one instance of this class is created per rolling window)
        @param device: Device on which the model is run
        """
        self.model_name = model_name
        self.amount = amount
        self.models = []
        self.device = device
        for i in range(self.amount): #Iterate over individual models and load them
            main_model = torch.load(self.model_name + str(i) + '.pt', map_location=self.device,
                                    weights_only=False)
            main_model.eval()
            self.models.append(main_model)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Perform prediction on the main model
        @param x: Main inputs
        @return Prediction
        """
        x_first = torch.tensor(np.array(x.cpu()), dtype=torch.float32).to(self.device)
        Y = self.models[0](x_first)
        for i in range(1, self.amount): #iterate over individual models
            y = self.models[i](x_first)
            Y += y
        Y = Y / self.amount #average over individual models
        return Y
