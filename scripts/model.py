"""
@file model.py
@brief Contains all models

This module all Neural Network based models used throughout the thesis

@details
Classes:
- AutoencoderModel - Model used for Autoencoder (AE)
- TransformerModel - Model used for Attention Network (Att)
- MainNet - Model used for FFN, Main-SMF and Mod-SMF
- DoubleNet - Model used for JSMF
- TripleNet -  Model used for JSRF
- Hypernetwork - Model used for Hypernetwork (Hyp) - actual Hypernetwork
- MainNetHypernetwork - Model used for Hypernetwork (Hyp) - MainNet

@package model
"""

import torch.nn as nn
import torch

class AutoencoderModel(nn.Module):
    """
    @brief Class for the Autoencoder (AE)
    """
    def __init__(self, input_dim_main: int, input_dim_context: int, hidden_layers_main: int,
                 hidden_layers_context: int, hidden_layers_final: int, hidden_dim_main: int,
                 hidden_dim_context: int, hidden_dim_final: int, factor_amount: int, dropout: float) -> None:
        """
        @brief Initializes the Autoencoder network
        @param input_dim_main: Dimension of main input
        @param input_dim_context: Dimension of modulating input
        @param hidden_layers_main: Amount of hidden layers in the main network
        @param hidden_layers_context: Amount of hidden layers in the modulating network
        @param hidden_layers_final: Amount of hidden layers in the fusion network
        @param hidden_dim_main: Dimension of hidden layers in the main network
        @param hidden_dim_context: Dimension of hidden layers in the modulating network
        @param hidden_dim_final: Dimension of hidden layers in the fusion network
        @param factor_amount: Amount of factors to use in the autoencoder network
        @param dropout: Dropout rate
        """
        super(AutoencoderModel, self).__init__()

        #linear layer to encode factors
        self.factor_layer = nn.Linear(input_dim_main+input_dim_context, factor_amount)

        #layers main net
        layers = []
        prev_dim = input_dim_main
        for i in range(hidden_layers_main):
            layers.append(nn.Linear(prev_dim, hidden_dim_main))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_main
        #final output layer
        layers.append(nn.Linear(prev_dim, hidden_dim_main))
        self.net_main = nn.Sequential(*layers)

        #layers modulating net
        layers2 = []
        prev_dim = input_dim_context
        for i in range(hidden_layers_context):
            layers2.append(nn.Linear(prev_dim, hidden_dim_context))
            layers2.append(nn.LeakyReLU())
            layers2.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_context
        #final output layer
        layers2.append(nn.Linear(prev_dim, hidden_dim_context))
        self.net_context = nn.Sequential(*layers2)

        #layers fusion net
        layers3 = []
        prev_dim = hidden_dim_main + hidden_dim_context
        for i in range(hidden_layers_final):
            layers3.append(nn.Linear(prev_dim, hidden_dim_final))
            layers3.append(nn.LeakyReLU())
            layers3.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_final
        #final output layer
        layers3.append(nn.Linear(prev_dim, factor_amount))
        self.net_final = nn.Sequential(*layers3)

    def forward(self, main_feats: torch.Tensor, context_feats: torch.Tensor,
                returns: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the Autoencoder network
            (only used during training as returns are necessary for factors)
        @param main_feats: (batch_size, input_dim_main)
        @param context_feats: (batch_size, input_dim_modulating)
        @param returns: (batch_size, 1)
        @return: (batch_size, 1)
        """
        #left side
        net_main = self.net_main(main_feats)
        net_context = self.net_context(context_feats)
        final = torch.cat((net_main, net_context), dim=1)
        loadings = self.net_final(final)

        #right side
        feats = torch.cat((main_feats, context_feats), dim=1)
        portf = (torch.linalg.pinv(feats.T @ feats) @ feats.T @ returns).T
        factors = self.factor_layer(portf).T

        output = loadings @ factors
        return output

    def initFactorList(self) -> None:
        """
        @brief Initializes the saved factors for inference
        """
        self.savedFactorsList = []

    def saveFactors(self, main_feats: torch.Tensor, context_feats: torch.Tensor,
                    returns: torch.Tensor) -> None:
        """
        @brief Saves the calculated factors for inference
        @param main_feats: (batch_size, input_dim_main)
        @param context_feats: (batch_size, input_dim_modulating)
        @param returns: (batch_size, 1)
        """
        feats = torch.cat((main_feats, context_feats), dim=1)
        portf = (torch.linalg.pinv(feats.T @ feats) @ feats.T @ returns).T
        factors = self.factor_layer(portf).T
        self.savedFactorsList.append(factors)

    def averageFactors(self) -> None:
        """
        @brief Averages the saved factors
        """
        self.savedFactors = torch.stack(self.savedFactorsList, dim=0).mean(dim=0)

    def validate(self, main_feats: torch.Tensor, context_feats: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass for inference using saved factors
        @param main_feats: (batch_size, input_dim_main)
        @param context_feats: (batch_size, input_dim_modulating)
        @return: (batch_size, 1)
        """
        #left side
        net_main = self.net_main(main_feats)
        net_context = self.net_context(context_feats)
        final = torch.cat((net_main, net_context), dim=1)
        loadings = self.net_final(final)

        output = loadings @ self.savedFactors
        return output

class TransformerModel(nn.Module):
    """
    @brief Class for the Attention Network (Att)
    """
    def __init__(self, d_model: int, nhead: int, num_layers_decoder: int, dim_feedforward_transformer: int,
                 dim_feedforward: int, layers_feedforward: int, input_dim: int, dropout: float) -> None:
        """
        @brief Initializes the attention network
        @param d_model: Model dimension
        @param nhead: Number of attention heads
        @param num_layers_decoder: Number of decoder layers
        @param dim_feedforward_transformer: Dimension of hidden feedforward layers in transformer
        @param dim_feedforward: Dimension of hidden layers in feedforward net after transformer
        @param layers_feedforward: Number of hidden layers in feedforward net after transformer
        @param input_dim: Dimensionality of input
        @param dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()

        #Linear embedding of main and modulating features
        self.main_proj = nn.Linear(1, d_model)     #input: (B, 286, 1) → (B, 286, d_model)
        self.context_proj = nn.Linear(1, d_model)  #input: (B, 5, 1) → (B, 5, d_model)

        #initialize transformer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=1,
                                          num_decoder_layers=num_layers_decoder,
                                          dim_feedforward=dim_feedforward_transformer,
                                          dropout=dropout, activation=nn.LeakyReLU(), batch_first=True)

        #layers for feedforward net after transformer
        layers = []
        layers.append(nn.Linear(input_dim*d_model, dim_feedforward))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout))
        for i in range(layers_feedforward-1):
            layers.append(nn.Linear(dim_feedforward,dim_feedforward))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        #final output layer
        layers.append(nn.Linear(dim_feedforward, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, main_feats: torch.Tensor, context_feats: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the attention network
        @param main_feats: (batch_size, input_dim_main, 1)
        @param context_feats: (batch_size, input_dim_modulating, 1)
        @return: (batch_size, 1)
        """

        #linear embedding
        main_proj = self.main_proj(main_feats) #(B, 286, d_model)
        context_proj = self.context_proj(context_feats) #(B, 5, d_model)

        #transformer forward pass (context_proj to encoder and main_proj to decoder)
        output_transformer = self.transformer(context_proj, main_proj,
                                              src_is_causal=False, tgt_is_causal=False)
        output = output_transformer.view(output_transformer.size(0), -1) #stacking output

        #final feedforward neural network
        output = self.net(output)
        return output

class MainNet(nn.Module):
    """
    @brief Class for simple feedforward network (used for FFN, Main-SMF and Mod-SMF
        (for the latter two we always use two instances of this class learned separately))
    """
    def __init__(self, input_dim: int, hidden_layers: int, hidden_dim: int, dropout: float) -> None:
        """
        @brief Initializes the network
        @param input_dim: Dimensionality of input
        @param hidden_layers: Amount of hidden layers
        @param hidden_dim: Dimensionality of hidden layers
        @param dropout: Dropout rate
        """
        super().__init__()

        #layers for network
        layers = []
        prev_dim = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        #final output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the network
        @param x: Input tensor
        @return: Output tensor
        """
        return self.net(x)


class DoubleNet(nn.Module):
    """
    @brief Class for fusion network using two networks (used for JSMF)
    """
    def __init__(self, input_dim_main: int, input_dim_context: int, hidden_layers_main: int,
                 hidden_layers_context: int, hidden_dim_main: int, hidden_dim_context: int,
                 dropout: float) -> None:
        """
        @brief Initializes the network
        @param input_dim_main: Dimensionality of main input
        @param input_dim_context: Dimensionality of modulating input
        @param hidden_layers_main: Number of hidden layers in main net
        @param hidden_layers_context: Number of hidden layers in modulating net
        @param hidden_dim_main: Dimensionality of hidden layers in main net
        @param hidden_dim_context: Dimensionality of hidden layers in modulating net
        @param dropout: Dropout rate
        """
        super().__init__()

        #layers for main net
        layers = []
        prev_dim = input_dim_main
        for i in range(hidden_layers_main):
            layers.append(nn.Linear(prev_dim, hidden_dim_main))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_main
        #final output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.net_main = nn.Sequential(*layers)

        #layers for modulating net
        layers2 = []
        prev_dim = input_dim_context
        for i in range(hidden_layers_context):
            layers2.append(nn.Linear(prev_dim, hidden_dim_context))
            layers2.append(nn.LeakyReLU())
            layers2.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_context
        #final output layer
        layers2.append(nn.Linear(prev_dim, 1))
        self.net_context = nn.Sequential(*layers2)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the network
        @param x: Input to main net (batch_size, input_dim_main)
        @param context: (batch_size, input_dim_modulating)
        @return: (batch_size, 1)
        """
        net_main = self.net_main(x)
        net_context = self.net_context(context)
        return net_main * net_context #Simple multiplication of output, but learning together

class TripleNet(nn.Module):
    """
    @brief Class for fusion network using three networks (used in JSRF)
    """
    def __init__(self, input_dim_main: int, input_dim_context: int, hidden_layers_main: int,
                 hidden_layers_context: int, hidden_layers_final: int, hidden_dim_main: int,
                 hidden_dim_context: int, hidden_dim_final: int, dropout: float) -> None:
        """
        @brief Initializes the network
        @param input_dim_main: Dimensionality of main input
        @param input_dim_context: Dimensionality of modulating input
        @param hidden_layers_main: Amount of hidden layers in main net
        @param hidden_layers_context: Amount of hidden layers in modulating net
        @param hidden_layers_final: Amount of hidden layers in fusion net
        @param hidden_dim_main: Dimensionality of hidden layers in main net
        @param hidden_dim_context: Dimensionality of hidden layers in modulating net
        @param hidden_dim_final: Dimensionality of hidden layers in fusion net
        @param dropout: Dropout rate
        """
        super().__init__()

        #layers main net
        layers = []
        prev_dim = input_dim_main
        for i in range(hidden_layers_main):
            layers.append(nn.Linear(prev_dim, hidden_dim_main))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_main
        #final output layer
        layers.append(nn.Linear(prev_dim, hidden_dim_main))
        self.net_main = nn.Sequential(*layers)

        #layers modulating net
        layers2 = []
        prev_dim = input_dim_context
        for i in range(hidden_layers_context):
            layers2.append(nn.Linear(prev_dim, hidden_dim_context))
            layers2.append(nn.LeakyReLU())
            layers2.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_context
        #final output layer
        layers2.append(nn.Linear(prev_dim, hidden_dim_context))
        self.net_context = nn.Sequential(*layers2)

        #layers fusion net
        layers3 = []
        prev_dim = hidden_dim_main + hidden_dim_context
        for i in range(hidden_layers_final):
            layers3.append(nn.Linear(prev_dim, hidden_dim_final))
            layers3.append(nn.LeakyReLU())
            layers3.append(nn.Dropout(dropout))
            prev_dim = hidden_dim_final
        #final output layer
        layers3.append(nn.Linear(prev_dim, 1))
        self.net_final = nn.Sequential(*layers3)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the network
        @param x: Input to main net (batch_size, input_dim_main)
        @param context: (batch_size, input_dim_modulating)
        @return: (batch_size, 1)
        """
        net_main = self.net_main(x)
        net_context = self.net_context(context)
        final = torch.cat((net_main, net_context), dim=1)
        #outputs of individual networks are concatentated and put into final fusion network
        return self.net_final(final)

class Hypernetwork(nn.Module):
    """
    @brief Class for Hypernetwork (Hyp) (this implements the actual hypernetwork outputting multiple weights)
    """
    def __init__(self, input_dim: int, hidden_layers: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        """
        @brief Initializes the hypernetwork
        @param input_dim: Dimensionality of modulating input
        @param hidden_layers: Amount of hidden layers in hypernetwork
        @param hidden_dim: Dimensionality of hidden layers in hypernetwork
        @param output_dim: Dimensionality of output (weights for main network)
        @param dropout: Dropout rate
        """
        super(Hypernetwork, self).__init__()
        super().__init__()

        #layers of network
        layers = []
        prev_dim = input_dim
        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        #final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the network
        @param x: Input to hypernetwork (batch_size, input_dim_modulating)
        @return: (batch_size, output_dim)
        """
        return self.net(x)

class MainNetHypernetwork(nn.Module):
    """
    @brief Class for Hypernetwork (Hyp) (this implements the main network outputting the final prediction)
    """
    def __init__(self, hidden_layers: int, hidden_dim: int, hypernet: Hypernetwork,
                 dropout: float) -> None:
        """
        @brief Initializes the main network
        @param hidden_layers: Number of hidden layers in main net
        @param hidden_dim: Dimensionality of hidden layers in main net
        @param hypernet: Hypernetwork of class Hypernetwork
        @param dropout: Dropout rate
        """
        super(MainNetHypernetwork, self).__init__()

        #initialize hypernet
        self.hypernet = hypernet
        self.hidden_dim = hidden_dim

        #layers for main net
        layers = []
        for i in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
        #final output layer
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, hypervariables: torch.Tensor) -> torch.Tensor:
        """
        @brief Forward pass of the network
        @param x: Input to main network (batch_size, input_dim_main)
        @param hypervariables: Input to hypernetwork (batch_size, input_dim_modulating)
        @return: (batch_size, 1)
        """
        #get weights via hypernetwork
        hyper_weights = self.hypernet(hypervariables)
        # View hyperweights as [batch_size, input_dim, hidden_dim]
        hyper_weights = hyper_weights.view(x.shape[0], x.shape[1], self.hidden_dim)
        #apply the hyperweights to the input
        #since hyper_weights is [batch_size, input_dim, hidden_dim],
        #we can perform batch-wise matrix multiplication
        x = torch.bmm(x.unsqueeze(1), hyper_weights).squeeze(1)  #[batch_size, hidden_dim]
        x = nn.LeakyReLU()(x)
        return self.net(x)
