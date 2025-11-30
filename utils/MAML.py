import torch
import torch.nn.functional as F

class MAML:
    def __init__(self, model, lr_inner, lr_outer, device):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr_outer
            )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=10)
        self.device = device

    def inner_update(self, features, labels):
        # Perform inner update to get updated parameters
        outputs = self.model(features)
        loss = F.cross_entropy(outputs, labels)

        grads = torch.autograd.grad(
            loss, 
            self.model.parameters(), 
            create_graph=True
            )
        
        updated_params = {
            name: param - self.lr_inner * grad for ((name, param), grad) in zip(self.model.named_parameters(), grads)
            }

        return updated_params

    def forward(self, features, params=None):
        # Perform forward pass with either original or updated parameters
        if params is None:
            return self.model(features)
        else:
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = params[name].to(self.device)
            outputs = self.model(features)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = original_params[name].to(self.device)
            return outputs

    def outer_update(self, updated_loss):
        # Perform outer update with original optimizer
        self.optimizer.zero_grad()

        updated_loss.backward()

        self.optimizer.step()
        self.scheduler.step()