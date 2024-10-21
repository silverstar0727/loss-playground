import torch


class CustomLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss_1 = torch.nn.CrossEntropyLoss()
        self.loss_2 = torch.nn.BCELoss()

        self.label2label = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
            8: 0,
            9: 0,
        }


    def forward(self, pred, target):
        loss = 0
        loss += self.loss_1(pred, target)

        pred = torch.argmax(pred, dim=1)

        b_predict = torch.tensor([self.label2label[p] for p in pred.detach().cpu().numpy()]).float()
        b_target = torch.tensor([self.label2label[t] for t in target.detach().cpu().numpy()]).float()
        
        loss += self.loss_2(b_predict, b_target)

        return loss