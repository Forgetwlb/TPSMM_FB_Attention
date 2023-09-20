from torch import nn
import torch
from torchvision import models
from ptflops import get_model_complexity_info

class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps

        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps*5*2)

        
    def forward(self, image):

        fg_kp = self.fg_encoder(image)
        bs, _, = fg_kp.shape
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        out = {'fg_kp': fg_kp.view(bs, self.num_tps*5, -1)}

        return out

if __name__ == '__main__':
    import torch

    model = KPDetector(10)
    model = model.cuda()
    input = torch.rand(1, 3, 256, 256).cuda()
    output = model(input)
    # print(model)

    ### thop cal ###
    # input_shape = (1, 3, 384, 384) # 输入的形状
    # input_data = torch.randn(*input_shape)
    # macs, params = profile(model, inputs=(input_data,))
    # print(f"FLOPS: {macs / 1e9:.2f}G")
    # print(f"params: {params / 1e6:.2f}M")

    ### ptflops cal ###
    # input1 = (3, 256, 256)
    # input2 = (50, 2)
    # flops_count, params_count = get_model_complexity_info(model, (input1, input2), as_strings=True, print_per_layer_stat=False)
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    # print('flops: ', flops_count)
    # print('params: ', params_count)