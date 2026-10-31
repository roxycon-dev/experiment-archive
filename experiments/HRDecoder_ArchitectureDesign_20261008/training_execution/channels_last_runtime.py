"""Optional measured CUDA tensor layout; architecture, losses and SGD unchanged."""
import torch
import private_modules

ORIGINAL_NORM = private_modules.ChannelLayerNorm2d.forward

def enable_norm_layout(enabled):
    def forward(self, x):
        if x.is_cuda:
            return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return ORIGINAL_NORM(self,x)
    private_modules.ChannelLayerNorm2d.forward = forward if enabled else ORIGINAL_NORM

def install(train):
    enable_norm_layout(True)
    original_student = train.V3Student
    class LayoutStudent(original_student):
        def cuda(self, device=None):
            super().cuda(device)
            return self.to(memory_format=torch.channels_last)
    train.V3Student = LayoutStudent
    original_move = train.move
    def move(batch):
        result = original_move(batch)
        if 'image' in result: result['image']=result['image'].contiguous(memory_format=torch.channels_last)
        return result
    train.move=move
    original_load = torch.optim.SGD.load_state_dict
    def load_state(optimizer, state):
        result=original_load(optimizer,state)
        for param, values in optimizer.state.items():
            if param.ndim==4:
                for name,value in values.items():
                    if torch.is_tensor(value) and value.shape==param.shape:
                        values[name]=value.contiguous(memory_format=torch.channels_last)
        return result
    torch.optim.SGD.load_state_dict=load_state
