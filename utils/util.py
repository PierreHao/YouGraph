import torch
import copy

def clones(module, k):
    return torch.nn.ModuleList(copy.deepcopy(module) for _ in range(k))

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch**3 * init_lr / num_batch_warm_up**3

def flag(model_forward, perturb_shape, y, optimizer, device, criterion, step_size=8e-3, m=3) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()
                    
    perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= m
    for _ in range(m-1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0
        
        out = forward(perturb)
        loss = criterion(out, y)
        loss /= m                                      
    loss.backward()
    optimizer.step()
    return loss, out

def train_with_flag(model, device, loader, optimizer, multicls_criterion):
    total_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            forward = lambda perturb : model(batch, perturb).to(torch.float32)
            model_forward = (model, forward)
            y = batch.y.view(-1,)
            perturb_shape = (batch.x.shape[0], model.config.hidden)
            loss, _ = flag(model_forward, perturb_shape, y, optimizer, device, multicls_criterion)
            total_loss += loss.item()
            
    #print(total_loss/len(loader))
    return total_loss/len(loader)
