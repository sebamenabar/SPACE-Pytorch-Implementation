import torch

def sample_gaussean(mean, loc):
    dist = torch.distributions.normal.Normal(mean, loc)
    return dist.rsample()

def sample_bernoulli(p_logits, hard=True, clamp=False, temperature=1):
    dist = torch.distributions.RelaxedBernoulli(temperature=1, logits=p_logits)
    obj_prob = dist.rsample(p_logits.size()).to(device=p_logits.device)
    if hard: # Use ST-trick
        obj_prob_hard = (obj_prob >= 0.5).to(dtype=torch.float)
        return (obj_prob_hard - obj_prob).detach() + obj_prob, obj_prob
    else:
        return obj_prob, obj_prob

def manual_sample_obj_pres(p_logits, hard=True, clamp=False, eps=1e-20):
    if clamp:
        # In original SPAIR implementation logits al clamped
        p_logits = torch.clamp(p_logits, -10., 10.)
    
    # Gumbel-softmax trick
    u = torch.rand(p_logits.size()) # [0,1) uniform
    # Sample Gumbel
    noise = torch.log(u + eps) - torch.log(1.0 - u + eps)
    # Sample bernoulli
    obj_pre_sigmoid = (p_logits + noise)
    obj_prob = torch.sigmoid(obj_pre_sigmoid)

    # Use 
    if hard:
        obj_prob_hard = (obj_prob >= 0.5).to(p_logits.dtype)
        return (obj_prob_hard - obj_prob).detach() + obj_prob, obj_prob
    else:
        return obj_prob, obj_prob