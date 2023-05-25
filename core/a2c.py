import torch
from torch.nn.functional import binary_cross_entropy_with_logits as f_loss_label

def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):

    """update critic"""
    values_pred = value_net(states)
    value_loss = (values_pred - returns).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()

def assembly_a2c_step(ac_net, optimizer, values, states, actions, returns, advantages, l2_reg):

    """update critic"""
    values_pred = values
    value_loss = (values_pred - returns).pow(2).mean()

    # # weight decay
    # for param in value_net.parameters():
    #     value_loss += param.pow(2).sum() * l2_reg

    """update policy"""
    log_probs = ac_net.get_log_prob(states, actions)
    policy_loss = -(log_probs * advantages).mean()

    total_loss = value_loss + policy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(ac_net.parameters(), 40)
    optimizer.step()

def a2c_label_step(policy_net, value_net, 
                   optimizer_policy, optimizer_value, 
                   label_img, state_img, 
                   actions, returns, advantages, l2_reg):
    
    """update critic"""
    values_pred, v_label_pred = value_net(state_img)
    value_loss = (values_pred - returns).pow(2).mean()
    vlabel_loss = f_loss_label(v_label_pred.flatten(), label_img)
    total_loss_v = value_loss + vlabel_loss

    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    total_loss_v.backward()
    optimizer_value.step()

    """update policy"""
    log_probs, p_label_pred = policy_net.get_log_prob(state_img, actions)
    policy_loss = -(log_probs * advantages).mean()
    plabel_loss = f_loss_label(p_label_pred.flatten(), label_img)
    total_loss_p = policy_loss + plabel_loss

    optimizer_policy.zero_grad()
    total_loss_p.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()
