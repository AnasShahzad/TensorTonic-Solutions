import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m_t = (beta1 * m) + ((1-beta1) * grad)
    v_t = (beta2 * v) + ((1-beta2) * (grad * grad))
    m_hat_t = (m_t) / ( 1 - np.pow(beta1, t))
    v_hat_t = (v_t) / ( 1 - np.pow(beta2, t))
    param_new = param - ((lr * m_hat_t) / (np.pow(v_hat_t,0.5) + eps))
    return (param_new, m_t, v_t)