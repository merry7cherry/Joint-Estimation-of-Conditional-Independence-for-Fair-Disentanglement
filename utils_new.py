import torch
import torch.nn.functional as F


def permute_zs(zs):
    B, _ = zs[0].size()
    perm_z = []

    for z_i in zs:
        perm = torch.randperm(B).cuda()
        perm_z.append(z_i[perm])
    return torch.cat(perm_z, 1)


def Joint_CMI_loss(joint_cls, cls_y, cls_a, z_y, z_s, z_r, beta=0.5):
    # This function computes I_phi(Y;S|Z_R) directly:
    # I(Y;S|Z_R) = H(Y|Z_R) + H(S|Z_R) - H(Y,S|Z_R)
    # We use the batch to marginalize over z_Y and z_S, similar to FADES.

    # ======== Compute H(Y|Z_R) ========
    # Repeat z_y and z_r along batch to form N² combinations
    z_y_rep = z_y.unsqueeze(1).expand(-1, z_y.size(0), -1)  # [N, N, dim_z_y]
    z_r_rep_y = z_r.unsqueeze(0).expand(z_y.size(0), -1, -1) # [N, N, dim_z_r]
    z_yr = torch.cat([z_y_rep, z_r_rep_y], dim=-1).reshape(z_y.size(0)**2, -1) # [N², dim_z_y+dim_z_r]

    p_y = torch.sigmoid(cls_y(z_yr)).view(z_y.size(0), z_y.size(0), -1) # [N, N, 1]
    p_y_zr = p_y.mean(0) # average over z_y => p(y|z_r)
    H_y_z = -(p_y_zr * torch.log(p_y_zr + 1e-7) + (1 - p_y_zr)*torch.log(1 - p_y_zr + 1e-7)).mean()

    # ======== Compute H(S|Z_R) ========
    z_s_rep = z_s.unsqueeze(1).expand(-1, z_s.size(0), -1)  # [N, N, dim_z_s]
    z_r_rep_s = z_r.unsqueeze(0).expand(z_s.size(0), -1, -1) # [N, N, dim_z_r]
    z_sr = torch.cat([z_s_rep, z_r_rep_s], dim=-1).reshape(z_s.size(0)**2, -1) # [N², dim_z_s+dim_z_r]

    p_s = torch.sigmoid(cls_a(z_sr)).view(z_s.size(0), z_s.size(0), -1) # [N, N, 1]
    p_s_zr = p_s.mean(0) # p(s|z_r)
    H_s_z = -(p_s_zr * torch.log(p_s_zr + 1e-7) + (1 - p_s_zr)*torch.log(1 - p_s_zr + 1e-7)).mean()

    # ======== Compute H(Y,S|Z_R) ========
    # Create N² combinations of (z_y,z_s) and pair with z_r
    z_y_expand = z_y.unsqueeze(1).unsqueeze(2).expand(-1, z_s.size(0), z_r.size(0), -1)  # [N, N, N, dim_z_y]
    z_s_expand = z_s.unsqueeze(0).unsqueeze(2).expand(z_y.size(0), -1, z_r.size(0), -1)  # [N, N, N, dim_z_s]
    z_r_expand = z_r.unsqueeze(0).unsqueeze(0).expand(z_y.size(0), z_s.size(0), -1, -1)  # [N, N, N, dim_z_r]

    z_ysr = torch.cat([z_y_expand, z_s_expand, z_r_expand], dim=-1)  # [N, N, N, dim_z_y+dim_z_s+dim_z_r]
    z_ysr = z_ysr.reshape(-1, z_ysr.size(-1))  # [N³, dim_z_y+dim_z_s+dim_z_r]

    logits_ysr = joint_cls(z_ysr) # [N³, 4]
    p_ysr = F.softmax(logits_ysr, dim=-1).view(z_y.size(0), z_s.size(0), z_r.size(0), 4) # [N, N, N, 4]

    p_ys_zr = p_ysr.mean(dim=0).mean(dim=0)  # [N, 4] # average over z_y,z_s => p(y,s|z_r) a vector of length 4
    H_ys_z = -(p_ys_zr * torch.log(p_ys_zr + 1e-7)).sum(dim=-1).mean()

    # I(Y;S|Z_R) = H(Y|Z_R) + H(S|Z_R) - H(Y,S|Z_R)
    I_ys_z = (1-beta)*(H_y_z + H_s_z) - H_ys_z
    return I_ys_z

