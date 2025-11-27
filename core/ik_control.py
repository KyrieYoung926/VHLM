import torch
class GenesisDiffIKController:
    def __init__(self, robot, device="cuda"):
        self.robot = robot
        self.device = device
        # 对应 Isaac Lab config 中的 damping (lambda_val)
        # 如果训练时没改，通常默认是 0.05 ~ 0.1 左右
        self.lambda_val = 0.1 

    def solve(self, link, local_dofs, pos_delta, euler_delta):
        """
        link: Genesis Link 对象
        local_dofs: 关节索引列表
        pos_delta, euler_delta: Policy 输出的 Action
        """
        # --- 1. 维度安全检查 (Fixing the RuntimeError) ---
        # 确保输入是 (Batch, 3) 或者是 (1, 3) 而不是 (3,)
        if pos_delta.dim() == 1:
            pos_delta = pos_delta.unsqueeze(0) # (3,) -> (1, 3)
        if euler_delta.dim() == 1:
            euler_delta = euler_delta.unsqueeze(0) # (3,) -> (1, 3)
            
        # 1. 获取 Jacobian
        # J_full shape 可能是 (N, 6, Total_Dofs) 或者 (6, Total_Dofs)
        J_full = self.robot.get_jacobian(link)
        
        # 确保 Jacobian 也是 3D (Batch, 6, Total_Dofs)
        if J_full.dim() == 2:
            J_full = J_full.unsqueeze(0)

        # 2. 截取对应手臂关节的列
        J_arm = J_full[:, :, local_dofs] # (Batch, 6, N_arm_dofs)

        # 3. 拼接 Action
        # 现在 pos_delta 和 euler_delta 都是 (Batch, 3)，结果是 (Batch, 6)
        delta_pose = torch.cat([pos_delta, euler_delta], dim=-1) 

        # 4. DLS 求解
        # delta_q = J.T @ (J @ J.T + lambda^2 * I)^-1 @ delta_x
        
        # J @ J.T -> (Batch, 6, 6)
        JJT = torch.bmm(J_arm, J_arm.transpose(1, 2))
        
        # Damping Matrix
        damping = (self.lambda_val ** 2) * torch.eye(6, device=self.device)
        damping = damping.unsqueeze(0) # 扩展 Batch 维度以匹配 bmm
        
        # Inverse
        inv_term = torch.linalg.inv(JJT + damping)
        
        # Step 1: J.T @ inv_term -> (Batch, Dofs, 6)
        step1 = torch.bmm(J_arm.transpose(1, 2), inv_term)
        
        # Step 2: step1 @ delta_pose
        # delta_pose 是 (Batch, 6)，unsqueeze 后变成 (Batch, 6, 1)
        # bmm 输出 (Batch, Dofs, 1)
        delta_q = torch.bmm(step1, delta_pose.unsqueeze(-1)).squeeze(-1)
        
        # 如果原始输入没有 batch (是 1D 的)，为了方便外部加法，我们可以把结果压回 1D
        # 但考虑到你的外部代码可能有 curr_q + delta_q，保持 (1, Dofs) 通常也没问题，会自动广播
        # 如果需要严格匹配 (Dofs,)，可以解开：
        if delta_q.shape[0] == 1:
            return delta_q.squeeze(0)
            
        return delta_q