import torch

def to_torch(x, dtype=torch.float32, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)

def quat_apply_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q (torch tensors). q: (..., 4) wxyz."""
    if q.dim() == 1: q = q.unsqueeze(0)
    if v.dim() == 1: v = v.unsqueeze(0)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # conjugate
    wc, xc, yc, zc = w, -x, -y, -z
    # rotation matrix from conjugate
    m00 = wc * wc + xc * xc - yc * yc - zc * zc
    m01 = 2 * (xc * yc - wc * zc)
    m02 = 2 * (xc * zc + wc * yc)
    m10 = 2 * (xc * yc + wc * zc)
    m11 = wc * wc - xc * xc + yc * yc - zc * zc
    m12 = 2 * (yc * zc - wc * xc)
    m20 = 2 * (xc * zc - wc * yc)
    m21 = 2 * (yc * zc + wc * xc)
    m22 = wc * wc - xc * xc - yc * yc + zc * zc
    
    rx = v[..., 0] * m00 + v[..., 1] * m01 + v[..., 2] * m02
    ry = v[..., 0] * m10 + v[..., 1] * m11 + v[..., 2] * m12
    rz = v[..., 0] * m20 + v[..., 1] * m21 + v[..., 2] * m22
    return torch.stack([rx, ry, rz], dim=-1).squeeze(0)

def quat_apply(q, v):
    """Rotate vector v by quaternion q (torch tensors). q in wxyz."""
    if q.dim() == 1: q = q.unsqueeze(0)
    if v.dim() == 1: v = v.unsqueeze(0)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m00 = w * w + x * x - y * y - z * z
    m01 = 2 * (x * y - w * z)
    m02 = 2 * (x * z + w * y)
    m10 = 2 * (x * y + w * z)
    m11 = w * w - x * x + y * y - z * z
    m12 = 2 * (y * z - w * x)
    m20 = 2 * (x * z - w * y)
    m21 = 2 * (y * z + w * x)
    m22 = w * w - x * x - y * y + z * z
    
    rx = v[..., 0] * m00 + v[..., 1] * m01 + v[..., 2] * m02
    ry = v[..., 0] * m10 + v[..., 1] * m11 + v[..., 2] * m12
    rz = v[..., 0] * m20 + v[..., 1] * m21 + v[..., 2] * m22
    return torch.stack([rx, ry, rz], dim=-1).squeeze(0)

def quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
    if quat.dim() == 1: quat = quat.unsqueeze(0)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    v1 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    v2 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    v3 = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.cat([v1, v3], dim=-1)

def euler_to_quat_wxyz(euler, device="cpu"):
    """Convert euler (roll, pitch, yaw) to quaternion (w,x,y,z)."""
    if not torch.is_tensor(euler):
        euler = to_torch(euler, device=device)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cy = torch.cos(yaw * 0.5); sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5); sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5); sr = torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z]).to(dtype=torch.float32, device=device)

def quat_to_euler_wxyz(quat_wxyz: torch.Tensor, device="cpu") -> torch.Tensor:
    """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)."""
    q = quat_wxyz if isinstance(quat_wxyz, torch.Tensor) else to_torch(quat_wxyz, device=device)
    w, x, y, z = q[0], q[1], q[2], q[3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w * y - z * x)
    sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp_clamped)
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw]).to(dtype=torch.float32, device=device)

def quat_mul(q1, q2):
    # Quaternion multiplication
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z])
        
def quat_inv(q):
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device)