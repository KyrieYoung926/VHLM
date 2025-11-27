import os
import time
import collections
import numpy as np
import torch
import genesis as gs

from utils.math_ops import to_torch, quat_apply, quat_apply_inverse, quat_to_tan_norm, euler_to_quat_wxyz, quat_to_euler_wxyz, quat_mul, quat_inv
import cfg.robot_config as cfg
from core.ik_control import GenesisDiffIKController

class HomieGenesisControllerNew:
    def __init__(
        self,
        homie_policy_path: str,
        robot_xml_path: str,
        device: str = "cuda:0",
        render: bool = True,
        dt: float = 0.002,
        control_decimation: int = 10,
        high_level_policy_path=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dt = dt
        self.control_decimation = control_decimation
        self.render = render

        # History and Dimensions from config
        self.obs_history_len = cfg.OBS_HISTORY_LEN
        self.single_obs_dim = cfg.SINGLE_OBS_DIM
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        # High-level observation params
        self.high_obs_history_len = 6
        self.high_obs_dim = 108  # 单帧总维度
        
        # 1. 定义 buffer，保持 (History, Total_Dim) 的形状，方便更新
        self.high_obs_buffer = torch.zeros((self.high_obs_history_len, self.high_obs_dim), device=self.device)
        
        # 2. 定义每个部分的维度 (名称, 维度)，顺序必须与 _compute_single_high_level_obs 的 torch.cat 顺序严格一致！
        self.high_obs_segments = [
            ("base_ang_vel", 3),             # 对应 base_ang
            ("projected_gravity", 3),        # 对应 proj_g
            ("joint_pos", 29),               # 对应 joint_pos
            ("joint_vel", 29),               # 对应 joint_vel
            ("actions", 19),                 # 对应 last_actions
            ("left_hand_pos", 3),            # 对应 left_rel
            ("right_hand_pos", 3),           # 对应 right_rel
            ("box_root_info", 9),            # 对应 object_root_info (3 pos + 6 rot)
            ("box_target_pose_boundary", 9), # 对应 object_target_pose (3 pos + 6 rot)
            ("box_size", 1),                 # 对应 box_size
        ]
        # Scales from config
        self.ang_vel_scale = cfg.ANG_VEL_SCALE
        self.dof_pos_scale = cfg.DOF_POS_SCALE
        self.dof_vel_scale = cfg.DOF_VEL_SCALE
        self.cmd_scale = to_torch(cfg.CMD_SCALE, device=self.device)
        self.height_cmd_scale = cfg.HEIGHT_CMD_SCALE
        self.policy_action_scale_legs = cfg.POLICY_ACTION_SCALE_LEGS
        self.hierarchical_action_scale = cfg.HIERARCHICAL_ACTION_SCALE
        self.waist_action_scale = cfg.WAIST_ACTION_SCALE
        self.arm_ik_action_scale = cfg.ARM_IK_ACTION_SCALE

        # --- Initialize Config Data ---
        self._init_joint_configs()

        # --- Load Policies ---
        self._load_policies(homie_policy_path, high_level_policy_path)

        # --- Setup Genesis Scene ---
        self._setup_scene(robot_xml_path)
        # --- Build Index Maps ---
        self._setup_joint_indices()

        # Initialize State
        single = np.zeros(self.single_obs_dim, dtype=np.float32)
        for _ in range(self.obs_history_len):
            self.obs_history.append(single.copy())
        self.last_leg_action = np.zeros(12, dtype=np.float32)
        
        self.standing_point_debug_node = None
        self.ik_controller = GenesisDiffIKController(self.robot, device=self.device)


    def _init_joint_configs(self):
        # High-level defaults
        self.default_qpos_isaac_29 = torch.zeros(29, device=self.device)
        for i, v in cfg.ISAAC_DEFAULTS_DICT.items():
            self.default_qpos_isaac_29[i] = v
            
        # Low-level defaults
        self.low_level_default_qpos_27 = torch.zeros(27, device=self.device)
        for i, v in cfg.LOW_LEVEL_DEFAULTS_DICT.items():
            self.low_level_default_qpos_27[i] = v
            
        # Genesis defaults
        self.default_qpos_29 = torch.zeros(29, device=self.device)
        idx29 = {n: i for i, n in enumerate(cfg.GENESIS_JOINT_NAMES_29)}
        for n, v in cfg.GENESIS_DEFAULTS_DICT.items():
            if n in idx29:
                self.default_qpos_29[idx29[n]] = v

    def _load_policies(self, homie_path, high_level_path):
        print(f"Loading Homie policy: {homie_path}")
        self.policy = torch.jit.load(homie_path, map_location=self.device)
        self.policy.eval()

        self.high_level_policy = None
        self._high_level_enabled = False
        self._last_high_actions = torch.zeros(19, device=self.device)
        self._low_level_cmd = to_torch([0.0, 0.0, 0.0, 0.7], device=self.device)

        if high_level_path and os.path.exists(high_level_path):
            print(f"Loading High-level policy: {high_level_path}")
            self.high_level_policy = torch.jit.load(high_level_path, map_location=self.device)
            self.high_level_policy.eval()
            self._high_level_enabled = True

    def _setup_scene(self, robot_xml_path):
        gs.init(logging_level="DEBUG")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=4),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=self.render,
        )
        self.scene.add_entity(gs.morphs.Plane())
        
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=robot_xml_path, pos=np.array([0.0, 0.0, 0.793]), quat=np.array([1.0, 0.0, 0.0, 0.0])),
            # vis_mode="collision",
        )

        # Setup Box and Target
        self.box_size = cfg.BOX_SIZE

        self.box = self.scene.add_entity(
            gs.morphs.Box(size=self.box_size, pos=np.array([4.17, 0.07, 0.8])),
            material=gs.materials.Rigid(coup_friction=1.0,friction=1.0),
        )
        self.table = self.scene.add_entity(
            gs.morphs.Box(size=[0.2,0.2,0.5], pos=np.array([4.17, 0.07, 0.3])),
        )
        self.box_target_pos = np.array([4.0, 2.0, 0.62])
        self.target_marker = self.scene.add_entity(
            gs.morphs.Box(size=(0.52, 0.52, 0.01), pos=self.box_target_pos)
        )

        self.scene.build(n_envs=1)
        self._configure_robot_motors()
        self.box.set_mass(0.3)
        self.table.set_mass(100)


    def _configure_robot_motors(self):
        self.motor_dofs_29 = []
        for name in cfg.GENESIS_JOINT_NAMES_29:
            joint = self.robot.get_joint(name)
            if joint is not None and hasattr(joint, "dofs_idx_local") and len(joint.dofs_idx_local) > 0:
                self.motor_dofs_29.append(joint.dofs_idx_local[0])

        kp_values, kd_values = cfg.get_pd_gains(cfg.GENESIS_JOINT_NAMES_29)
        self.robot.set_dofs_position(self.default_qpos_29, self.motor_dofs_29, zero_velocity=True)
        self.robot.set_dofs_kp(kp_values, self.motor_dofs_29)
        self.robot.set_dofs_kv(kd_values, self.motor_dofs_29)
        self.robot.set_pos(np.array([0.0, 0.0, 0.80]))
        self.robot.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

    def _setup_joint_indices(self):
        name_to_idx29 = {n: i for i, n in enumerate(cfg.GENESIS_JOINT_NAMES_29)}
        self.input_dofs_27 = [name_to_idx29[n] for n in cfg.LOW_LEVEL_JOINT_NAMES_27]
        self.leg_dofs_12 = [name_to_idx29[n] for n in cfg.LOW_LEVEL_OUTPUT_LEG_12]

        self.legs_dofs_local = []
        for n in cfg.LOW_LEVEL_OUTPUT_LEG_12:
            j = self.robot.get_joint(n)
            if j and hasattr(j, "dofs_idx_local"): self.legs_dofs_local.append(j.dofs_idx_local[0])

        self.left_arm_dofs_local = []
        for n in cfg.LEFT_ARM_JOINT_NAMES:
            j = self.robot.get_joint(n)
            if j and hasattr(j, "dofs_idx_local"): self.left_arm_dofs_local.append(j.dofs_idx_local[0])
            
        self.right_arm_dofs_local = []
        for n in cfg.RIGHT_ARM_JOINT_NAMES:
            j = self.robot.get_joint(n)
            if j and hasattr(j, "dofs_idx_local"): self.right_arm_dofs_local.append(j.dofs_idx_local[0])

        self.left_arm_indices29 = [name_to_idx29[n] for n in cfg.LEFT_ARM_JOINT_NAMES if n in name_to_idx29]
        self.right_arm_indices29 = [name_to_idx29[n] for n in cfg.RIGHT_ARM_JOINT_NAMES if n in name_to_idx29]

        self.left_ee_link = self.robot.get_link("left_rubber_hand_target")
        self.right_ee_link = self.robot.get_link("right_rubber_hand_target")
        
        self.high_policy_motor_dofs = list(self.motor_dofs_29)
        wj = self.robot.get_joint("waist_pitch_joint")
        self.waist_pitch_local = [wj.dofs_idx_local[0]] if (wj and hasattr(wj, "dofs_idx_local")) else []

    def _extract_arm_cmd(self, qpos_arr, arm_indices29, arm_dofs_local):
        if qpos_arr is None: return None
        if isinstance(qpos_arr, torch.Tensor): q = qpos_arr.detach().cpu().numpy().reshape(-1)
        else: q = np.array(qpos_arr).reshape(-1)
        
        L = q.shape[0]
        if L >= 36:   base_offset = 7; vals = [float(q[base_offset + i]) for i in arm_indices29]
        elif L >= 29: base_offset = L - 29; vals = [float(q[base_offset + i]) for i in arm_indices29]
        else:         vals = [float(v) for v in q[: len(arm_indices29)]]
        
        return to_torch(vals, device=self.device).unsqueeze(0)

    def _compute_single_obs(self):
        base_quat = to_torch(self.robot.get_quat()[0], device=self.device)
        base_ang = to_torch(self.robot.get_ang()[0], device=self.device)
        proj_g = quat_apply_inverse(base_quat, to_torch([0.0, 0.0, -1.0], device=self.device))

        qpos29 = to_torch(self.robot.get_dofs_position(self.motor_dofs_29)[0], device=self.device)
        qvel29 = to_torch(self.robot.get_dofs_velocity(self.motor_dofs_29)[0], device=self.device)
        qpos27 = qpos29[self.input_dofs_27]
        qvel27 = qvel29[self.input_dofs_27]

        cmd4 = getattr(self, "_low_level_cmd", None)
        cmd3 = cmd4[:3] * self.cmd_scale
        height_scaled = cmd4[3:4]

        default_pos_27 = torch.zeros(27, device=self.device)
        default_pos_27[:12] = self.low_level_default_qpos_27[:12]
        joint_pos_scaled = (qpos27 - default_pos_27) * self.dof_pos_scale
        joint_vel_scaled = qvel27 * self.dof_vel_scale
        last_act = to_torch(self.last_leg_action, device=self.device)

        obs = torch.cat([
            cmd3, height_scaled, base_ang * self.ang_vel_scale, proj_g,
            joint_pos_scaled, joint_vel_scaled, last_act
        ], dim=0)
        return obs.detach().cpu().numpy().astype(np.float32)

    def _build_obs_batch(self):
        single = self._compute_single_obs()
        self.obs_history.append(single)
        while len(self.obs_history) < self.obs_history_len:
            self.obs_history.appendleft(single.copy())
        return np.concatenate(list(self.obs_history), axis=0).astype(np.float32).reshape(1, -1)

    @torch.inference_mode()
    def _policy_act(self, obs_batch_np):
        obs_t = torch.from_numpy(obs_batch_np).to(self.device)
        out = self.policy(obs_t)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        return np.array(out).squeeze().astype(np.float32)


    def _compute_single_high_level_obs(self):
        # ... logic identical to original but using helpers ...
        # Simplified for brevity in refactor display, but keep full logic from original
        qpos = to_torch(self.robot.get_dofs_position(self.high_policy_motor_dofs)[0], device=self.device)
        qvel = to_torch(self.robot.get_dofs_velocity(self.high_policy_motor_dofs)[0], device=self.device)
        base_pos = to_torch(self.robot.get_pos()[0], device=self.device)
        base_quat = to_torch(self.robot.get_quat()[0], device=self.device)
        base_ang = to_torch(self.robot.get_ang()[0], device=self.device) # world frame angular velocity
        base_ang_robot = quat_apply_inverse(base_quat, base_ang)

        proj_g = quat_apply_inverse(base_quat, to_torch([0.0, 0.0, -1.0], device=self.device))
        joint_pos = (qpos - self.default_qpos_isaac_29)
        joint_vel = qvel
        last_actions = getattr(self, "_last_high_actions", torch.zeros(19, device=self.device))

        l_pos_w = to_torch(self.left_ee_link.get_pos()[0], device=self.device)
        r_pos_w = to_torch(self.right_ee_link.get_pos()[0], device=self.device)

        left_rel = quat_apply_inverse(base_quat, l_pos_w - base_pos)
        right_rel = quat_apply_inverse(base_quat, r_pos_w - base_pos)


        if self.box is not None:
            box_pos = to_torch(self.box.get_pos()[0], device=self.device)
            box_quat = to_torch(self.box.get_quat()[0], device=self.device)
            
            box_pos_robot = quat_apply_inverse(base_quat, box_pos - base_pos)
            box_quat_rel = quat_mul(quat_inv(base_quat), box_quat)
            
            # 2. 对相对四元数进行编码
            box_quat_tan = quat_to_tan_norm(box_quat_rel)
            object_root_info = torch.cat([box_pos_robot, box_quat_tan.squeeze(0)])

            half = to_torch(self.box_size, device=self.device) / 2.0
            corners = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        c_local = to_torch([sx * half[0], sy * half[1], sz * half[2]], device=self.device)
                        c_world = box_pos + quat_apply(box_quat, c_local)
                        c_robot = quat_apply_inverse(base_quat, c_world - base_pos)
                        corners.append(c_robot)

            standing_offset = to_torch(cfg.BOX_STANDING_OFFSET, device=self.device)
            standing_world = box_pos + quat_apply(box_quat, standing_offset)
            standing_world[2] = 0
            
            # Draw debug
            if self.render:
                if self.standing_point_debug_node:
                    self.scene.clear_debug_object(self.standing_point_debug_node)
                self.standing_point_debug_node = self.scene.draw_debug_spheres(
                    standing_world.detach().cpu().numpy().reshape(1, 3), radius=0.1, color=(0., 1., 0., 1.))

            target_pos = to_torch(self.box_target_pos, device=self.device)
            target_pos_robot = quat_apply_inverse(base_quat, target_pos - base_pos)
            target_quat = quat_to_tan_norm(to_torch([1.0, 0.0, 0.0, 0.0], device=self.device))
            object_target_pose = torch.cat([target_pos_robot, target_quat.squeeze(0)])
            box_size = torch.tensor([self.box_size[0]], device=self.device)
        return torch.cat([
            base_ang_robot, proj_g, joint_pos, joint_vel, last_actions,
            left_rel, right_rel, object_root_info, object_target_pose, box_size
        ], dim=0).unsqueeze(0).to(dtype=torch.float32)

    def _compute_high_level_obs(self):
            # 1. 获取当前帧 (108,)
            single = self._compute_single_high_level_obs().squeeze(0)
            
            # 2. 更新 Buffer (FIFO)
            # 这里的 buffer 依然是 [Time, Feature] 结构，例如 [6, 108]
            self.high_obs_buffer = torch.roll(self.high_obs_buffer, shifts=-1, dims=0)
            self.high_obs_buffer[-1, :] = single
            
            # 3. 重组输出：将 [Time, Feature] 转为 [Feature_History, ...]
            output_parts = []
            start_idx = 0
            
            # 遍历我们在 init 里定义的结构
            for name, dim in self.high_obs_segments:
                end_idx = start_idx + dim
                
                # 核心逻辑：
                # 取出该特征的所有历史数据 -> shape: (History_Len, Dim)
                # 例如 joint_pos 取出来是 (6, 29)
                segment_hist = self.high_obs_buffer[:, start_idx:end_idx]
                
                # 展平该特征的历史 -> shape: (History_Len * Dim,)
                # 结果就是 [t-5_pos[0], t-5_pos[1]... t_pos[0], t_pos[1]...]
                output_parts.append(segment_hist.reshape(-1))
                
                start_idx = end_idx

            # 4. 拼接所有特征的历史并增加 batch 维度 -> (1, 648)
            return torch.cat(output_parts, dim=0).unsqueeze(0)
        

    def step_control(self):
        hl_left_pos_delta = hl_left_euler_delta = hl_right_pos_delta = hl_right_euler_delta = None
        
        # High Level Logic
        with torch.inference_mode():
            high_act = self.high_level_policy(self._compute_high_level_obs())
            if isinstance(high_act, (list, tuple)): high_act = high_act[0]
            high_act = to_torch(high_act, device=self.device).squeeze(0)
            self._last_high_actions = high_act.detach()
            
            self._low_level_cmd = high_act[0:4] * self.hierarchical_action_scale
            if len(self.waist_pitch_local) == 1:
                self.waist_pitch_target = float(high_act[4].item() * self.waist_action_scale)
            
            left_arm = high_act[7:13]; right_arm = high_act[13:19]
            hl_left_pos_delta = left_arm[0:3] * self.arm_ik_action_scale
            hl_left_euler_delta = left_arm[3:6] * self.arm_ik_action_scale
            hl_right_pos_delta = right_arm[0:3] * self.arm_ik_action_scale
            hl_right_euler_delta = right_arm[3:6] * self.arm_ik_action_scale


        # Low Level Logic
        act12 = self._policy_act(self._build_obs_batch())
        # act12 = np.clip(act12, -3.0, 3.0)
        self.last_leg_action = act12.copy()

        target29 = self.default_qpos_29.clone().cpu().numpy()
        target29[self.leg_dofs_12] += act12 * self.policy_action_scale_legs
        self.robot.control_dofs_position(to_torch(target29, device=self.device), self.motor_dofs_29)
        self.robot.control_dofs_position([getattr(self, 'waist_pitch_target', 0.0)], self.waist_pitch_local)

        # IK / Arm Control
        self._handle_arm_ik_new(hl_left_pos_delta, hl_left_euler_delta, hl_right_pos_delta, hl_right_euler_delta)
        


    def _handle_arm_ik_new(self, l_pos_d, l_eu_d, r_pos_d, r_eu_d):
        # --- 左臂 ---
        base_quat = to_torch(self.robot.get_quat()[0], device=self.device)
        if l_pos_d is not None:
            # A. 位置变换: Base Frame -> World Frame
            l_pos_d_world = quat_apply(base_quat, l_pos_d)
            
            # B. 旋转变换: 将 Euler Delta 视为角速度向量，从 Base Frame 转到 World Frame
            #    这一步至关重要，否则你的手腕转动方向会是乱的
            l_eu_d_world = quat_apply(base_quat, l_eu_d)            
            delta_q = self.ik_controller.solve(
                link=self.left_ee_link,  # <--- 传入新加的 Link
                local_dofs=self.left_arm_dofs_local,
                pos_delta=l_pos_d_world,
                euler_delta=l_eu_d_world
            )
            # 执行控制
            curr_q = self.robot.get_dofs_position(self.left_arm_dofs_local)
            self.robot.control_dofs_position(curr_q + delta_q, self.left_arm_dofs_local)

        # --- 右臂 ---
        if r_pos_d is not None:
            # 同理变换
            r_pos_d_world = quat_apply(base_quat, r_pos_d)
            r_eu_d_world = quat_apply(base_quat, r_eu_d)
            delta_q = self.ik_controller.solve(
                link=self.right_ee_link, # <--- 传入新加的 Link
                local_dofs=self.right_arm_dofs_local,
                pos_delta=r_pos_d_world,
                euler_delta=r_eu_d_world
            )
            curr_q = self.robot.get_dofs_position(self.right_arm_dofs_local)
            self.robot.control_dofs_position(curr_q + delta_q, self.right_arm_dofs_local)


    def _handle_arm_ik(self, l_pos_d, l_eu_d, r_pos_d, r_eu_d):
        if l_pos_d is not None and r_pos_d is not None:
            self._solve_and_control_ik_new(
                self.left_ee_link, self.left_arm_dofs_local, 
                self.left_arm_indices29, l_pos_d, l_eu_d
            )
            self._solve_and_control_ik_new(
                self.right_ee_link, self.right_arm_dofs_local, 
                self.right_arm_indices29, r_pos_d, r_eu_d
            )

    def _solve_and_control_ik(self, link, local_dofs, indices29, pos_delta, euler_delta):
        if link is None or not local_dofs:
            return
        
        curr_pos = to_torch(link.get_pos()[0], device=self.device)
        curr_quat = to_torch(link.get_quat()[0], device=self.device)
        
        tgt_pos = (curr_pos + pos_delta).detach().cpu().numpy().reshape(1, -1)
        
        curr_euler = quat_to_euler_wxyz(curr_quat, device=self.device)
        new_euler = curr_euler + euler_delta
        tgt_quat = euler_to_quat_wxyz(new_euler, device=self.device).detach().cpu().numpy().reshape(1, -1)

            
        qpos, _ = self.robot.inverse_kinematics(
            link=link, pos=tgt_pos, quat=tgt_quat, 
            return_error=True, dofs_idx_local=local_dofs
        )
        cmd = self._extract_arm_cmd(qpos, indices29, local_dofs)
        self.robot.control_dofs_position(cmd, local_dofs)

    def _solve_and_control_ik_new(self, link, local_dofs, indices29, pos_delta, euler_delta):
        if link is None or not local_dofs:
            return

        curr_pos_w = to_torch(link.get_pos()[0], device=self.device)
        curr_quat_w = to_torch(link.get_quat()[0], device=self.device)

        base_quat_w = to_torch(self.robot.get_quat()[0], device=self.device)
        
        # --- 2. 坐标系变换 (核心修正) ---
        
        # A. 位置处理：将 Base Frame 的 delta 转换到 World Frame
        # pos_delta_world = Rotate_by_Base(pos_delta_local)
        pos_delta_w = quat_apply(base_quat_w, pos_delta)
        tgt_pos_w = curr_pos_w + pos_delta_w
        
        # B. 旋转处理：将 Euler Delta 叠加
        # 简单做法：把 Euler Delta 转成 Quat Delta (假设是在 Base Frame 下的旋转)
        # 严谨做法比较复杂，这里采用近似：在当前 World Frame 姿态上叠加 Delta
        # 为了 Sim2Sim 对齐，通常 policy 输出的 euler delta 是相对于当前末端姿态的“微调”
        # 但这个微调的方向是定义在 Base Frame 的
        
        # 1. 把 euler (roll, pitch, yaw) 转成 delta quaternion
        delta_quat_local = euler_to_quat_wxyz(euler_delta)
        
        # 2. 把局部旋转转到世界坐标系: q_delta_world = q_base * q_delta_local * q_base_inv
        delta_quat_w = quat_mul(
            quat_mul(base_quat_w, delta_quat_local), 
            quat_inv(base_quat_w)
        )
        
        # 3. 应用旋转: q_target = q_delta_world * q_curr (左乘表示在全局轴旋转，或者根据定义右乘)
        # Isaac Lab 通常是 Local 叠加，这里我们尝试叠加到当前姿态
        tgt_quat_w = quat_mul(delta_quat_w, curr_quat_w)
        
        # 准备数据给 Genesis IK (转 numpy)
        tgt_pos_np = tgt_pos_w.detach().cpu().numpy().flatten().reshape(1, -1)
        tgt_quat_np = tgt_quat_w.detach().cpu().numpy().flatten().reshape(1, -1)
        
        # --- 3. 关键：获取当前关节角度作为 Seed ---
        # 这一步是为了防止 IK 解跳变 (比如胳膊突然反转)
        curr_qpos = self.robot.get_qpos() # 获取全身 qpos
        
        # --- 4. 调用 Genesis IK ---
        qpos_sol, err = self.robot.inverse_kinematics(
            link=link, 
            pos=tgt_pos_np, 
            quat=tgt_quat_np, 
            init_qpos=curr_qpos,    # <--- 必须传这个！
            respect_joint_limit=True,
            damping=0.1,            # 增加阻尼，防止奇异点飞车
            return_error=True, 
            dofs_idx_local=local_dofs
        )
        
        # --- 5. 提取控制量 ---
        # qpos_sol 是全身的或者对应 env 的，需要提取对应关节的
        # Genesis 返回的 qpos_sol 通常是 (n_envs, n_dofs) 或 (n_dofs,)
        # 如果是 tensor，需要处理；如果是 numpy，直接提取
        
        cmd = self._extract_arm_cmd(qpos_sol, indices29, local_dofs)
        self.robot.control_dofs_position(cmd, local_dofs)

    def run(self, duration_sec: float = 30.0):
        start = time.time()
        counter = 0
        while time.time() - start < duration_sec:
            counter += 1
            if counter % self.control_decimation == 0:
                self.step_control()
                self.scene.step()