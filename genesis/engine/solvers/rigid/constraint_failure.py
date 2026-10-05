"""Immutable first non-finite constraint solve, captured before integration.

Version 4 records each island's line-search diagnostics and dof membership.
Unused qacc_prev and qf_actuator fields removed upstream are no longer reported.
Raw contact data is associated with constraint rows through contact_sort_idx.
"""
import dataclasses
import numpy as np
import quadrants as qd
import genesis as gs
from genesis.utils import array_class
from genesis.utils.array_class import V, V_VEC, V_SCALAR_FROM
from genesis.utils.misc import qd_to_numpy

def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return "nan"
        if np.isposinf(number):
            return "+inf"
        if np.isneginf(number):
            return "-inf"
        return number
    return value

@dataclasses.dataclass(eq=True, kw_only=False, frozen=True)
class ConstraintFailureSnapshot:
    captured: qd.Tensor
    candidate_environment_index: qd.Tensor
    invalid_environment_mask: qd.Tensor
    environment_index: qd.Tensor
    first_invalid_dof_index: qd.Tensor
    substep_global: qd.Tensor
    n_contacts: qd.Tensor
    n_constraints: qd.Tensor
    n_constraints_equality: qd.Tensor
    n_constraints_frictionloss: qd.Tensor
    cost: qd.Tensor
    n_islands: qd.Tensor
    dofs_island_idx: qd.Tensor
    island_ls_evaluated: qd.Tensor
    island_ls_improvement: qd.Tensor
    island_ls_alpha: qd.Tensor
    island_ls_gtol: qd.Tensor
    island_ls_it: qd.Tensor
    island_ls_result: qd.Tensor
    qpos: qd.Tensor
    geom_friction_ratio: qd.Tensor
    dof_pos: qd.Tensor
    dof_vel: qd.Tensor
    dof_vel_prev: qd.Tensor
    dof_vel_next: qd.Tensor
    dof_acc: qd.Tensor
    dof_acc_before_solve: qd.Tensor
    dof_acc_smooth: qd.Tensor
    dof_qf_smooth: qd.Tensor
    dof_qf_constraint: qd.Tensor
    dof_qf_applied: qd.Tensor
    dof_qf_bias: qd.Tensor
    dof_qf_passive: qd.Tensor
    dof_force: qd.Tensor
    dof_ctrl_force: qd.Tensor
    dof_ctrl_pos: qd.Tensor
    dof_ctrl_vel: qd.Tensor
    dof_ctrl_mode: qd.Tensor
    link_pos: qd.Tensor
    link_quat: qd.Tensor
    link_cd_vel: qd.Tensor
    link_cd_ang: qd.Tensor
    link_inertial_mass: qd.Tensor
    link_inertial_position_world: qd.Tensor
    link_cfrc_applied_ang: qd.Tensor
    link_cfrc_applied_vel: qd.Tensor
    link_contact_force: qd.Tensor
    contact_sort_idx: qd.Tensor
    contact_geom_a: qd.Tensor
    contact_geom_b: qd.Tensor
    contact_link_a: qd.Tensor
    contact_link_b: qd.Tensor
    contact_penetration: qd.Tensor
    contact_normal: qd.Tensor
    contact_pos: qd.Tensor
    contact_friction: qd.Tensor
    contact_sol_params: qd.Tensor
    contact_force: qd.Tensor
    constraint_qacc: qd.Tensor
    constraint_qacc_ws: qd.Tensor
    constraint_qacc_ws_before_solve: qd.Tensor
    pre_solve_qacc_ws: qd.Tensor
    pre_solve_dof_acc: qd.Tensor
    pre_solve_is_warmstart: qd.Tensor
    is_warmstart_before_solve: qd.Tensor
    constraint_qfrc: qd.Tensor
    constraint_Ma: qd.Tensor
    constraint_grad: qd.Tensor
    constraint_diag: qd.Tensor
    constraint_aref: qd.Tensor
    constraint_Jaref: qd.Tensor
    constraint_efc_D: qd.Tensor
    constraint_efc_force: qd.Tensor
    constraint_active: qd.Tensor
    constraint_jac: qd.Tensor

def get_constraint_failure_snapshot(solver):
    max_contacts = solver.collider.collider_state.contact_data.geom_a.shape[0]
    len_constraints = solver.constraint_solver.len_constraints_
    invalid_environment_mask = V(dtype=gs.qd_int, shape=(solver._B,))
    invalid_environment_mask.fill(0)

    return ConstraintFailureSnapshot(
        captured=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
        candidate_environment_index=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._B),
        invalid_environment_mask=invalid_environment_mask,
        environment_index=V_SCALAR_FROM(dtype=gs.qd_int, value=solver._B),
        first_invalid_dof_index=V_SCALAR_FROM(dtype=gs.qd_int, value=solver.n_dofs_),
        substep_global=V_SCALAR_FROM(dtype=qd.i64 if gs.backend != gs.metal else qd.i32, value=-1),
        n_contacts=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
        n_constraints=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
        n_constraints_equality=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
        n_constraints_frictionloss=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
        cost=V_SCALAR_FROM(dtype=gs.qd_float, value=0.0),
        n_islands=V_SCALAR_FROM(dtype=gs.qd_int, value=0),
        dofs_island_idx=V(dtype=gs.qd_int, shape=(solver.n_dofs_,)),
        island_ls_evaluated=V(dtype=gs.qd_bool, shape=(solver.n_trees_,)),
        island_ls_improvement=V(dtype=gs.qd_float, shape=(solver.n_trees_,)),
        island_ls_alpha=V(dtype=gs.qd_float, shape=(solver.n_trees_,)),
        island_ls_gtol=V(dtype=gs.qd_float, shape=(solver.n_trees_,)),
        island_ls_it=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        island_ls_result=V(dtype=gs.qd_int, shape=(solver.n_trees_,)),
        qpos=V(dtype=gs.qd_float, shape=(solver.n_qs_,)),
        geom_friction_ratio=V(dtype=gs.qd_float, shape=(solver.n_geoms_,)),
        dof_pos=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_vel=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_vel_prev=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_vel_next=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_acc=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_acc_before_solve=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_acc_smooth=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_qf_smooth=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_qf_constraint=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_qf_applied=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_qf_bias=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_qf_passive=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_force=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_ctrl_force=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_ctrl_pos=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_ctrl_vel=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        dof_ctrl_mode=V(dtype=gs.qd_int, shape=(solver.n_dofs_,)),
        link_pos=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        link_quat=V(dtype=gs.qd_vec4, shape=(solver.n_links_,)),
        link_cd_vel=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        link_cd_ang=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        link_inertial_mass=V(dtype=gs.qd_float, shape=(solver.n_links_,)),
        link_inertial_position_world=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        link_cfrc_applied_ang=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        link_cfrc_applied_vel=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        link_contact_force=V(dtype=gs.qd_vec3, shape=(solver.n_links_,)),
        contact_sort_idx=V(dtype=gs.qd_int, shape=(max_contacts,)),
        contact_geom_a=V(dtype=gs.qd_int, shape=(max_contacts,)),
        contact_geom_b=V(dtype=gs.qd_int, shape=(max_contacts,)),
        contact_link_a=V(dtype=gs.qd_int, shape=(max_contacts,)),
        contact_link_b=V(dtype=gs.qd_int, shape=(max_contacts,)),
        contact_penetration=V(dtype=gs.qd_float, shape=(max_contacts,)),
        contact_normal=V(dtype=gs.qd_vec3, shape=(max_contacts,)),
        contact_pos=V(dtype=gs.qd_vec3, shape=(max_contacts,)),
        contact_friction=V(dtype=gs.qd_float, shape=(max_contacts,)),
        contact_sol_params=V_VEC(7, dtype=gs.qd_float, shape=(max_contacts,)),
        contact_force=V(dtype=gs.qd_vec3, shape=(max_contacts,)),
        constraint_qacc=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        constraint_qacc_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        constraint_qacc_ws_before_solve=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        pre_solve_qacc_ws=V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver._B)),
        pre_solve_dof_acc=V(dtype=gs.qd_float, shape=(solver.n_dofs_, solver._B)),
        pre_solve_is_warmstart=V(dtype=gs.qd_bool, shape=(solver._B,)),
        is_warmstart_before_solve=V_SCALAR_FROM(dtype=gs.qd_bool, value=False),
        constraint_qfrc=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        constraint_Ma=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        constraint_grad=V(dtype=gs.qd_float, shape=(solver.n_dofs_,)),
        constraint_diag=V(dtype=gs.qd_float, shape=(len_constraints,)),
        constraint_aref=V(dtype=gs.qd_float, shape=(len_constraints,)),
        constraint_Jaref=V(dtype=gs.qd_float, shape=(len_constraints,)),
        constraint_efc_D=V(dtype=gs.qd_float, shape=(len_constraints,)),
        constraint_efc_force=V(dtype=gs.qd_float, shape=(len_constraints,)),
        constraint_active=V(dtype=gs.qd_bool, shape=(len_constraints,)),
        constraint_jac=V(dtype=gs.qd_float, shape=(len_constraints, solver.n_dofs_)),
    )

def diagnostic(self) -> dict:
    snapshot = self._constraint_failure_snapshot

    def value(field):
        return qd_to_numpy(field)

    def scalar(field):
        return value(field).item()

    env_index = int(scalar(snapshot.environment_index))
    n_contacts = int(scalar(snapshot.n_contacts))
    n_constraints = int(scalar(snapshot.n_constraints))
    first_invalid_dof_index = int(scalar(snapshot.first_invalid_dof_index))

    contact_fields = {
        name: value(getattr(snapshot, f"contact_{name}"))
        for name in (
            "geom_a",
            "geom_b",
            "link_a",
            "link_b",
            "penetration",
            "normal",
            "pos",
            "friction",
            "sol_params",
            "force",
        )
    }
    contact_constraint_start = int(
        scalar(snapshot.n_constraints_equality)
        + scalar(snapshot.n_constraints_frictionloss)
    )
    efc_force = value(snapshot.constraint_efc_force)
    contacts = []
    contact_sort = value(snapshot.contact_sort_idx)
    rows_per_contact = self.rigid_config.rows_per_contact
    for sorted_index in range(n_contacts):
        contact_index = int(contact_sort[sorted_index])
        geom_a = int(contact_fields["geom_a"][contact_index])
        geom_b = int(contact_fields["geom_b"][contact_index])
        constraint_start = contact_constraint_start + rows_per_contact * sorted_index
        contacts.append(
            {
                "contact_index": contact_index,
                "geom_a": geom_a,
                "geom_b": geom_b,
                "geom_a_link": self.geoms[geom_a].link.name,
                "geom_b_link": self.geoms[geom_b].link.name,
                "link_a": int(contact_fields["link_a"][contact_index]),
                "link_b": int(contact_fields["link_b"][contact_index]),
                "penetration": contact_fields["penetration"][contact_index],
                "normal": contact_fields["normal"][contact_index],
                "position": contact_fields["pos"][contact_index],
                "friction": contact_fields["friction"][contact_index],
                "sol_params": contact_fields["sol_params"][contact_index],
                "reported_force": contact_fields["force"][contact_index],
                "constraint_force": efc_force[
                    constraint_start : constraint_start + rows_per_contact
                ],
            }
        )

    dof_fields = {
        name: value(getattr(snapshot, f"dof_{name}"))
        for name in (
            "pos",
            "vel",
            "vel_prev",
            "vel_next",
            "acc",
            "acc_before_solve",
            "acc_smooth",
            "qf_smooth",
            "qf_constraint",
            "qf_applied",
            "qf_bias",
            "qf_passive",
            "force",
            "ctrl_force",
            "ctrl_pos",
            "ctrl_vel",
            "ctrl_mode",
        )
    }
    constraint_fields = {
        "qacc": value(snapshot.constraint_qacc),
        "qacc_ws": value(snapshot.constraint_qacc_ws),
        "qacc_ws_before_solve": value(snapshot.constraint_qacc_ws_before_solve),
        "qfrc_constraint": value(snapshot.constraint_qfrc),
        "Ma": value(snapshot.constraint_Ma),
        "grad": value(snapshot.constraint_grad),
        "diag": value(snapshot.constraint_diag)[:n_constraints],
        "aref": value(snapshot.constraint_aref)[:n_constraints],
        "Jaref": value(snapshot.constraint_Jaref)[:n_constraints],
        "efc_D": value(snapshot.constraint_efc_D)[:n_constraints],
        "efc_force": efc_force[:n_constraints],
        "active": value(snapshot.constraint_active)[:n_constraints],
        "jac": value(snapshot.constraint_jac)[:n_constraints],
    }
    scalar_constraint_fields = {
        name: scalar(getattr(snapshot, name))
        for name in (
            "n_constraints",
            "n_constraints_equality",
            "n_constraints_frictionloss",
            "is_warmstart_before_solve",
            "cost",
        )
    }
    dof_names = [f"dof_{index}" for index in range(self.n_dofs_)]
    for joint in self.joints:
        for local_index, dof_index in enumerate(
            range(joint.dof_start, joint.dof_end)
        ):
            suffix = f"[{local_index}]" if joint.n_dofs > 1 else ""
            dof_names[dof_index] = f"{joint.name}{suffix}"

    equality_end = int(scalar(snapshot.n_constraints_equality))
    frictionloss_end = equality_end + int(
        scalar(snapshot.n_constraints_frictionloss)
    )
    contact_end = frictionloss_end + rows_per_contact * n_contacts
    constraint_rows = []
    jac = constraint_fields["jac"]
    for row_index in range(n_constraints):
        if row_index < equality_end:
            kind = "equality"
            source = {}
        elif row_index < frictionloss_end:
            kind = "friction_loss"
            source = {}
        elif row_index < contact_end:
            kind = "contact"
            contact_row = row_index - frictionloss_end
            source = {
                "contact_index": int(contact_sort[contact_row // rows_per_contact]),
                "contact_direction_index": contact_row % rows_per_contact,
            }
        else:
            kind = "joint_limit"
            source = {}
        relevant_dofs = np.flatnonzero(np.abs(jac[row_index]) > gs.EPS)
        constraint_rows.append(
            {
                "row_index": row_index,
                "kind": kind,
                **source,
                "relevant_dof_indices": relevant_dofs,
                "relevant_dof_names": [
                    dof_names[index] for index in relevant_dofs
                ],
                "active": constraint_fields["active"][row_index],
                "diag": constraint_fields["diag"][row_index],
                "aref": constraint_fields["aref"][row_index],
                "Jaref": constraint_fields["Jaref"][row_index],
                "efc_D": constraint_fields["efc_D"][row_index],
                "efc_force": constraint_fields["efc_force"][row_index],
            }
        )
    link_fields = {
        name: value(getattr(snapshot, f"link_{name}"))
        for name in (
            "pos",
            "quat",
            "cd_vel",
            "cd_ang",
            "inertial_mass",
            "inertial_position_world",
            "cfrc_applied_ang",
            "cfrc_applied_vel",
            "contact_force",
        )
    }
    invalid_environment_mask = value(snapshot.invalid_environment_mask)

    return _json_safe(
        {
            "schema_version": 4,
            "removed_upstream_fields": ["constraint.qacc_prev", "dofs.qf_actuator"],
            "error": "invalid_constraint_qacc",
            "snapshot": "first_failure_pre_integration",
            "environment_index": env_index,
            "first_invalid_dof_index": first_invalid_dof_index,
            "invalid_environment_indices": np.flatnonzero(
                invalid_environment_mask
            ),
            "invalid_dof_indices": np.flatnonzero(
                ~np.isfinite(constraint_fields["qacc"])
            ),
            "simulation": {
                "substep_global": int(scalar(snapshot.substep_global)),
                "substep_dt_s": self._substep_dt,
                "n_envs": self._B,
                "exact_terrain_surface": (
                    self.collider.collider_config.has_exact_terrain_surface
                ),
            },
            "replay_state": {
                "qpos": value(snapshot.qpos),
                "geom_friction_ratio": value(snapshot.geom_friction_ratio),
            },
            "constraint": {
                **scalar_constraint_fields,
                **constraint_fields,
                "rows": constraint_rows,
                "dofs_island_idx": value(snapshot.dofs_island_idx),
                "islands": [
                    {
                        "island_index": i,
                        "line_search_evaluated": bool(value(snapshot.island_ls_evaluated)[i]),
                        **{
                            name: value(getattr(snapshot, "island_" + name))[i]
                            if bool(value(snapshot.island_ls_evaluated)[i]) else None
                            for name in ("ls_improvement", "ls_alpha", "ls_gtol", "ls_it", "ls_result")
                        },
                    }
                    for i in range(int(scalar(snapshot.n_islands)))
                ],
            },
            "dofs": {"names": dof_names, **dof_fields},
            "links": {
                "names": [link.name for link in self.links],
                **link_fields,
            },
            "contacts": contacts,
        }
    )

@qd.kernel(fastcache=True)
def kernel_detect_constraint_failure(
    constraint_state: array_class.ConstraintState,
    snapshot: ConstraintFailureSnapshot,
):
    n_dofs = constraint_state.qacc.shape[0]
    _B = constraint_state.qacc.shape[1]
    # Top-level loops keep the full-environment scan parallel on the GPU.
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if snapshot.captured[None] == 0:
            if qd.math.isnan(constraint_state.qacc[i_d, i_b]) or qd.math.isinf(constraint_state.qacc[i_d, i_b]):
                qd.atomic_min(snapshot.candidate_environment_index[None], i_b)
                snapshot.invalid_environment_mask[i_b] = 1

@qd.kernel(fastcache=True)
def kernel_capture_constraint_failure(
    rigid_global_info: array_class.RigidInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    constraint_state: array_class.ConstraintState,
    snapshot: ConstraintFailureSnapshot,
    substep_global: qd.i32,
):
    if snapshot.captured[None] == 0:
        i_b = snapshot.candidate_environment_index[None]
        if i_b < dofs_state.acc.shape[1]:
            snapshot.captured[None] = 1
            snapshot.environment_index[None] = i_b
            snapshot.substep_global[None] = substep_global
            snapshot.n_contacts[None] = collider_state.n_contacts[i_b]
            snapshot.n_constraints[None] = constraint_state.n_constraints[i_b]
            snapshot.n_constraints_equality[None] = constraint_state.n_constraints_equality[i_b]
            snapshot.n_constraints_frictionloss[None] = constraint_state.n_constraints_frictionloss[i_b]
            snapshot.cost[None] = constraint_state.cost[i_b]
            snapshot.n_islands[None] = constraint_state.island.n_islands[i_b]
            for i_island in range(constraint_state.island.n_islands[i_b]):
                snapshot.island_ls_evaluated[i_island] = constraint_state.island.ls_evaluated[i_island, i_b]
                snapshot.island_ls_improvement[i_island] = constraint_state.island.ls_improvement[i_island, i_b]
                snapshot.island_ls_alpha[i_island] = constraint_state.island.ls_alpha[i_island, i_b]
                snapshot.island_ls_gtol[i_island] = constraint_state.island.ls_gtol[i_island, i_b]
                snapshot.island_ls_it[i_island] = constraint_state.island.ls_it[i_island, i_b]
                snapshot.island_ls_result[i_island] = constraint_state.island.ls_result[i_island, i_b]
            snapshot.is_warmstart_before_solve[None] = snapshot.pre_solve_is_warmstart[i_b]

            for i_q in range(rigid_global_info.qpos.shape[0]):
                snapshot.qpos[i_q] = rigid_global_info.qpos[i_q, i_b]

            for i_g in range(geoms_state.friction_ratio.shape[0]):
                snapshot.geom_friction_ratio[i_g] = geoms_state.friction_ratio[i_g, i_b]

            for i_d in range(dofs_state.acc.shape[0]):
                snapshot.dofs_island_idx[i_d] = constraint_state.island.dofs_island_idx[i_d, i_b]
                snapshot.dof_pos[i_d] = dofs_state.pos[i_d, i_b]
                snapshot.dof_vel[i_d] = dofs_state.vel[i_d, i_b]
                snapshot.dof_vel_prev[i_d] = dofs_state.vel_prev[i_d, i_b]
                snapshot.dof_vel_next[i_d] = dofs_state.vel_next[i_d, i_b]
                snapshot.dof_acc[i_d] = dofs_state.acc[i_d, i_b]
                snapshot.dof_acc_before_solve[i_d] = snapshot.pre_solve_dof_acc[i_d, i_b]
                snapshot.dof_acc_smooth[i_d] = dofs_state.acc_smooth[i_d, i_b]
                snapshot.dof_qf_smooth[i_d] = dofs_state.qf_smooth[i_d, i_b]
                snapshot.dof_qf_constraint[i_d] = dofs_state.qf_constraint[i_d, i_b]
                snapshot.dof_qf_applied[i_d] = dofs_state.qf_applied[i_d, i_b]
                snapshot.dof_qf_bias[i_d] = dofs_state.qf_bias[i_d, i_b]
                snapshot.dof_qf_passive[i_d] = dofs_state.qf_passive[i_d, i_b]
                snapshot.dof_force[i_d] = dofs_state.force[i_d, i_b]
                snapshot.dof_ctrl_force[i_d] = dofs_state.ctrl_force[i_d, i_b]
                snapshot.dof_ctrl_pos[i_d] = dofs_state.ctrl_pos[i_d, i_b]
                snapshot.dof_ctrl_vel[i_d] = dofs_state.ctrl_vel[i_d, i_b]
                snapshot.dof_ctrl_mode[i_d] = dofs_state.ctrl_mode[i_d, i_b]
                snapshot.constraint_qacc[i_d] = constraint_state.qacc[i_d, i_b]
                snapshot.constraint_qacc_ws[i_d] = constraint_state.qacc_ws[i_d, i_b]
                snapshot.constraint_qacc_ws_before_solve[i_d] = snapshot.pre_solve_qacc_ws[i_d, i_b]
                snapshot.constraint_qfrc[i_d] = constraint_state.qfrc_constraint[i_d, i_b]
                snapshot.constraint_Ma[i_d] = constraint_state.Ma[i_d, i_b]
                snapshot.constraint_grad[i_d] = constraint_state.grad[i_d, i_b]
                if qd.math.isnan(constraint_state.qacc[i_d, i_b]) or qd.math.isinf(constraint_state.qacc[i_d, i_b]):
                    qd.atomic_min(snapshot.first_invalid_dof_index[None], i_d)

            for i_l in range(links_state.pos.shape[0]):
                snapshot.link_pos[i_l] = links_state.pos[i_l, i_b]
                snapshot.link_quat[i_l] = links_state.quat[i_l, i_b]
                snapshot.link_cd_vel[i_l] = links_state.cd_vel[i_l, i_b]
                snapshot.link_cd_ang[i_l] = links_state.cd_ang[i_l, i_b]
                snapshot.link_inertial_mass[i_l] = links_state.cinr_mass[i_l, i_b]
                snapshot.link_inertial_position_world[i_l] = links_state.i_pos[i_l, i_b]
                snapshot.link_cfrc_applied_ang[i_l] = links_state.cfrc_applied_ang[i_l, i_b]
                snapshot.link_cfrc_applied_vel[i_l] = links_state.cfrc_applied_vel[i_l, i_b]
                snapshot.link_contact_force[i_l] = links_state.contact_force[i_l, i_b]

            for i_c in range(collider_state.contact_data.geom_a.shape[0]):
                snapshot.contact_sort_idx[i_c] = collider_state.contact_sort_idx[i_c, i_b]
                snapshot.contact_geom_a[i_c] = collider_state.contact_data.geom_a[i_c, i_b]
                snapshot.contact_geom_b[i_c] = collider_state.contact_data.geom_b[i_c, i_b]
                snapshot.contact_link_a[i_c] = collider_state.contact_data.link_a[i_c, i_b]
                snapshot.contact_link_b[i_c] = collider_state.contact_data.link_b[i_c, i_b]
                snapshot.contact_penetration[i_c] = collider_state.contact_data.penetration[i_c, i_b]
                snapshot.contact_normal[i_c] = collider_state.contact_data.normal[i_c, i_b]
                snapshot.contact_pos[i_c] = collider_state.contact_data.pos[i_c, i_b]
                snapshot.contact_friction[i_c] = collider_state.contact_data.friction[i_c, i_b]
                snapshot.contact_sol_params[i_c] = collider_state.contact_data.sol_params[i_c, i_b]
                snapshot.contact_force[i_c] = collider_state.contact_data.force[i_c, i_b]

            for i_c in range(constraint_state.diag.shape[0]):
                snapshot.constraint_diag[i_c] = constraint_state.diag[i_c, i_b]
                snapshot.constraint_aref[i_c] = constraint_state.aref[i_c, i_b]
                snapshot.constraint_Jaref[i_c] = constraint_state.Jaref[i_c, i_b]
                snapshot.constraint_efc_D[i_c] = constraint_state.efc_D[i_c, i_b]
                snapshot.constraint_efc_force[i_c] = constraint_state.efc_force[i_c, i_b]
                snapshot.constraint_active[i_c] = constraint_state.active[i_c, i_b]
                for i_d in range(dofs_state.acc.shape[0]):
                    snapshot.constraint_jac[i_c, i_d] = constraint_state.jac[i_c, i_d, i_b]


@qd.kernel(fastcache=True)
def capture_pre_solve(dyn_state: array_class.DynState, constraint_state: array_class.ConstraintState,
                      snapshot: ConstraintFailureSnapshot):
    # Keep the copy loops top-level for the same reason as the detection loop.
    for i_island, i_b in qd.ndrange(*constraint_state.island.ls_evaluated.shape):
        constraint_state.island.ls_evaluated[i_island, i_b] = False
    for i_d, i_b in qd.ndrange(*constraint_state.qacc_ws.shape):
        if snapshot.captured[None] == 0:
            snapshot.pre_solve_qacc_ws[i_d, i_b] = constraint_state.qacc_ws[i_d, i_b]
            snapshot.pre_solve_dof_acc[i_d, i_b] = dyn_state.dofs.acc[i_d, i_b]
    for i_b in range(constraint_state.is_warmstart.shape[0]):
        if snapshot.captured[None] == 0:
            snapshot.pre_solve_is_warmstart[i_b] = constraint_state.is_warmstart[i_b]
