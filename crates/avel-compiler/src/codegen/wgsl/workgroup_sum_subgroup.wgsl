// The {{lanes}} invocations of the workgroup form a single subgroup.
fn workgroup_sum(value: f32, lane: u32) -> f32 {
    return subgroupAdd(value);
}

