var<workgroup> row_partials: array<{{ty}}, {{lanes}}>;

fn row_merge(value: {{ty}}, lane: u32) -> {{ty}} {
    row_partials[lane] = value;
    workgroupBarrier();
    for (var width = {{half}}u; width > 0u; width >>= 1u) {
        if (lane < width) {
            row_partials[lane] = {{combine}}(row_partials[lane], row_partials[lane + width]);
        }
        workgroupBarrier();
    }
    return row_partials[0];
}
