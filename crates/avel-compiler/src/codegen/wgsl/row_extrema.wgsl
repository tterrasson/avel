struct RowState {
    value: {{ty}},
    index: u32,
}

fn combine(a: RowState, b: RowState) -> RowState {
    if (b.index == 0xffffffffu) { return a; }
    if (a.index == 0xffffffffu) { return b; }
    {{nan}}
    if (({{prefer}}) || (a.value == b.value && b.index < a.index)) { return b; }
    return a;
}
