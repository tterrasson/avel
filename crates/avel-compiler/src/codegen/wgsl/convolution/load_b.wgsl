{
    let column = tile_column + local.x + {{offset}}u;
    var value: {{ty}} = 0.0;
    if (rb < {{reduction}}u && column < {{group_outputs}}u) {
        value = {{element}};
    }
    tile_b[local.y * {{bn}}u + local.x + {{offset}}u] = value;
}
