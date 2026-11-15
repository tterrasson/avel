{
    var value: {{ty}} = 0.0;
    {{positions}}
    if ({{checks}}) {
        value = {{element}};
    }
    tile_a[local.x * {{bm}}u + local.y + {{offset}}u] = value;
}
