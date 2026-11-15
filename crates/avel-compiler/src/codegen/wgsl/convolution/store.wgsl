{
    let column = tile_column + local.x + {{offset}}u;
    if (m{{i}} < {{rows}}u && column < {{columns}}u) {
        let n = n{{i}};
        {{position}}
        let channel = {{channel}};
        {{store}}
    }
}
