import wkv_layer
import mlp_layer

def create_block(layer_id, layers, config):
    batch_first = config["batch_first"]
    do_rearrange = config["do_rearrange"]

    mlp_config = config["mlp"]
    attn_config = config["attn"]
    
    attn_post, mlp_post = ("_f", "_g") if layer_id % 2 == 1 else ("_g", "_f")

    mlp = mlp_layer.create_transformer_layer(
        mlp_config["layer_name"],
        mlp_config.get("init_scale", 2. / layers),
        layer_id,
        layers,
        widening_factor=mlp_config.get("widening_factor", 4),
        name=mlp_config.get("name", f'l{layer_id}_mlp{mlp_post}'),
        activation_function=None,
        bias=mlp_config.get("bias", None)
    ) if mlp_config["layer_name"] in mlp_layer.TRANSFORMER_LAYERS else mlp_layer.create_rwkv_layer(
            mlp_config["layer_name"],
            mlp_config.get("init_scale", 2. / layers),
            layer_id,
            layers,
            widening_factor=mlp_config.get("widening_factor", 4),
            name=mlp_config.get("name", f'l{layer_id}_mlp{mlp_post}'),
            activation_function=None,
            bias=mlp_config.get("bias", None)
        )
    attn = wkv_layer.create_layer(
        attn_config.get("init_scale", 2. / layers),
        layer_id,
        layers,
        widening_factor=attn_config.get("widening_factor", 1),
        name=attn_config.get("name", f'l{layer_id}_attn{attn_post}'),
        batch_first=batch_first,
        do_rearrange=do_rearrange
    )
    if layer_id % 2 == 1:
        return attn, mlp
    else:
        return mlp, attn