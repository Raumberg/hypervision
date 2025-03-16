import tensorrt as trt
import click

@click.command()
@click.option('--onnx', required=True, help='Path to the input ONNX file.')
@click.option('--engine', required=True, help='Path to save the output TensorRT engine file.')
def build_engine(onnx, engine):
    """
    Convert an ONNX model to a TensorRT engine.
    """
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    click.echo(f"Loading ONNX model from {onnx}...")
    with open(onnx, 'rb') as model:
        if not parser.parse(model.read()):
            click.echo("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                click.echo(parser.get_error(error))
            return
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB workspace
    
    # Set optimization profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)
    
    # Build engine
    click.echo("Building TensorRT engine...")
    engine_serialized = builder.build_serialized_network(network, config)
    if engine_serialized is None:
        click.echo("Failed to build engine.")
        return
    
    # Save engine
    click.echo(f"Saving TensorRT engine to {engine}...")
    with open(engine, "wb") as f:
        f.write(engine_serialized)
    
    click.echo("Engine built successfully!")

if __name__ == '__main__':
    build_engine()