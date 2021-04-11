def save_info(dir_name, args, model):
    with open(f"{dir_name}/info.txt", "w") as f:
        f.write(f"number of class: {args['n_classes']}\n")
        f.write(f"learning rate: {args['lr']}\n")
        f.write(f"batch size: {args['batch_size']}\n")
        f.write(f"total epochs: {args['epochs']}\n")
        f.write(f"penalty: {args['penalty']}\n")
        f.write(f"dropout rate: {args['dropout']}\n")
        f.write(f"decay rate: {args['decay']}\n")
        f.write(f"optimizer: {args['optimizer']}\n")
        f.write(f"backbone network: {args['backbone']}\n")
        f.write(f"training data: {args['train_data']}\n")
        f.write(f"validation data: {args['validation_data']}\n")
        f.write(f"used parameters: {args['use_param_folder']}\n")

        f.write("=============layer information===============\n")
        for layer in model.layers:
            f.write(f"{layer.name}: {layer.trainable}\n")

        f.write("=============================================\n")
