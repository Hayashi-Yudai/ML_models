import argparse

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--n_classes', default=10, type=int)
	parser.add_argument('--epochs', default=100, type=int)
	parser.add_argument('--batch_size', default=10, type=int)
	parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'])
	parser.add_argument('--lr', default=0.1, type=float)
	parser.add_argument('--penalty', default=0.5, type=float)
	parser.add_argument('--enhance', default=65, type=float)
	parser.add_argument('--decay', default=5e-4, type=float)
	parser.add_argument('--backbone', default="ResNet50", choices=['VGG16', 'ResNet50'])
	parser.add_argument('--dropout', default=0.5, type=float)
	parser.add_argument('--train_data', default='./', type=str)
	parser.add_argument('--validation_data', default='./', type=str)
	parser.add_argument('--save_path', default='./params', type=str)
	parser.add_argument('--use_param_folder', default='', type=str)
	args = parser.parse_args()

	return args


def save_info(dir_name, args, model):
	with open(f"{dir_name}/info.txt", "w") as f:
		f.write(f"number of class: {args.n_classes}\n")
		f.write(f"learning rate: {args.lr}\n")
		f.write(f"batch size: {args.batch_size}\n")
		f.write(f"total epochs: {args.epochs}\n")
		f.write(f"penalty: {args.penalty}\n")
		f.write(f"dropout rate: {args.dropout}\n")
		f.write(f"decay rate: {args.decay}\n")
		f.write(f"optimizer: {args.optimizer}\n")
		f.write(f"backbone network: {args.backbone}\n")
		f.write(f"training data: {args.train_data}\n")
		f.write(f"validation data: {args.validation_data}\n")
		f.write(f"used parameters: {args.use_param_folder}\n")

		f.write("=============layer information===============\n")
		for layer in model.layers:
			f.write(f"{layer.name}: {layer.trainable}\n")
			
		f.write("=============================================\n")